"""話者クラスタの照合結果の保持と、未知クラスタの抽出・永続化・再マッピング。

話者分離が返す「クラスタID」は音声内での話者の区別でしかないため、声紋DBと
照合して実名（または ``Unknown_NN``）へ対応づける。その対応と、照合に使った
代表区間・埋め込みを :class:`ClusterAssignments` にまとめて持ち回る。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyannote.audio.core.io import Audio
from pyannote.core import Segment

from src.common.audio import extract_audio
from src.common.logging import get_logger
from src.config import MIN_REPRESENTATIVE_SEC, UNKNOWN_LABEL_PREFIX

logger = get_logger(__name__)


@dataclass
class ClusterAssignments:
    """クラスタIDごとの照合結果と、照合に使った材料をまとめて持つ。"""

    #: クラスタID → 話者名（実名 または ``Unknown_NN``）
    speaker_mapping: Dict[str, str] = field(default_factory=dict)
    #: クラスタID → 最近傍の登録話者とのコサイン距離
    distance_mapping: Dict[str, Optional[float]] = field(default_factory=dict)
    #: クラスタID → 全登録話者との距離
    candidate_distance_mapping: Dict[str, Optional[Dict[str, float]]] = field(
        default_factory=dict
    )
    #: クラスタID → 代表音声区間
    segments: Dict[str, Segment] = field(default_factory=dict)
    #: クラスタID → 抽出済み埋め込み
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def set_speaker(
        self,
        cluster_id: str,
        speaker_name: str,
        distance: Optional[float],
        candidate_distances: Optional[Dict[str, float]],
    ) -> None:
        """クラスタの話者名・距離・全候補距離をまとめて記録する。"""
        self.speaker_mapping[cluster_id] = speaker_name
        self.distance_mapping[cluster_id] = distance
        self.candidate_distance_mapping[cluster_id] = candidate_distances

    def unknown_cluster_ids(self) -> List[str]:
        """まだ ``Unknown_NN`` のままのクラスタIDを、記録した順で返す。"""
        return [
            cid
            for cid, name in self.speaker_mapping.items()
            if name.startswith(UNKNOWN_LABEL_PREFIX)
        ]


def pick_representative_segment(timeline) -> Segment:
    """クラスタの代表となる音声区間（最長）を選ぶ。

    特徴量抽出の安定性を高めるため :data:`MIN_REPRESENTATIVE_SEC` 秒以上の区間を
    優先し、無ければ全区間の中の最長を使う。
    """
    segments = list(timeline)
    valid = [s for s in segments if s.duration >= MIN_REPRESENTATIVE_SEC]
    return max(valid or segments, key=lambda s: s.duration)


def collect_representative_segments(
    diarization, assignments: ClusterAssignments
) -> None:
    """全クラスタの代表区間を :class:`ClusterAssignments` に記録する。"""
    for cluster_id in diarization.labels():
        assignments.segments[cluster_id] = pick_representative_segment(
            diarization.label_timeline(cluster_id)
        )


def label_all_as_unknown(diarization, assignments: ClusterAssignments) -> None:
    """識別器が無い場合に、全クラスタを ``Unknown_NN`` として記録する。"""
    for i, cluster_id in enumerate(sorted(diarization.labels()), start=1):
        assignments.segments[cluster_id] = pick_representative_segment(
            diarization.label_timeline(cluster_id)
        )
        assignments.set_speaker(
            cluster_id, f"{UNKNOWN_LABEL_PREFIX}{i:02d}", None, None
        )


def identify_clusters(
    diarization, identifier, wav_path: Path, assignments: ClusterAssignments
) -> None:
    """各クラスタの代表区間から埋め込みを抽出し、登録話者と照合する。

    波形の切り出し・埋め込み抽出・照合はクラスタごとに独立しているため、
    どこかで失敗したクラスタだけを ``Unknown_NN`` として扱い、処理は続行する。
    """
    collect_representative_segments(diarization, assignments)

    waveforms = _crop_waveforms(wav_path, identifier, assignments)
    embeddings = _extract_embeddings(waveforms, identifier, assignments)

    for cluster_id, embedding in embeddings:
        try:
            name, distance, candidates = identifier.identify_speaker_with_distances(
                embedding
            )
        except Exception as e:
            _fallback_to_unknown(cluster_id, identifier, assignments, e)
            continue
        assignments.set_speaker(cluster_id, name, distance, candidates)
        _log_identification(cluster_id, name, distance)


def _crop_waveforms(
    wav_path: Path, identifier, assignments: ClusterAssignments
) -> List[Tuple[str, Any, int]]:
    """各クラスタの代表区間の波形を切り出す。"""
    audio_io = Audio()
    waveforms = []
    for cluster_id, segment in assignments.segments.items():
        try:
            waveform, sample_rate = audio_io.crop(str(wav_path), segment)
        except Exception as e:
            _fallback_to_unknown(cluster_id, identifier, assignments, e)
            continue
        waveforms.append((cluster_id, waveform, sample_rate))
    return waveforms


def _extract_embeddings(
    waveforms: List[Tuple[str, Any, int]], identifier, assignments: ClusterAssignments
) -> List[Tuple[str, np.ndarray]]:
    """切り出した波形から埋め込みを抽出する。"""
    embeddings = []
    for cluster_id, waveform, sample_rate in waveforms:
        try:
            embedding = identifier.get_embedding_from_waveform(waveform, sample_rate)
        except Exception as e:
            _fallback_to_unknown(cluster_id, identifier, assignments, e)
            continue
        assignments.embeddings[cluster_id] = embedding
        embeddings.append((cluster_id, embedding))
    return embeddings


def _fallback_to_unknown(
    cluster_id: str, identifier, assignments: ClusterAssignments, error: Exception
) -> None:
    """照合できなかったクラスタに ``Unknown_NN`` を割り当てる。"""
    logger.warning(f"Failed to identify speaker for cluster {cluster_id}: {error}")
    assignments.set_speaker(cluster_id, identifier._next_unknown_name(), None, None)


def _log_identification(cluster_id: str, name: str, distance: Optional[float]) -> None:
    """照合結果を1行でログに出す。"""
    if distance is None:
        logger.info(f"Cluster '{cluster_id}' identified as -> {name}")
    else:
        logger.info(
            f"Cluster '{cluster_id}' identified as -> {name} "
            f"(cosine_distance={distance:.6f})"
        )


def dominant_cluster(diarization, segment: Segment) -> Optional[str]:
    """指定区間と最も長く重なるクラスタIDを返す。重なりが無ければ ``None``。"""
    durations: Dict[str, float] = {}
    for cluster_segment, _, cluster_id in diarization.itertracks(yield_label=True):
        overlap = segment & cluster_segment
        if overlap:
            durations[cluster_id] = durations.get(cluster_id, 0.0) + overlap.duration
    if not durations:
        return None
    return max(durations, key=durations.get)


def persist_unknown_clusters(
    assignments: ClusterAssignments, wav_path: Optional[Path], output_dir: Path
) -> List[Dict[str, Any]]:
    """``Unknown_NN`` のクラスタの代表音声を書き出し、メタ情報を返す。

    Web UI で事後ラベル付けするために使う。

    Args:
        assignments: 照合結果。
        wav_path: 切り出し元の WAV。``None`` や不在なら何もしない。
        output_dir: 代表音声の出力先。

    Returns:
        各要素が ``cluster_id`` / ``unknown_label`` / ``distance`` /
        ``candidate_distances`` / ``clip_filename`` / ``segment_start`` /
        ``segment_end`` を持つ辞書のリスト。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if wav_path is None or not Path(wav_path).exists():
        logger.warning("一時 WAV が無いため Unknown クラスタを永続化できません。")
        return []

    result = []
    for cluster_id in assignments.unknown_cluster_ids():
        entry = _persist_one_unknown(assignments, wav_path, output_dir, cluster_id)
        if entry is not None:
            result.append(entry)
    return result


def _persist_one_unknown(
    assignments: ClusterAssignments,
    wav_path: Path,
    output_dir: Path,
    cluster_id: str,
) -> Optional[Dict[str, Any]]:
    """1クラスタ分の代表音声を書き出し、メタ情報を返す。失敗時は ``None``。"""
    segment = assignments.segments.get(cluster_id)
    if segment is None:
        return None

    clip_filename = f"clip_{cluster_id}.wav"
    try:
        extract_audio(
            wav_path,
            output_dir / clip_filename,
            start=segment.start,
            end=segment.end,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"クラスタ {cluster_id} の音声切り出し失敗: {e.stderr}")
        return None

    return {
        "cluster_id": cluster_id,
        "unknown_label": assignments.speaker_mapping[cluster_id],
        "distance": assignments.distance_mapping.get(cluster_id),
        "candidate_distances": assignments.candidate_distance_mapping.get(cluster_id),
        "clip_filename": clip_filename,
        "segment_start": float(segment.start),
        "segment_end": float(segment.end),
    }


def recompute_distances_for_cluster(
    identifier, assignments: ClusterAssignments, cluster_id: str
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """対話的に登録したクラスタについて、全登録話者との距離を取り直す。

    candidate_distances を埋めて CSV 出力で N/A にならないようにするのが目的。

    Returns:
        (最短距離, 全候補距離) の組。求められない場合は ``(None, None)``。
    """
    if identifier is None:
        return None, None
    embedding = assignments.embeddings.get(cluster_id)
    if embedding is None:
        return None, None
    try:
        _name, distance, candidates = identifier.identify_speaker_with_distances(
            embedding
        )
    except Exception as e:
        logger.warning(f"クラスタ {cluster_id} の距離再計算に失敗: {e}")
        return None, None
    return distance, candidates


def remap_remaining_unknowns(
    identifier, assignments: ClusterAssignments
) -> List[Tuple[str, str, Optional[float]]]:
    """新たに話者を登録した後、残りの ``Unknown_NN`` を再照合する。

    閾値以下でヒットしたものだけ話者名を更新する。再照合の結果また別の
    ``Unknown_NN`` が払い出された場合は、もとのラベルを維持する。

    Returns:
        更新した (クラスタID, 話者名, 距離) のリスト。
    """
    if identifier is None:
        return []

    remapped = []
    for cluster_id in assignments.unknown_cluster_ids():
        embedding = assignments.embeddings.get(cluster_id)
        if embedding is None:
            continue
        try:
            name, distance, candidates = identifier.identify_speaker_with_distances(
                embedding
            )
        except Exception as e:
            logger.warning(f"クラスタ {cluster_id} の再照合に失敗: {e}")
            continue

        if name.startswith(UNKNOWN_LABEL_PREFIX):
            continue

        assignments.set_speaker(cluster_id, name, distance, candidates)
        remapped.append((cluster_id, name, distance))
    return remapped
