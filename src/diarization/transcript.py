"""文字起こしセグメントへの話者割り当てと、文字起こし CSV の行の組み立て。

Whisper が返すセグメントと、話者分離が返すクラスタは境界が一致しない。
各セグメントに最も長く重なるクラスタを選び、そのクラスタに割り当てられた
話者名を持たせる。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pyannote.core import Segment

from src.common.logging import get_logger
from src.common.timecode import format_time
from src.diarization.clusters import ClusterAssignments, dominant_cluster

logger = get_logger(__name__)

#: 文字起こし CSV の列。順序も含めて後段（Web UI・字幕生成）が依存する。
CSV_HEADER = ["start", "end", "speaker", "text", "cosine_distance"]

#: どのクラスタとも重ならなかったセグメントに付ける話者名。
_NO_CLUSTER_SPEAKER = "Unknown"


def build_rows(
    diarization, segments: List[Dict], assignments: ClusterAssignments
) -> List[List[str]]:
    """文字起こしセグメントから CSV の全行（ヘッダ込み）を組み立てる。

    Args:
        diarization: 話者分離の結果（``Annotation``）。
        segments: Whisper のセグメント。
        assignments: クラスタごとの照合結果。

    Returns:
        1行目がヘッダの二次元リスト。本文が空のセグメントは除く。
    """
    rows = [CSV_HEADER]
    for seg in segments:
        row = _build_row(diarization, seg, assignments)
        if row is not None:
            rows.append(row)
    return rows


def _build_row(
    diarization, seg: Dict, assignments: ClusterAssignments
) -> Optional[List[str]]:
    """1つのセグメントを CSV の1行にする（本文が空なら ``None``）。"""
    text = seg["text"].strip()
    if not text:
        return None

    cluster_id = dominant_cluster(diarization, Segment(seg["start"], seg["end"]))
    speaker, distance, candidates = _resolve_speaker(cluster_id, assignments)

    start_str = format_time(seg["start"])
    end_str = format_time(seg["end"])
    distance_str = f"{distance:.6f}" if distance is not None else ""
    _log_row(
        start_str, end_str, cluster_id or "", speaker, text, distance_str, candidates
    )
    return [start_str, end_str, speaker, text, distance_str]


def _resolve_speaker(cluster_id: Optional[str], assignments: ClusterAssignments):
    """クラスタIDから話者名・距離・全候補距離を引く。

    Returns:
        (話者名, 距離, 全候補距離) の組。クラスタが無い場合は
        :data:`_NO_CLUSTER_SPEAKER` と ``None`` を返す。
    """
    if cluster_id is None:
        return _NO_CLUSTER_SPEAKER, None, None
    return (
        assignments.speaker_mapping.get(cluster_id, cluster_id),
        assignments.distance_mapping.get(cluster_id),
        assignments.candidate_distance_mapping.get(cluster_id),
    )


def _log_row(
    start_str: str,
    end_str: str,
    cluster_id: str,
    speaker: str,
    text: str,
    distance_str: str,
    candidates: Optional[Dict[str, float]],
) -> None:
    """1行分の割り当て結果と候補距離をログに出す。"""
    logger.info(
        f"  [{start_str} - {end_str}] {speaker}: {text} "
        f"(cosine_distance={distance_str or 'N/A'})"
    )
    if candidates:
        candidates_str = ", ".join(
            f"{name}={dist:.6f}"
            for name, dist in sorted(candidates.items(), key=lambda item: item[1])
        )
    else:
        candidates_str = "N/A"
    logger.info(
        f"  [{start_str} - {end_str}] cluster={cluster_id} "
        f"speaker={speaker} candidates: {candidates_str}"
    )
