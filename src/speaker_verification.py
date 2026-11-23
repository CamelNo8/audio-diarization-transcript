"""窓口2: 音声特徴データベースと、話者分離で得た区間を照合する。

入口は2つ。

- :func:`load_verifier` … 音声特徴DB の全話者を登録した照合器を作る
- :func:`verify_segments` … 話者分離の各クラスタに話者名を割り当てる

**区間（diarization）はここでは作らない。** 文字起こし側
（:mod:`src.transcription`）が1回だけ話者分離した結果を受け取る。
同じ音声をここでもう一度分離すると、一番重い工程が二重に走るため。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.common.logging import get_logger
from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_SPEAKER_THRESHOLD
from src.diarization import clusters
from src.diarization.clusters import ClusterAssignments
from src.diarization.registry import (
    collect_registry_files,
    get_cached_speaker_identifier,
)
from src.diarization.speaker_identifier import SpeakerIdentifier

logger = get_logger(__name__)


def load_verifier(
    db_dir: Path,
    *,
    hf_token: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    threshold: float = DEFAULT_SPEAKER_THRESHOLD,
) -> SpeakerIdentifier:
    """音声特徴データベース内の全話者を登録した照合器を返す。

    話者名はファイル名（拡張子なし）をそのまま使う。
    モデルの初期化は重いので、モデル名ごとにプロセス内で使い回す。

    Args:
        db_dir: 音声特徴ファイルを置いたデータベースのディレクトリ。
        hf_token: Hugging Face トークン。
        model_name: 音声特徴（埋め込み）の抽出モデル。
        threshold: 同一話者と判定するコサイン距離の閾値。

    Returns:
        登録済みの照合器。
    """
    identifier = get_cached_speaker_identifier(
        model_name=model_name,
        hf_token=hf_token,
        threshold=threshold,
    )
    registry_paths = collect_registry_files(db_dir)
    logger.info(
        f"登録話者の特徴量を抽出しています... ({len(registry_paths)} 名: {db_dir})"
    )
    for name, path in registry_paths.items():
        identifier.register_speaker(name, path)
    return identifier


def verify_segments(
    diarization,
    identifier: Optional[SpeakerIdentifier],
    wav_path: Optional[Path],
    assignments: Optional[ClusterAssignments] = None,
) -> ClusterAssignments:
    """各クラスタの代表区間を照合し、話者名を割り当てる。

    Args:
        diarization: pyannote の話者分離結果。
        identifier: :func:`load_verifier` が返した照合器。
            ``None`` なら照合せず、全クラスタを ``Unknown_NN`` にする。
        wav_path: 代表区間を切り出す元の WAV。
        assignments: 書き込み先。``None`` なら新しく作る。

    Returns:
        クラスタIDごとの話者名・距離を持つ割り当て結果。
    """
    assignments = assignments if assignments is not None else ClusterAssignments()
    if identifier is None:
        clusters.label_all_as_unknown(diarization, assignments)
        return assignments

    clusters.identify_clusters(diarization, identifier, wav_path, assignments)
    return assignments
