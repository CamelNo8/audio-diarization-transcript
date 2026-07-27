"""mlx-whisper（Apple Silicon / Metal 最適化）による文字起こし。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


def transcribe(audio_path: Path, model_id: str, language: str) -> dict[str, Any]:
    """mlx-whisper で音声全体を文字起こしする。

    Args:
        audio_path: 音声ファイルのパス。
        model_id: mlx-community のリポジトリ名。
        language: 言語コード（例: ``ja``）。

    Returns:
        ``{"segments": [{"start", "end", "text"}, ...]}``。

    Raises:
        ImportError: mlx-whisper が導入されていない場合（Apple Silicon 以外）。
    """
    import mlx_whisper  # 遅延インポート（Apple Silicon 以外には存在しない）

    logger.info(f"Running mlx-whisper transcription ({model_id})...")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        verbose=False,
        language=language,
    )
    return {"segments": result.get("segments", [])}
