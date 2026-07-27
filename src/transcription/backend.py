"""文字起こしバックエンドの選択と実行（公開 API）。

プラットフォームに応じて Whisper 実装を切り替える:

- macOS (Apple Silicon) … mlx-whisper（Metal 最適化）
- Windows / Linux (x86)  … faster-whisper（CTranslate2 CUDA）
- Linux aarch64 (GB10等) … transformers + torch（CUDA）に自動フォールバック
                           （CTranslate2 に aarch64 用 CUDA ビルドが無いため）

いずれのバックエンドも ``{"segments": [{"start", "end", "text"}, ...]}`` という
共通フォーマットを返すため、呼び出し側は実装差を意識しない。
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import torch

from src.common.logging import get_logger
from src.transcription import faster_whisper, mlx, transformers_backend
from src.transcription.model_ids import to_mlx_repo

logger = get_logger(__name__)

#: 明示指定として受け付けるバックエンド名。
BACKEND_NAMES = ("mlx", "faster", "transformers")


def is_apple_silicon() -> bool:
    """実行環境が Apple Silicon（mlx-whisper が使える）かを返す。"""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def resolve_backend(backend: str | None) -> str:
    """使用するバックエンド名を決定する。

    Args:
        backend: ``mlx`` / ``faster`` / ``transformers`` は明示指定として尊重する。
            ``auto`` / ``None`` / 未知の名前は実行環境から判定する。

    Returns:
        :data:`BACKEND_NAMES` のいずれか。
    """
    normalized = (backend or "auto").lower()
    if normalized in BACKEND_NAMES:
        return normalized
    return "mlx" if is_apple_silicon() else "faster"


def select_whisper_device(prefer: torch.device | None = None) -> str:
    """faster-whisper 用のデバイス文字列を返す。

    CTranslate2(faster-whisper) は cuda と cpu のみ対応（mps 非対応）。

    Args:
        prefer: 希望するデバイス。``cuda`` のときだけ尊重する。

    Returns:
        ``cuda`` または ``cpu``。
    """
    if prefer is not None and prefer.type == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_full(
    audio_path: Path,
    model_id: str,
    language: str = "ja",
    prefer_device: torch.device | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """音声全体を文字起こしする。

    Args:
        audio_path: 音声ファイルのパス。
        model_id: 品質キー（``large-v3`` 等）/ mlx リポジトリ / サイズ名のいずれか。
            選択したバックエンドに合わせて自動変換される。
        language: 言語コード。
        prefer_device: 希望するデバイス。
        backend: ``auto`` / ``mlx`` / ``faster`` / ``transformers``。

    Returns:
        ``{"segments": [{"start", "end", "text"}, ...]}``。
    """
    chosen = resolve_backend(backend)

    if chosen == "mlx":
        try:
            return mlx.transcribe(audio_path, to_mlx_repo(model_id), language)
        except ImportError:
            logger.warning(
                "mlx-whisper が見つからないため faster-whisper にフォールバックします。"
            )
            chosen = "faster"

    device = select_whisper_device(prefer_device)

    if chosen == "transformers":
        return transformers_backend.transcribe(audio_path, model_id, language, device)

    # chosen == "faster"。ただし CTranslate2 が CUDA 非対応ビルド（aarch64 等）で
    # GPU を使いたい場合は、CPU に落ちる代わりに transformers(torch CUDA) を使う。
    if device == "cuda" and not transformers_backend.ctranslate2_supports_cuda():
        logger.warning(
            "CTranslate2 が CUDA 非対応のため、"
            "transformers(torch CUDA) backend を使用します。"
        )
        return transformers_backend.transcribe(audio_path, model_id, language, device)

    return faster_whisper.transcribe(audio_path, model_id, language, device)
