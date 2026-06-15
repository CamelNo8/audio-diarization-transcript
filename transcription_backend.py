"""文字起こしバックエンドの抽象化。

プラットフォームに応じて Whisper 実装を切り替える:

- macOS (Apple Silicon) … mlx-whisper（Metal 最適化）
- Windows / Linux        … faster-whisper（CUDA 対応・CPU フォールバック）

いずれのバックエンドも ``{"segments": [{"start", "end", "text"}, ...]}`` という
共通フォーマットを返すため、呼び出し側（AudioProcessor）は実装差を意識しない。
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


# faster-whisper が解釈できるモデルサイズ名（長い名前から先にマッチさせる）
_FW_SIZES = [
    "large-v3-turbo",
    "large-v3",
    "large-v2",
    "large-v1",
    "distil-large-v3",
    "distil-large-v2",
    "turbo",
    "large",
    "medium",
    "small",
    "base",
    "tiny",
]


def _to_faster_whisper_model(model_id: str) -> str:
    """mlx 向けモデルID（例: ``mlx-community/whisper-large-v3-mlx``）や品質キー
    （例: ``large-v3``）を faster-whisper が解釈できるサイズ名へ変換する。
    判別できなければ ``large-v3``。
    """
    lowered = model_id.lower()
    for size in _FW_SIZES:
        if size in lowered:
            return size
    return "large-v3"


# 品質キー / サイズ名 → mlx-community の Whisper リポジトリ
_MLX_REPO = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def _to_mlx_repo(model_id: str) -> str:
    """品質キー（例: ``large-v3``）を mlx-whisper 用のリポジトリへ変換する。
    既に mlx リポジトリ/フルパスが指定されている場合はそのまま使う。
    """
    if "mlx" in model_id.lower():
        return model_id
    size = _to_faster_whisper_model(model_id)
    return _MLX_REPO.get(size, "mlx-community/whisper-large-v3-mlx")


def _resolve_backend(backend: Optional[str]) -> str:
    """使用するバックエンド名（"mlx" / "faster"）を決定する。

    backend が "mlx" / "faster" の場合は明示指定として尊重する。
    "auto" / None の場合は実行環境から判定する。
    """
    normalized = (backend or "auto").lower()
    if normalized in ("mlx", "faster"):
        return normalized
    return "mlx" if is_apple_silicon() else "faster"


# faster-whisper のモデルは初期化が重いため、(model, device, compute_type) でキャッシュする。
_FW_MODEL_CACHE: Dict[tuple, Any] = {}


def _transcribe_mlx(audio_path: Path, model_id: str, language: str) -> Dict[str, Any]:
    import mlx_whisper  # 遅延インポート（Apple Silicon 以外には存在しない）

    logging.info(f"Running mlx-whisper transcription ({model_id})...")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        verbose=False,
        language=language,
    )
    return {"segments": result.get("segments", [])}


def _transcribe_faster_whisper(
    audio_path: Path, model_id: str, language: str, device: str
) -> Dict[str, Any]:
    from faster_whisper import WhisperModel  # 遅延インポート

    fw_model_name = _to_faster_whisper_model(model_id)
    compute_type = "float16" if device == "cuda" else "int8"

    cache_key = (fw_model_name, device, compute_type)
    model = _FW_MODEL_CACHE.get(cache_key)
    if model is None:
        logging.info(
            f"Loading faster-whisper model ({fw_model_name}, device={device}, "
            f"compute_type={compute_type})..."
        )
        try:
            model = WhisperModel(fw_model_name, device=device, compute_type=compute_type)
        except Exception:
            # CUDA 上で float16 が使えない / cuDNN 不整合などのフォールバック
            if device == "cuda":
                logging.warning(
                    "faster-whisper の float16/CUDA 初期化に失敗したため "
                    "compute_type=int8_float16 で再試行します。"
                )
                compute_type = "int8_float16"
                cache_key = (fw_model_name, device, compute_type)
                model = WhisperModel(
                    fw_model_name, device=device, compute_type=compute_type
                )
            else:
                raise
        _FW_MODEL_CACHE[cache_key] = model

    logging.info(f"Running faster-whisper transcription ({fw_model_name})...")
    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=False,
    )
    segments: List[Dict[str, Any]] = [
        {"start": float(s.start), "end": float(s.end), "text": s.text}
        for s in segments_iter
    ]
    return {"segments": segments}


def select_whisper_device(prefer: Optional[torch.device] = None) -> str:
    """faster-whisper 用のデバイス文字列（"cuda" / "cpu"）を返す。

    CTranslate2(faster-whisper) は cuda と cpu のみ対応（mps 非対応）。
    """
    if prefer is not None and prefer.type == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_full(
    audio_path: Path,
    model_id: str,
    language: str = "ja",
    prefer_device: Optional[torch.device] = None,
    backend: str = "auto",
) -> Dict[str, Any]:
    """音声全体を文字起こしし、``{"segments": [...]}`` を返す。

    backend:
      - "auto"  … 実行環境で判定（Apple Silicon は mlx、それ以外は faster-whisper）
      - "mlx"   … mlx-whisper を強制（未導入時は faster-whisper にフォールバック）
      - "faster"… faster-whisper を強制

    model_id には品質キー（"large-v3" 等）/ mlx リポジトリ /
    faster-whisper サイズ名のいずれを渡してもよく、選択したバックエンドに
    合わせて自動変換される。
    """
    chosen = _resolve_backend(backend)

    if chosen == "mlx":
        try:
            return _transcribe_mlx(audio_path, _to_mlx_repo(model_id), language)
        except ImportError:
            logging.warning(
                "mlx-whisper が見つからないため faster-whisper にフォールバックします。"
            )

    device = select_whisper_device(prefer_device)
    return _transcribe_faster_whisper(audio_path, model_id, language, device)
