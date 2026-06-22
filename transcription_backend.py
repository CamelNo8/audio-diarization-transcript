"""文字起こしバックエンドの抽象化。

プラットフォームに応じて Whisper 実装を切り替える:

- macOS (Apple Silicon) … mlx-whisper（Metal 最適化）
- Windows / Linux (x86)  … faster-whisper（CTranslate2 CUDA）
- Linux aarch64 (GB10等) … transformers + torch（CUDA）に自動フォールバック
                           （CTranslate2 に aarch64 用 CUDA ビルドが無いため）

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
    if normalized in ("mlx", "faster", "transformers"):
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
        except Exception as e:
            err = str(e).lower()
            if device == "cuda" and "not compiled with cuda" in err:
                # CTranslate2 が CUDA 非対応ビルド（aarch64/ARM の PyPI wheel は CPU 専用）。
                # GPU が使えないので CPU(int8) にフォールバックする。
                logging.warning(
                    "CTranslate2 が CUDA 非対応のため、faster-whisper を CPU(int8) で実行します。"
                )
                device, compute_type = "cpu", "int8"
                cache_key = (fw_model_name, device, compute_type)
                model = _FW_MODEL_CACHE.get(cache_key) or WhisperModel(
                    fw_model_name, device=device, compute_type=compute_type
                )
            elif device == "cuda":
                # CUDA 上で float16 が使えない / cuDNN 不整合などのフォールバック
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


# ---- transformers (torch) backend -------------------------------------------
# faster-whisper(CTranslate2) は aarch64 では CUDA 非対応ビルドしか配布されていない。
# そのため GPU で文字起こししたい aarch64 環境（例: NVIDIA GB10）では、torch CUDA を
# 使う transformers 実装にフォールバックする（同じ Whisper large-v3 重みなので精度は同等）。

_HF_WHISPER_REPO = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "large-v1": "openai/whisper-large-v1",
    "large": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
    "base": "openai/whisper-base",
    "tiny": "openai/whisper-tiny",
    "distil-large-v3": "distil-whisper/distil-large-v3",
    "distil-large-v2": "distil-whisper/distil-large-v2",
}


def _to_hf_whisper_repo(model_id: str) -> str:
    """品質キー（"large-v3" 等）を transformers 用の HF リポジトリへ変換する。
    既に ``openai/`` や ``distil-whisper/`` のフルリポジトリ指定ならそのまま使う。
    """
    if model_id.lower().startswith(("openai/", "distil-whisper/")):
        return model_id
    return _HF_WHISPER_REPO.get(_to_faster_whisper_model(model_id), "openai/whisper-large-v3")


def _ctranslate2_supports_cuda() -> bool:
    """インストール済みの CTranslate2 が CUDA で実行可能か（CUDA ビルド かつ GPU あり）。

    aarch64 の PyPI wheel は CPU 専用ビルドのため 0 を返す → transformers へ切り替える。
    """
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# transformers ASR パイプラインは初期化が重いため (repo, device, dtype) でキャッシュ。
_HF_ASR_CACHE: Dict[tuple, Any] = {}


def _transcribe_transformers(
    audio_path: Path, model_id: str, language: str, device: str
) -> Dict[str, Any]:
    from transformers import pipeline  # 遅延インポート

    repo = _to_hf_whisper_repo(model_id)
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32

    cache_key = (repo, device, str(dtype))
    asr = _HF_ASR_CACHE.get(cache_key)
    if asr is None:
        logging.info(
            f"Loading transformers Whisper ({repo}, device={device}, dtype={dtype})..."
        )
        asr = pipeline(
            "automatic-speech-recognition",
            model=repo,
            torch_dtype=dtype,
            device=0 if use_cuda else -1,
        )
        _HF_ASR_CACHE[cache_key] = asr

    logging.info(f"Running transformers Whisper transcription ({repo})...")
    # chunk_length_s を指定しない = Whisper 本来の逐次(sequential)ロングフォーム復号。
    # モデルが予測するタイムスタンプ境界で文単位に分割されるため、faster-whisper と
    # 同様の細かいセグメントが得られる（chunked 方式はまとめ過ぎて字幕が繋がる）。
    result = asr(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={"language": language, "task": "transcribe"},
    )

    segments: List[Dict[str, Any]] = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            start = segments[-1]["end"] if segments else 0.0
        if end is None:
            end = start
        segments.append(
            {"start": float(start), "end": float(end), "text": chunk.get("text", "")}
        )
    if not segments and result.get("text"):
        segments = [{"start": 0.0, "end": 0.0, "text": result["text"]}]
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
            chosen = "faster"

    device = select_whisper_device(prefer_device)

    if chosen == "transformers":
        return _transcribe_transformers(audio_path, model_id, language, device)

    # chosen == "faster"。ただし CTranslate2 が CUDA 非対応ビルド（aarch64 等）で
    # GPU を使いたい場合は、CPU に落ちる代わりに transformers(torch CUDA) を使う。
    if device == "cuda" and not _ctranslate2_supports_cuda():
        logging.warning(
            "CTranslate2 が CUDA 非対応のため、transformers(torch CUDA) backend を使用します。"
        )
        return _transcribe_transformers(audio_path, model_id, language, device)

    return _transcribe_faster_whisper(audio_path, model_id, language, device)
