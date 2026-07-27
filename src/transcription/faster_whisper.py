"""faster-whisper（CTranslate2）による文字起こし。

CUDA での初期化に失敗する環境（CTranslate2 が CPU 専用ビルド、cuDNN 不整合など）
では compute_type やデバイスを落として再試行する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.transcription.model_ids import to_faster_whisper_model

logger = get_logger(__name__)

#: モデルの初期化が重いため (モデル名, デバイス, compute_type) でキャッシュする。
_MODEL_CACHE: dict[tuple, Any] = {}

#: 復号時のビーム幅。
_BEAM_SIZE = 5


def transcribe(
    audio_path: Path, model_id: str, language: str, device: str
) -> dict[str, Any]:
    """faster-whisper で音声全体を文字起こしする。

    Args:
        audio_path: 音声ファイルのパス。
        model_id: 品質キーまたはサイズ名。
        language: 言語コード（例: ``ja``）。
        device: ``cuda`` または ``cpu``。

    Returns:
        ``{"segments": [{"start", "end", "text"}, ...]}``。
    """
    model_name = to_faster_whisper_model(model_id)
    model = _get_model(model_name, device)

    logger.info(f"Running faster-whisper transcription ({model_name})...")
    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=_BEAM_SIZE,
        vad_filter=False,
    )
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text}
        for s in segments_iter
    ]
    return {"segments": segments}


def _get_model(model_name: str, device: str) -> Any:
    """キャッシュ済みのモデルを返す。無ければ読み込んでキャッシュする。"""
    compute_type = "float16" if device == "cuda" else "int8"
    requested_key = (model_name, device, compute_type)
    model = _MODEL_CACHE.get(requested_key)
    if model is not None:
        return model

    logger.info(
        f"Loading faster-whisper model ({model_name}, device={device}, "
        f"compute_type={compute_type})..."
    )
    model, actual_key = _load_model(model_name, device, compute_type)
    # 実際に使えた設定と、要求された設定の両方にキャッシュを張る。
    # 要求側を張らないと、フォールバックが起きた環境では毎回 CUDA 初期化を
    # 試みては失敗し直すことになる。
    _MODEL_CACHE[actual_key] = model
    _MODEL_CACHE[requested_key] = model
    return model


def _load_model(model_name: str, device: str, compute_type: str) -> tuple[Any, tuple]:
    """モデルを読み込む。CUDA で失敗した場合は設定を落として再試行する。

    Returns:
        (モデル, 実際に使った設定のキャッシュキー) の組。

    Raises:
        Exception: CPU での読み込みに失敗した場合はそのまま送出する。
    """
    from faster_whisper import WhisperModel  # 遅延インポート

    try:
        return (
            WhisperModel(model_name, device=device, compute_type=compute_type),
            (model_name, device, compute_type),
        )
    except Exception as e:
        if device != "cuda":
            raise
        if "not compiled with cuda" in str(e).lower():
            # CTranslate2 が CUDA 非対応ビルド
            # （aarch64/ARM の PyPI wheel は CPU 専用）。
            # GPU が使えないので CPU(int8) にフォールバックする。
            logger.warning(
                "CTranslate2 が CUDA 非対応のため、"
                "faster-whisper を CPU(int8) で実行します。"
            )
            key = (model_name, "cpu", "int8")
            model = _MODEL_CACHE.get(key) or WhisperModel(
                model_name, device="cpu", compute_type="int8"
            )
            return model, key
        # CUDA 上で float16 が使えない / cuDNN 不整合などのフォールバック
        logger.warning(
            "faster-whisper の float16/CUDA 初期化に失敗したため "
            "compute_type=int8_float16 で再試行します。"
        )
        key = (model_name, device, "int8_float16")
        return WhisperModel(model_name, device=device, compute_type="int8_float16"), key
