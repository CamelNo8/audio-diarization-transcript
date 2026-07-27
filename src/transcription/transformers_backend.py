"""transformers + torch による文字起こし。

faster-whisper(CTranslate2) は aarch64 では CUDA 非対応ビルドしか配布されていない。
そのため GPU で文字起こししたい aarch64 環境（例: NVIDIA GB10）では、torch CUDA を
使うこの実装にフォールバックする（同じ Whisper の重みなので精度は同等）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.common.logging import get_logger
from src.transcription.model_ids import to_hf_whisper_repo

logger = get_logger(__name__)

#: ASR パイプラインの初期化が重いため (repo, device, dtype) でキャッシュする。
_ASR_CACHE: dict[tuple, Any] = {}


def ctranslate2_supports_cuda() -> bool:
    """インストール済みの CTranslate2 が CUDA で実行可能かを返す。

    aarch64 の PyPI wheel は CPU 専用ビルドのため False になる
    （呼び出し側はこのバックエンドへ切り替える）。
    """
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def transcribe(
    audio_path: Path, model_id: str, language: str, device: str
) -> dict[str, Any]:
    """transformers の Whisper で音声全体を文字起こしする。

    Args:
        audio_path: 音声ファイルのパス。
        model_id: 品質キーまたは HuggingFace リポジトリ。
        language: 言語コード（例: ``ja``）。
        device: ``cuda`` または ``cpu``。

    Returns:
        ``{"segments": [{"start", "end", "text"}, ...]}``。
    """
    repo = to_hf_whisper_repo(model_id)
    asr = _get_asr(repo, device)

    logger.info(f"Running transformers Whisper transcription ({repo})...")
    # chunk_length_s を指定しない = Whisper 本来の逐次(sequential)ロングフォーム復号。
    # モデルが予測するタイムスタンプ境界で文単位に分割されるため、faster-whisper と
    # 同様の細かいセグメントが得られる（chunked 方式はまとめ過ぎて字幕が繋がる）。
    result = asr(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    return {"segments": _to_segments(result)}


def _get_asr(repo: str, device: str) -> Any:
    """キャッシュ済みの ASR パイプラインを返す。無ければ読み込んでキャッシュする。"""
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    cache_key = (repo, device, str(dtype))

    asr = _ASR_CACHE.get(cache_key)
    if asr is None:
        from transformers import pipeline  # 遅延インポート

        logger.info(
            f"Loading transformers Whisper ({repo}, device={device}, dtype={dtype})..."
        )
        asr = pipeline(
            "automatic-speech-recognition",
            model=repo,
            torch_dtype=dtype,
            device=0 if use_cuda else -1,
        )
        _ASR_CACHE[cache_key] = asr
    return asr


def _to_segments(result: dict) -> list[dict[str, Any]]:
    """パイプラインの出力を共通のセグメント形式へ整える。

    タイムスタンプが欠けたチャンクは、直前の終了時刻（無ければ 0 秒）で補う。
    """
    segments: list[dict[str, Any]] = []
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
    return segments
