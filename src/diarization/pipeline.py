"""pyannote の話者分離パイプラインの読み込みとデバイス選択。

パイプラインの初期化は重いため、(モデルID, トークン) 単位でプロセス内に
キャッシュする。
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch

from src.diarization.torch_compat import patch_torch_load

patch_torch_load()

from pyannote.audio import Pipeline  # noqa: E402

from src.common.logging import get_logger  # noqa: E402

# Hugging Face トークンに関する FutureWarning を抑制
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

logger = get_logger(__name__)

#: (モデルID, HFトークン) → 読み込み済みパイプライン
_PIPELINE_CACHE: dict[tuple[str, str], Pipeline] = {}


def select_device() -> torch.device:
    """利用できる中で最も速いデバイスを返す（CUDA > MPS > CPU）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cached_pipeline(model_id: str, hf_token: str, device: torch.device) -> Pipeline:
    """話者分離パイプラインを取得し、指定デバイスへ載せて返す。

    Args:
        model_id: pyannote のモデルID。
        hf_token: Hugging Face アクセストークン。
        device: パイプラインを載せるデバイス。

    Returns:
        読み込み済みのパイプライン（2回目以降はキャッシュを再利用）。
    """
    cache_key = (model_id, hf_token)
    pipeline = _PIPELINE_CACHE.get(cache_key)
    if pipeline is None:
        logger.info(f"Loading Pyannote pipeline ({model_id})...")
        pipeline = load_pipeline_with_auth(model_id, hf_token)
        _PIPELINE_CACHE[cache_key] = pipeline
    pipeline.to(device)
    return pipeline


def load_pipeline_with_auth(model_id: str, hf_token: str) -> Pipeline:
    """新旧 huggingface_hub / pyannote-audio 両対応でパイプラインを読み込む。

    ``token`` と ``use_auth_token`` のどちらを受け付けるか、``local_files_only``
    をサポートするかが pyannote のバージョンで変わるため、組み合わせを順に試す。

    Raises:
        RuntimeError: すべての組み合わせが失敗した場合。
    """
    last_error: Optional[Exception] = None
    # 優先度順: 引数が新しい (token) / オフライン優先 → 引数が古い / オンライン
    attempts = [
        {"token": hf_token, "local_files_only": True},
        {"token": hf_token},
        {"use_auth_token": hf_token, "local_files_only": True},
        {"use_auth_token": hf_token},
    ]
    for kwargs in attempts:
        try:
            return Pipeline.from_pretrained(model_id, **kwargs)
        except TypeError as e:
            # 引数自体が受け付けられない → 次の組み合わせへ
            last_error = e
        except Exception as e:
            # ネットワーク不通等 → 次の組み合わせへ
            last_error = e
    raise RuntimeError(f"Pipeline.from_pretrained に失敗しました: {last_error}")


def unwrap_diarization(result: Any) -> Any:
    """パイプラインの出力から ``Annotation`` を取り出す。

    pyannote-audio 4.x は ``DiarizeOutput`` を返す（3.x は ``Annotation`` を直接返す）。
    以降は ``Annotation`` の API（labels / label_timeline / itertracks）を使う。
    """
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    return result
