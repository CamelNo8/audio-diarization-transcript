"""Whisper のモデル指定を各バックエンドの表記へ変換する。

呼び出し側は品質キー（``large-v3`` 等）・mlx リポジトリ・HF リポジトリの
どれを渡してもよく、選択したバックエンドに合わせてここで正規化する。
"""

from __future__ import annotations

#: faster-whisper が解釈できるモデルサイズ名（長い名前から先にマッチさせる）
FASTER_WHISPER_SIZES = [
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

#: 判別できない指定に使うサイズ名。
DEFAULT_SIZE = "large-v3"

#: サイズ名 → mlx-community の Whisper リポジトリ
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

#: サイズ名 → transformers 用の HuggingFace リポジトリ
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

#: そのまま使う（変換しない）HuggingFace リポジトリの接頭辞。
_HF_FULL_REPO_PREFIXES = ("openai/", "distil-whisper/")


def to_faster_whisper_model(model_id: str) -> str:
    """モデル指定から faster-whisper が解釈できるサイズ名を取り出す。

    Args:
        model_id: 品質キー・mlx リポジトリ・HF リポジトリのいずれか。

    Returns:
        サイズ名。判別できなければ :data:`DEFAULT_SIZE`。
    """
    lowered = model_id.lower()
    for size in FASTER_WHISPER_SIZES:
        if size in lowered:
            return size
    return DEFAULT_SIZE


def to_mlx_repo(model_id: str) -> str:
    """モデル指定を mlx-whisper 用のリポジトリへ変換する。

    Args:
        model_id: 品質キー、または mlx リポジトリ／フルパス。

    Returns:
        mlx-community のリポジトリ名。``mlx`` を含む指定はそのまま返す。
    """
    if "mlx" in model_id.lower():
        return model_id
    return _MLX_REPO.get(to_faster_whisper_model(model_id), _MLX_REPO[DEFAULT_SIZE])


def to_hf_whisper_repo(model_id: str) -> str:
    """モデル指定を transformers 用の HuggingFace リポジトリへ変換する。

    Args:
        model_id: 品質キー、または ``openai/`` ``distil-whisper/`` で始まるリポジトリ。

    Returns:
        HuggingFace のリポジトリ名。フルリポジトリ指定はそのまま返す。
    """
    if model_id.lower().startswith(_HF_FULL_REPO_PREFIXES):
        return model_id
    return _HF_WHISPER_REPO.get(
        to_faster_whisper_model(model_id), _HF_WHISPER_REPO[DEFAULT_SIZE]
    )
