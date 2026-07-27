"""話者照合器の用意。

Step 1（文字起こし）と未知話者のラベル付けの両方で、声紋DB の中身を
すべて登録した照合器が要る。手順が同じなのでここにまとめる。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.diarization.registry import (
    collect_registry_files,
    get_cached_speaker_identifier,
)
from src.web.errors import WebInputError


def resolve_hf_token(override: str) -> str:
    """Hugging Face トークンを決める。詳細設定の入力を環境変数より優先する。

    Raises:
        WebInputError: どちらにも設定が無い場合。
    """
    token = override or os.getenv("HF_TOKEN", "")
    if not token:
        raise WebInputError(
            "Hugging Face Token が設定されていません。"
            ".env の HF_TOKEN または詳細設定で指定してください。"
        )
    return token


def load_identifier(
    registry_dir: Path, *, model_name: str, hf_token: str, threshold: float
) -> Any:
    """声紋DB 内の全話者を登録した照合器を返す。

    Args:
        registry_dir: 声紋ファイルを置いた DB ディレクトリ。
        model_name: エンベディング抽出モデル。
        hf_token: Hugging Face トークン。
        threshold: 同一話者と判定するコサイン距離の閾値。
    """
    registry_paths = collect_registry_files(registry_dir)
    identifier = get_cached_speaker_identifier(
        model_name=model_name,
        hf_token=hf_token,
        threshold=threshold,
    )
    for name, path in registry_paths.items():
        identifier.register_speaker(name, path)
    return identifier
