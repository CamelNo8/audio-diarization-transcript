"""フォーム値のパース。

HTML フォームの値は常に文字列で届く。空欄と入力ミスを同じ「未指定」として
扱い、ルート側で分岐を増やさないようにする。
"""

from __future__ import annotations

import os
from typing import Optional

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


def parse_opt_float(raw: str) -> Optional[float]:
    """秒数の入力をパースする。空 / 不正 / 負値は ``None``（未指定）。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_opt_int(raw: str) -> Optional[int]:
    """整数の入力をパースする。空 / 不正は ``None``（未指定）。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
