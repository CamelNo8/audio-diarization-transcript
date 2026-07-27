"""利用者が指定した出力ファイル名の検証。

Web のフォームから受け取る出力ファイル名をそのままディレクトリと結合すると、
``../`` や絶対パスで作業ディレクトリの外へ書き出せてしまう
（``pathlib`` の ``/`` は右辺が絶対パスなら左辺を捨てる）。
ここを通した名前だけをパス結合に使う（規約7.1）。

DB名・話者名の検証は :func:`src.voice_db.registry.sanitize_name` が担う。
あちらは ``.`` を含む名前を拒否するためファイル名には使えない。
"""

from __future__ import annotations

from pathlib import Path

from src.common.logging import get_logger

logger = get_logger(__name__)

#: ディレクトリを指してしまう名前。
_RESERVED_NAMES = (".", "..")


def is_safe_output_name(raw: str) -> bool:
    """単一のファイル名として安全かを返す。

    Args:
        raw: 検証する文字列（前後の空白は無視する）。

    Returns:
        パス区切りを含まず、ディレクトリを指さない名前なら True。
    """
    name = (raw or "").strip()
    if not name or name in _RESERVED_NAMES:
        return False
    if "/" in name or "\\" in name:
        return False
    # 絶対パスやドライブ指定など、Path が単一名として扱わないものを除く
    return Path(name).name == name


def safe_output_name(raw: str, default: str) -> str:
    """出力ファイル名を検証し、安全でなければ既定名を返す。

    Args:
        raw: 利用者が指定したファイル名。
        default: 安全でなかった場合に使う既定のファイル名。

    Returns:
        パス結合に使ってよいファイル名。

    Note:
        不正な名前そのものはログに出さない（そのまま記録すると
        指定されたパスが痕跡として残るため）。
    """
    if is_safe_output_name(raw):
        return raw.strip()
    logger.warning(f"出力ファイル名が不正なため既定名を使用します: {default}")
    return default
