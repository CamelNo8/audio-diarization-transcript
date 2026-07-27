"""ファイル操作の共通処理。"""

from __future__ import annotations

from pathlib import Path

from src.common.logging import get_logger

logger = get_logger(__name__)


def remove_quietly(path: Path) -> None:
    """一時ファイルを削除する。消せなくても処理は続行する。

    残った一時ファイルは次の実行で上書きされるだけなので、利用者へは通知せず
    ログにだけ残す。存在しないファイルを渡してもエラーにならない。
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.debug(f"一時ファイルを削除できませんでした: {path} ({e})")
