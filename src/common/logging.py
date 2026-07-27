"""ロギングの共通設定。

エントリポイント（CLI / Web / バッチ）の入口で :func:`configure_logging` を
一度だけ呼び、各モジュールは :func:`get_logger` で取得したロガーへ出力する。
``print`` によるデバッグ出力は使わない（規約8.6）。

標準出力へ流すのは、Web UI がマッチング処理を別プロセスで起動し、その
標準出力を画面に転載しているため（stderr だと拾えない）。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

#: 全エントリポイント共通のログ書式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """ルートロガーを設定する。

    既にハンドラが設定済みの場合は何もしない（``logging.basicConfig`` の仕様）。

    Args:
        level: 出力する最低ログレベル。
        stream: 出力先。省略時は標準出力。
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=stream if stream is not None else sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """モジュール用のロガーを返す。

    Args:
        name: 通常は呼び出し元の ``__name__``。

    Returns:
        指定名のロガー。
    """
    return logging.getLogger(name)
