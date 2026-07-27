"""処理ログの捕捉。

Web UI は処理の経過を画面に抜粋表示する。そのため処理の間だけルートロガーに
ハンドラを足し、抜けるときに元へ戻す。

文字起こしはワーカースレッドで走り、同時に複数走りうる。ルートロガーの
ハンドラは全スレッドの出力を受け取ってしまうため、既定では
**呼び出し元スレッドの出力だけ**を拾う。
"""

from __future__ import annotations

import io
import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from src.common.logging import LOG_FORMAT


@contextmanager
def capture_root_logs(
    level: int = logging.INFO, *, current_thread_only: bool = True
) -> Iterator[io.StringIO]:
    """ルートロガーの出力を文字列バッファへ複製する。

    Args:
        level: 捕捉する最低ログレベル。処理の間だけルートロガーへ設定する。
        current_thread_only: ``True`` なら、この関数を呼んだスレッドから出た
            ログだけを拾う。並走する別ジョブのログが混ざらないようにするため。

    Yields:
        捕捉したログを溜めるバッファ。``with`` を抜けたあとも読める。
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    if current_thread_only:
        thread_id = threading.get_ident()
        handler.addFilter(lambda record: record.thread == thread_id)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    try:
        yield buffer
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
