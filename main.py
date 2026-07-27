"""CLI のエントリポイント。

実処理は :mod:`src.cli.runner` にある。ここでは実行前に必要な環境設定
（ロギング・``.env`` 読み込み・Hugging Face のオフライン指定）だけを行う。
"""

from __future__ import annotations

import os
import sys

from src.common.logging import configure_logging, get_logger

# .env 読み込み時の警告も拾えるよう、最初にロギングを設定する
configure_logging()
logger = get_logger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.warning(
        "python-dotenv is not installed. "
        "Environment variables from .env will not be loaded automatically."
    )
    logger.warning("To install: uv pip install python-dotenv")

# モデルはローカルキャッシュから読む（毎回のネットワーク確認を避けるため）。
# 未取得のモデルが必要な場合だけ src/diarization/registry.py が一時的に解除する。
os.environ["HF_HUB_OFFLINE"] = "1"

from src.cli.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
