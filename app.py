"""Web アプリのエントリポイント。

実体は :mod:`src.web.application` にある。ここでは import より前に必要な
環境設定（ロギング・``.env`` 読み込み・Hugging Face のオフライン指定）だけを行う。

``spark-up.sh`` が ``python app.py`` で直接起動するため、``uvicorn app:app``
と ``python app.py`` の両方が動く形を保つ。
"""

from __future__ import annotations

import os

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
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.web.application import app  # noqa: E402

#: ``uvicorn app:app`` から名前で参照されるため、明示的に再公開する。
__all__ = ["app"]

#: 開発用に待ち受けるアドレスとポート。
HOST = "127.0.0.1"
PORT = 8000

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
