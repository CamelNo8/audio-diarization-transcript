"""FastAPI アプリの組み立て。

ルータの登録と、作業ディレクトリの用意だけを行う。処理の実体は
``src/web/routes/`` 以下の各モジュールにある。
"""

from __future__ import annotations

from fastapi import FastAPI

from src.web import jobs, storage
from src.web.routes import (
    databases,
    generation,
    matching,
    pages,
    transcription,
    unknowns,
)

#: ブラウザのタブと OpenAPI に出るアプリ名。
APP_TITLE = "音声→字幕 統合アプリ"


def create_app() -> FastAPI:
    """ルータを登録した FastAPI アプリを作る。"""
    storage.TEMP_DIR.mkdir(exist_ok=True)
    jobs.CLUSTERS_ROOT.mkdir(exist_ok=True)

    app = FastAPI(title=APP_TITLE)
    for module in (pages, transcription, matching, generation, databases, unknowns):
        app.include_router(module.router)
    return app


app = create_app()
