"""テンプレート描画。

Jinja2 の自動エスケープに依存するため、テンプレート側で ``| safe`` を
新たに使わない（規約7.5）。
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import TEMPLATES_DIR

#: 全ルートで共有する Jinja2 環境。
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def render_error(request: Request, message: str) -> HTMLResponse:
    """エラー表示用の htmx フラグメントを返す。

    Args:
        request: 現在のリクエスト（テンプレートの共通変数に使う）。
        message: 画面に出す文言。
    """
    return templates.TemplateResponse(
        request,
        "partials/error.html",
        {"message": message},
    )
