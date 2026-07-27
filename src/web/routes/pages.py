"""画面の表示と生成物のダウンロード。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

import src.voice_db.registry as vdb
from src.common.filenames import is_safe_output_name
from src.web import storage
from src.web.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """3ステップの操作画面（トップページ）。"""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"databases": vdb.list_databases()},
    )


@router.get("/databases", response_class=HTMLResponse)
async def databases_page(request: Request):
    """声紋DB の管理画面。"""
    return templates.TemplateResponse(
        request,
        "databases.html",
        {"databases": vdb.list_databases(), "root": str(vdb.get_root())},
    )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """作業ディレクトリの生成物をダウンロードさせる。"""
    # ルーティング上ここへパス区切りは届かないが、多層防御として検査する
    if not is_safe_output_name(filename):
        return HTMLResponse("File not found", status_code=404)
    file_path = storage.temp_path(filename)
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("File not found", status_code=404)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )
