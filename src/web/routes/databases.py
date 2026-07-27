"""声紋DB の管理 API（htmx フラグメントを返す）。"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import src.voice_db.registry as vdb
from src.common.audio import extract_audio
from src.common.logging import get_logger
from src.web import storage
from src.web.forms import parse_opt_float
from src.web.templating import render_error, templates

logger = get_logger(__name__)

router = APIRouter(prefix="/api/databases")


def _render_db_list(request: Request) -> HTMLResponse:
    """DB 一覧のフラグメントを返す。"""
    return templates.TemplateResponse(
        request,
        "partials/db_list.html",
        {"databases": vdb.list_databases()},
    )


def _render_speakers(request: Request, db_name: str) -> HTMLResponse:
    """指定 DB の話者一覧フラグメントを返す。"""
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": db_name, "speakers": vdb.list_speakers(db_name)},
    )


@router.get("")
async def api_list_databases():
    """DB 一覧を JSON で返す。"""
    return {"databases": vdb.list_databases(), "root": str(vdb.get_root())}


@router.post("", response_class=HTMLResponse)
async def api_create_database(request: Request, name: str = Form(...)):
    """DB を新規作成する。"""
    safe = vdb.sanitize_name(name)
    if safe is None:
        return render_error(request, "DB名が無効です（空 or 使用不可文字）。")
    try:
        vdb.create_database(safe)
    except ValueError as e:
        return render_error(request, str(e))
    return _render_db_list(request)


@router.delete("/{name}", response_class=HTMLResponse)
async def api_delete_database(request: Request, name: str):
    """DB を話者ファイルごと削除する。"""
    try:
        vdb.delete_database(name)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))
    return _render_db_list(request)


@router.get("/list", response_class=HTMLResponse)
async def api_db_list_fragment(request: Request):
    """DB 一覧のフラグメントを返す（一覧の再描画用）。"""
    return _render_db_list(request)


@router.get("/select-options", response_class=HTMLResponse)
async def api_db_select_options(request: Request, selected: str = ""):
    """DB 選択プルダウンの ``<option>`` を返す。"""
    return templates.TemplateResponse(
        request,
        "partials/db_select_options.html",
        {"databases": vdb.list_databases(), "selected": selected},
    )


@router.get("/{name}/speakers", response_class=HTMLResponse)
async def api_list_speakers(request: Request, name: str):
    """DB 内の話者一覧を返す。"""
    try:
        return _render_speakers(request, name)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))


@router.post("/{name}/speakers/upload", response_class=HTMLResponse)
async def api_upload_speakers(
    request: Request,
    name: str,
    files: List[UploadFile] = File(...),
):
    """声紋ファイルをまとめて DB へ取り込む。対応外の拡張子は読み飛ばす。"""
    try:
        # DB の存在確認
        vdb.database_dir(name)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))

    for uf in files:
        if not uf or not uf.filename:
            continue
        if Path(uf.filename).suffix.lower() not in vdb.SUPPORTED_AUDIO_EXTENSIONS:
            continue
        rname = Path(uf.filename).name
        tmp = storage.save_upload(uf, f"upload_registry_{rname}")
        try:
            vdb.add_speaker_file(name, tmp, dest_filename=rname)
        except ValueError as e:
            logger.warning(f"upload skipped ({rname}): {e}")
        finally:
            storage.remove_quietly(tmp)

    return _render_speakers(request, name)


@router.delete("/{name}/speakers/{filename}", response_class=HTMLResponse)
async def api_delete_speaker(request: Request, name: str, filename: str):
    """話者ファイルを削除する。"""
    try:
        vdb.delete_speaker(name, filename)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))
    return _render_speakers(request, name)


@router.post("/{name}/speakers/{filename}/rename", response_class=HTMLResponse)
async def api_rename_speaker(
    request: Request,
    name: str,
    filename: str,
    new_name: str = Form(...),
):
    """話者ラベル（ファイル名の拡張子なし部分）を変更する。"""
    try:
        vdb.rename_speaker(name, filename, new_name)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))
    return _render_speakers(request, name)


@router.get("/{name}/speakers/{filename}/audio")
async def api_speaker_audio(name: str, filename: str):
    """話者ファイルを音声として返す（試聴用。キャッシュさせない）。"""
    try:
        path = vdb.speaker_path(name, filename)
    except (ValueError, FileNotFoundError):
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(
        path=path,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.post("/{name}/speakers/{filename}/trim", response_class=HTMLResponse)
async def api_trim_speaker(
    request: Request,
    name: str,
    filename: str,
    start: str = Form(""),
    end: str = Form(""),
):
    """登録済み話者ファイルを指定範囲で切り出して上書きする（純粋な声だけ残す）。"""
    try:
        path = vdb.speaker_path(name, filename)
    except (ValueError, FileNotFoundError) as e:
        return render_error(request, str(e))

    crop_start = parse_opt_float(start)
    crop_end = parse_opt_float(end)
    if (crop_start is None or crop_start <= 0) and crop_end is None:
        return render_error(request, "切り出し範囲が指定されていません。")
    if crop_start is not None and crop_end is not None and crop_end <= crop_start:
        return render_error(request, "終了時間は開始時間より後にしてください。")

    # 一時ファイルに切り出してから上書き（同じ拡張子を維持）
    tmp = storage.temp_path(f"_trim_{uuid.uuid4().hex}{path.suffix}")
    try:
        extract_audio(path, tmp, start=crop_start, end=crop_end, to_wav16k=False)
        shutil.move(str(tmp), str(path))
    except subprocess.CalledProcessError as e:
        storage.remove_quietly(tmp)
        return render_error(request, f"音声の切り出しに失敗しました: {e.stderr}")

    return _render_speakers(request, name)
