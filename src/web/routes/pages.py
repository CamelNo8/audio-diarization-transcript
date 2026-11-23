"""画面の表示と生成物のダウンロード。

手順ごとに1画面ずつ出す（Step 1 = ``/``、Step 2 = ``/matching``、
Step 3 = ``/generation``）。処理を実行する口は ``/process/*`` 側にあり、
ここは表示だけを担う。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

import src.voice_db.registry as vdb
from src.common.filenames import is_safe_output_name
from src.config import (
    DEFAULT_MATCHING_CSV_NAME,
    DEFAULT_SUBTITLE_SRT_NAME,
    DEFAULT_TRANSCRIPTION_SRT_NAME,
)
from src.web import jobs, storage
from src.web.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Step 1: 文字起こし＆話者分離の画面。"""
    return templates.TemplateResponse(
        request,
        "step1.html",
        {
            "databases": vdb.list_databases(),
            "default_srt_name": DEFAULT_TRANSCRIPTION_SRT_NAME,
        },
    )


@router.get("/matching", response_class=HTMLResponse)
async def matching_page(request: Request, job_id: str = ""):
    """Step 2: マッチングの画面。

    Args:
        job_id: Step 1 のジョブID。渡されると、その出力SRTを使う前提で表示する。
            未指定・不正・見つからない場合は Step 1 を先に実行するよう案内する。
    """
    stt_filename = _step1_srt_filename(job_id)
    return templates.TemplateResponse(
        request,
        "step2.html",
        {
            "job_id": job_id if stt_filename else "",
            "stt_filename": stt_filename,
            "default_csv_name": DEFAULT_MATCHING_CSV_NAME,
        },
    )


@router.get("/generation", response_class=HTMLResponse)
async def generation_page(request: Request, job_id: str = "", csv: str = ""):
    """Step 3: 字幕生成の画面。

    Args:
        job_id: Step 1 のジョブID（手順の表示を前の画面へ戻すためだけに使う）。
        csv: Step 2 が出力した対応表のファイル名。案内文に出す。
    """
    return templates.TemplateResponse(
        request,
        "step3.html",
        {
            "job_id": job_id,
            "matching_csv": csv if is_safe_output_name(csv) else "",
            "default_srt_name": DEFAULT_SUBTITLE_SRT_NAME,
        },
    )


@router.get("/databases", response_class=HTMLResponse)
async def databases_page(request: Request):
    """声紋DB の管理画面。"""
    return templates.TemplateResponse(
        request,
        "databases.html",
        {"databases": vdb.list_databases()},
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


def _step1_srt_filename(job_id: str) -> Optional[str]:
    """Step 1 のジョブから出力SRTのファイル名を取り出す。

    Args:
        job_id: URL から渡されたジョブID。

    Returns:
        SRT のファイル名。ジョブが無い・まだ完了していない場合は ``None``。
    """
    if not job_id:
        return None
    try:
        job = jobs.load_job(job_id)
    except ValueError:
        # job_id は URL から来るため不正な値もあり得る。画面は案内付きで出す
        return None
    if job is None:
        return None
    return job.get("srt_filename")
