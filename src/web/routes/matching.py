"""Step 2: 台本と文字起こし SRT を対応付けて対応表 CSV を作る。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Tuple

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.common.filenames import safe_output_name
from src.common.logging import get_logger
from src.config import DEFAULT_MATCHING_CSV_NAME, PROJECT_ROOT
from src.web import jobs, storage
from src.web.converters import txt_to_script_csv_bytes
from src.web.errors import WebInputError
from src.web.templating import render_error, templates

logger = get_logger(__name__)

router = APIRouter()

#: マッチング本体を動かすモジュール。PyTorch MPS と FAISS-CPU が同一スレッドで
#: 競合するため、Web プロセスとは別プロセスで実行する。
_MATCHER_MODULE = "src.subtitle.matcher"

#: 画面に出すログ抜粋の長さ（文字数）。
_FAILURE_LOG_CHARS = 3000


@router.post("/process/matching", response_class=HTMLResponse)
async def process_matching(
    request: Request,
    script_file: UploadFile = File(...),
    job_id: str = Form(...),
    output_csv_name: str = Form(DEFAULT_MATCHING_CSV_NAME),
):
    """台本をアップロードし、Step 1 の SRT との対応表を作る。"""
    try:
        if not script_file.filename:
            raise WebInputError(
                "台本or書き起こしテキストファイルが指定されていません。"
            )
        if not job_id:
            raise WebInputError("Step 1 (文字起こし) を先に実行してください。")

        srt_path = _resolve_step1_srt(job_id)
        script_path = await _save_script_upload(script_file)
        output_path = storage.temp_path(
            safe_output_name(output_csv_name, DEFAULT_MATCHING_CSV_NAME)
        )

        returncode, match_log = await _run_matcher(script_path, srt_path, output_path)
        if returncode != 0:
            return render_error(
                request,
                f"マッチングプロセスが異常終了しました (rc={returncode})\n\n"
                + match_log[-_FAILURE_LOG_CHARS:],
            )
        if not output_path.exists():
            raise WebInputError(
                "マッチング処理に失敗しました。出力ファイルが生成されませんでした。"
            )

        return templates.TemplateResponse(
            request,
            "partials/success_matching.html",
            {
                "filename": output_path.name,
                "download_url": f"/download/{output_path.name}",
            },
        )

    except WebInputError as e:
        return render_error(request, str(e))
    except Exception as e:
        logger.exception("matching failed")
        return render_error(request, f"エラーが発生しました: {e}")


def _resolve_step1_srt(job_id: str) -> Path:
    """Step 1 のジョブから SRT のパスを取り出す。

    Raises:
        WebInputError: ジョブや SRT が見つからない場合。
    """
    job = jobs.load_job(job_id)
    if job is None:
        raise WebInputError(f"Step 1 のジョブが見つかりません: {job_id}")
    srt_path_str = job.get("srt_path")
    if not srt_path_str:
        raise WebInputError("Step 1 の SRT 出力が記録されていません。")
    srt_path = Path(srt_path_str)
    if not srt_path.exists():
        raise WebInputError(f"Step 1 の SRT が見つかりません: {srt_path}")
    return srt_path


async def _save_script_upload(script_file: UploadFile) -> Path:
    """台本を作業ディレクトリへ保存する。``.txt`` は台本 CSV に変換する。"""
    script_name = Path(script_file.filename or "").name
    if Path(script_name).suffix.lower() != ".txt":
        return storage.save_upload(script_file, f"upload_script_{script_name}")

    csv_bytes = txt_to_script_csv_bytes(await script_file.read())
    script_path = storage.temp_path(f"upload_script_{Path(script_name).stem}.csv")
    script_path.write_bytes(csv_bytes)
    return script_path


async def _run_matcher(
    script_path: Path, srt_path: Path, output_path: Path
) -> Tuple[int, str]:
    """マッチングを別プロセスで実行し、``(終了コード, ログ)`` を返す。"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-m",
        _MATCHER_MODULE,
        str(script_path),
        str(srt_path),
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    stdout_bytes, _ = await proc.communicate()
    match_log = stdout_bytes.decode("utf-8", errors="replace")
    # 子プロセスのログは書式を足さずに親のコンソールへそのまま流す
    sys.stdout.write(match_log)
    sys.stdout.flush()
    return proc.returncode, match_log
