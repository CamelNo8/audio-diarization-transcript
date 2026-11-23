"""Step 3: 編集済みの対応表 CSV から字幕 SRT を生成する。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.common.filenames import safe_output_name
from src.common.logging import get_logger
from src.config import DEFAULT_SUBTITLE_SRT_NAME
from src.subtitle_gen import generate_srt
from src.web import storage
from src.web.errors import WebInputError
from src.web.templating import render_error, templates

logger = get_logger(__name__)

router = APIRouter()


@router.post("/process/generation", response_class=HTMLResponse)
async def process_generation(
    request: Request,
    edited_csv: UploadFile = File(...),
    output_srt_name: str = Form(DEFAULT_SUBTITLE_SRT_NAME),
):
    """編集済み対応表 CSV をアップロードして字幕 SRT を書き出す。"""
    try:
        if not edited_csv.filename:
            raise WebInputError("編集済み対応表CSVが指定されていません。")

        csv_name = Path(edited_csv.filename).name
        csv_path = storage.save_upload(edited_csv, f"upload_edited_{csv_name}")

        # 出力名はフォームから来るため、パス結合の前に検証する（規約7.1）
        output_path = storage.temp_path(
            safe_output_name(output_srt_name, DEFAULT_SUBTITLE_SRT_NAME)
        )
        subtitle_count = generate_srt(csv_path, output_path)
        if subtitle_count == 0:
            raise WebInputError(
                "字幕データの読み込みに失敗しました。CSVフォーマットを確認してください。"
            )
        if not output_path.exists():
            raise WebInputError("SRT生成に失敗しました。")

        return templates.TemplateResponse(
            request,
            "partials/success_generation.html",
            {
                "filename": output_path.name,
                "download_url": f"/download/{output_path.name}",
                "count": subtitle_count,
            },
        )

    except WebInputError as e:
        return render_error(request, str(e))
    except Exception as e:
        logger.exception("generation failed")
        return render_error(request, f"エラーが発生しました: {e}")
