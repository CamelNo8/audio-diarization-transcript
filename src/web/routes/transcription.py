"""Step 1: 音声から文字起こし CSV / SRT を作る。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

import src.voice_db.registry as vdb
from src.common.filenames import safe_output_name
from src.common.logging import get_logger
from src.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SPEAKER_THRESHOLD,
    DEFAULT_TRANSCRIPTION_SRT_NAME,
    DEFAULT_WHISPER_MODEL,
    DENOISE_MODELS,
)
from src.diarization.processor import AudioProcessor
from src.diarization.registry import SUPPORTED_REGISTRY_EXTENSIONS
from src.web import jobs, storage
from src.web.converters import csv_to_srt_with_speaker
from src.web.errors import WebInputError
from src.web.forms import parse_opt_int
from src.web.identification import load_identifier, resolve_hf_token
from src.web.log_capture import capture_root_logs
from src.web.templating import render_error, templates

logger = get_logger(__name__)

router = APIRouter()

#: 画面に出すログ抜粋の長さ（文字数）。
_SUCCESS_LOG_CHARS = 3000
_FAILURE_LOG_CHARS = 2000


@dataclass(frozen=True)
class TranscriptionResult:
    """文字起こしの生成物。"""

    csv_path: Path
    srt_path: Path
    unknown_clusters: List[Dict[str, Any]]


@dataclass(frozen=True)
class _TranscriptionRun:
    """検証を通したあとの、1回分の実行条件。"""

    job_id: str
    audio_path: Path
    options: "TranscriptionOptions"
    registry_dir: Optional[Path]
    hf_token: str


@dataclass
class TranscriptionOptions:
    """Step 1 のフォーム入力（音声ファイル以外）。"""

    db_choice: str = Form("none")  # "none" / "existing" / "new"
    db_existing_name: str = Form("")
    db_new_name: str = Form("")
    output_srt_name: str = Form(DEFAULT_TRANSCRIPTION_SRT_NAME)
    threshold: float = Form(DEFAULT_SPEAKER_THRESHOLD)
    num_speakers: str = Form("")
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL)
    mlx_model: str = Form(DEFAULT_WHISPER_MODEL)
    whisper_backend: str = Form("auto")  # "auto" / "mlx" / "faster"
    whisper_quality: str = Form("")  # "large-v3" / "medium" / "small" / "__custom__"
    whisper_custom_model: str = Form("")
    pyannote_model_id: str = Form(DEFAULT_DIARIZATION_MODEL)
    hf_token_override: str = Form("")
    denoise_mode: str = Form("off")

    @property
    def whisper_model(self) -> str:
        """Whisper のモデル ID を決める。

        品質ドロップダウンを優先し、カスタム指定なら自由入力を使う。
        どちらも空なら後方互換で旧 ``mlx_model`` フィールドの値を使う。
        """
        if self.whisper_quality == "__custom__":
            return self.whisper_custom_model.strip() or self.mlx_model
        return self.whisper_quality or self.mlx_model


@router.post("/process/transcription", response_class=HTMLResponse)
async def process_transcription(
    request: Request,
    audio_file: UploadFile = File(...),
    registry_files: List[UploadFile] = File(default=[]),
    options: TranscriptionOptions = Depends(),
):
    """音声をアップロードして文字起こし SRT を作り、ラベル付けジョブを登録する。"""
    try:
        _require_ffmpeg()
        hf_token = resolve_hf_token(options.hf_token_override)
        audio_path = _save_audio_upload(audio_file)

        registry_dir = _resolve_registry_dir(options)
        _store_registry_uploads(registry_files, registry_dir)

        run = _TranscriptionRun(
            job_id=jobs.new_job_id(),
            audio_path=audio_path,
            options=options,
            registry_dir=registry_dir,
            hf_token=hf_token,
        )
        with capture_root_logs() as log_buffer:
            output = _run_transcription(run)

        if output is None:
            return render_error(
                request,
                "文字起こし処理に失敗しました。\n"
                + log_buffer.getvalue()[-_FAILURE_LOG_CHARS:],
            )

        csv_to_srt_with_speaker(output.csv_path, output.srt_path)
        jobs.save_job(run.job_id, _build_job(run, output))
        return _render_success(request, run.job_id, output, log_buffer.getvalue())

    except WebInputError as e:
        return render_error(request, str(e))
    except Exception as e:
        logger.exception("transcription failed")
        return render_error(request, f"エラーが発生しました: {e}")


def _require_ffmpeg() -> None:
    """ffmpeg が使えることを確認する。

    Raises:
        WebInputError: PATH に見つからない場合。
    """
    if not shutil.which("ffmpeg"):
        raise WebInputError(
            "ffmpeg が PATH に見つかりません。"
            "`brew install ffmpeg` でインストールしてください。"
        )


def _save_audio_upload(audio_file: UploadFile) -> Path:
    """アップロードされた音声/動画を作業ディレクトリへ保存する。

    Raises:
        WebInputError: ファイルが選ばれていない場合。
    """
    if not audio_file.filename:
        raise WebInputError("音声/動画ファイルが指定されていません。")
    safe_name = Path(audio_file.filename).name
    return storage.save_upload(audio_file, f"upload_audio_{safe_name}")


def _resolve_registry_dir(options: TranscriptionOptions) -> Optional[Path]:
    """声紋DB の選択／新規作成を解決する。``none`` なら ``None``。

    Raises:
        WebInputError: 選択が不正、または DB を用意できない場合。
    """
    if options.db_choice == "existing":
        if not options.db_existing_name:
            raise WebInputError("既存DBが選択されていません。")
        try:
            return vdb.database_dir(options.db_existing_name)
        except (ValueError, FileNotFoundError) as e:
            raise WebInputError(f"DBエラー: {e}") from e

    if options.db_choice == "new":
        safe_new = vdb.sanitize_name(options.db_new_name)
        if safe_new is None:
            raise WebInputError("新規DB名が無効です。")
        try:
            return vdb.create_database(safe_new)
        except ValueError as e:
            # 既存ならそれを使う
            try:
                return vdb.database_dir(safe_new)
            except Exception as lookup_error:
                raise WebInputError(f"DB作成エラー: {e}") from lookup_error

    return None


def _store_registry_uploads(
    registry_files: List[UploadFile], registry_dir: Optional[Path]
) -> None:
    """アップロードされた声紋ファイルを DB へ取り込む。

    Raises:
        WebInputError: 取り込む対象があるのに保存先DB が選ばれていない場合。
    """
    valid_uploads = [
        rf
        for rf in registry_files
        if rf
        and rf.filename
        and Path(rf.filename).suffix.lower() in SUPPORTED_REGISTRY_EXTENSIONS
    ]
    if not valid_uploads:
        return
    if registry_dir is None:
        raise WebInputError(
            "声紋ファイルがアップロードされていますが、保存先DBが選択されていません。"
            "「既存DBを使う」または「新規DBを作成」を選択してください。"
        )
    for rf in valid_uploads:
        rname = Path(rf.filename).name
        tmp_upload = storage.save_upload(rf, f"upload_registry_{rname}")
        try:
            vdb.add_speaker_file(registry_dir.name, tmp_upload, dest_filename=rname)
        finally:
            storage.remove_quietly(tmp_upload)


def _resolve_output_paths(output_srt_name: str) -> Tuple[Path, Path]:
    """出力する ``(CSV, SRT)`` のパスを決める。

    名前はフォームから来るため、パス結合の前に検証する（規約7.1）。
    """
    srt_name = safe_output_name(output_srt_name, DEFAULT_TRANSCRIPTION_SRT_NAME)
    csv_name = f"{Path(srt_name).stem or 'transcription'}.csv"
    return storage.temp_path(csv_name), storage.temp_path(srt_name)


def _run_transcription(run: _TranscriptionRun) -> Optional[TranscriptionResult]:
    """文字起こしを実行して生成物を返す。失敗時は ``None``。"""
    options = run.options
    csv_path, srt_path = _resolve_output_paths(options.output_srt_name)

    identifier = None
    if run.registry_dir is not None:
        identifier = load_identifier(
            run.registry_dir,
            model_name=options.embedding_model,
            hf_token=run.hf_token,
            threshold=options.threshold,
        )

    separator_model = DENOISE_MODELS.get(options.denoise_mode)
    with AudioProcessor(
        audio_file=run.audio_path,
        output_csv_path=csv_path,
        mlx_model_id=options.whisper_model,
        pyannote_model_id=options.pyannote_model_id,
        hf_token=run.hf_token,
        identifier=identifier,
        registry_dir=run.registry_dir,
        interactive_unknown_resolve=False,
        denoise=separator_model is not None,
        separator_model=separator_model,
        whisper_backend=options.whisper_backend,
    ) as processor:
        success = processor.process_and_save_to_csv(
            known_num_speakers=parse_opt_int(options.num_speakers)
        )
        # Unknown クラスタの音声を永続化（成功時のみ）
        unknown_clusters = (
            processor.persist_unknown_clusters(jobs.job_dir(run.job_id))
            if success
            else []
        )

    if not success or not csv_path.exists():
        return None
    return TranscriptionResult(csv_path, srt_path, unknown_clusters)


def _build_job(run: _TranscriptionRun, result: TranscriptionResult) -> Dict[str, Any]:
    """ラベル付け画面が使うジョブ状態を組み立てる。"""
    for cluster in result.unknown_clusters:
        cluster["resolved"] = False
        cluster["resolved_name"] = None
    return {
        "job_id": run.job_id,
        "csv_path": str(result.csv_path),
        "csv_filename": result.csv_path.name,
        "srt_path": str(result.srt_path),
        "srt_filename": result.srt_path.name,
        "db_name": run.registry_dir.name if run.registry_dir is not None else None,
        "threshold": run.options.threshold,
        "embedding_model": run.options.embedding_model,
        "clusters": result.unknown_clusters,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _render_success(
    request: Request, job_id: str, result: TranscriptionResult, log_text: str
):
    """完了パネルのフラグメントを返す。"""
    return templates.TemplateResponse(
        request,
        "partials/success_transcription.html",
        {
            "filename": result.srt_path.name,
            "download_url": f"/download/{result.srt_path.name}",
            "log_excerpt": log_text[-_SUCCESS_LOG_CHARS:],
            "job_id": job_id,
            "unknown_count": len(result.unknown_clusters),
        },
    )
