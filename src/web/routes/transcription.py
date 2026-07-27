"""Step 1: 音声から文字起こし CSV / SRT を作る。"""

from __future__ import annotations

import io
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

import src.voice_db.registry as vdb
from src.common.files import remove_quietly
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

#: 進捗を取り直す間隔（htmx の hx-trigger に埋め込む）。
POLL_INTERVAL = "2s"

#: 処理中のログをジョブへ書き出す間隔（秒）。ポーリング間隔に合わせる。
LOG_FLUSH_INTERVAL_SEC = 2.0


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
    """入力を検証してジョブを登録し、進捗パネルを即座に返す。

    重い処理はワーカースレッドへ渡す。同期で走らせるとイベントループを
    占有し、処理中にアプリ全体が応答しなくなるため。
    """
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
        start_worker(run)
        return _render_progress(request, run.job_id)

    except WebInputError as e:
        return render_error(request, str(e))
    except Exception as e:
        logger.exception("transcription failed")
        return render_error(request, f"エラーが発生しました: {e}")


@router.get("/process/transcription/{job_id}/status", response_class=HTMLResponse)
async def transcription_status(request: Request, job_id: str):
    """進捗を返す。完了・失敗時はポーリングを含まない断片を返して止める。"""
    job = jobs.load_job(job_id)
    if job is None:
        return render_error(request, f"ジョブが見つかりません: {job_id}")
    if job.get("status") == jobs.STATUS_ERROR:
        message = job.get("error") or "文字起こし処理に失敗しました。"
        return render_error(request, message)
    if job.get("status") == jobs.STATUS_DONE:
        return _render_success(request, job)
    return _render_progress(request, job_id, job.get("log_excerpt", ""))


def start_worker(run: _TranscriptionRun) -> None:
    """ジョブを登録し、文字起こしをワーカースレッドで走らせる。

    テストではこの関数を差し替えて同期実行にする。
    """
    jobs.save_job(run.job_id, _initial_job(run))
    threading.Thread(target=run_job, args=(run,), daemon=True).start()


def run_job(run: _TranscriptionRun) -> None:
    """1件の文字起こしを最後まで処理し、結果をジョブへ書く。

    ワーカースレッドの入口。ここから先で起きた例外は画面に届かないため、
    すべて捕まえてジョブの ``error`` に残す。
    """
    try:
        jobs.update_job(run.job_id, status=jobs.STATUS_RUNNING)
        with capture_root_logs() as log_buffer:
            with _flush_log_periodically(run.job_id, log_buffer):
                result = _run_transcription(run)
        log_text = log_buffer.getvalue()

        if result is None:
            jobs.update_job(
                run.job_id,
                status=jobs.STATUS_ERROR,
                error="文字起こし処理に失敗しました。\n"
                + log_text[-_FAILURE_LOG_CHARS:],
            )
            return

        csv_to_srt_with_speaker(result.csv_path, result.srt_path)
        jobs.save_job(run.job_id, _build_job(run, result, log_text))
    except Exception as e:
        logger.exception("transcription failed")
        jobs.update_job(
            run.job_id,
            status=jobs.STATUS_ERROR,
            error=f"エラーが発生しました: {e}",
        )


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
            remove_quietly(tmp_upload)


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


@contextmanager
def _flush_log_periodically(job_id: str, log_buffer: io.StringIO) -> Iterator[None]:
    """処理中のログを定期的にジョブへ書き出す。

    進捗パネルはジョブの ``log_excerpt`` を読む。処理の最後にまとめて書くと
    走っている間ずっと空のままになるため、別スレッドで定期的に流し込む。
    ``with`` を抜けるときにスレッドを止めてから戻るので、呼び出し側が
    最終結果を書くのと入れ違いにはならない。
    """
    stop = threading.Event()

    def flush() -> None:
        while not stop.wait(LOG_FLUSH_INTERVAL_SEC):
            excerpt = log_buffer.getvalue()[-_SUCCESS_LOG_CHARS:]
            jobs.update_job(job_id, log_excerpt=excerpt)

    thread = threading.Thread(target=flush, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=LOG_FLUSH_INTERVAL_SEC)


def _initial_job(run: _TranscriptionRun) -> Dict[str, Any]:
    """処理開始前のジョブ状態。進捗の問い合わせはこれを読む。

    ``clusters`` を空で入れておくのは、完了前に ``/unknowns/<job_id>`` を
    開かれてもテンプレートが壊れないようにするため。
    """
    return {
        "job_id": run.job_id,
        "status": jobs.STATUS_RUNNING,
        "db_name": run.registry_dir.name if run.registry_dir is not None else None,
        "threshold": run.options.threshold,
        "embedding_model": run.options.embedding_model,
        "clusters": [],
        "log_excerpt": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _build_job(
    run: _TranscriptionRun, result: TranscriptionResult, log_text: str
) -> Dict[str, Any]:
    """完了後のジョブ状態を組み立てる。ラベル付け画面もこれを読む。"""
    for cluster in result.unknown_clusters:
        cluster["resolved"] = False
        cluster["resolved_name"] = None
    return {
        **_initial_job(run),
        "status": jobs.STATUS_DONE,
        "csv_path": str(result.csv_path),
        "csv_filename": result.csv_path.name,
        "srt_path": str(result.srt_path),
        "srt_filename": result.srt_path.name,
        "clusters": result.unknown_clusters,
        "log_excerpt": log_text[-_SUCCESS_LOG_CHARS:],
    }


def _render_progress(request: Request, job_id: str, log_excerpt: str = ""):
    """進捗パネルを返す。htmx がこの断片自身を定期的に取り直す。"""
    return templates.TemplateResponse(
        request,
        "partials/transcription_progress.html",
        {
            "job_id": job_id,
            "log_excerpt": log_excerpt,
            "poll_interval": POLL_INTERVAL,
        },
    )


def _render_success(request: Request, job: Dict[str, Any]):
    """完了パネルを返す。ポーリング属性を含まないため、取得はここで止まる。"""
    srt_filename = job.get("srt_filename", "")
    return templates.TemplateResponse(
        request,
        "partials/success_transcription.html",
        {
            "filename": srt_filename,
            "download_url": f"/download/{srt_filename}",
            "log_excerpt": job.get("log_excerpt", ""),
            "job_id": job["job_id"],
            "unknown_count": len(job.get("clusters", [])),
        },
    )
