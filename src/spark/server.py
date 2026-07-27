"""NVIDIA DGX Spark 上で動かす推論 API サーバー。

PC から音声をアップロードすると、以下を GPU(CUDA) で実行して結果を返す:
  1. 背景音 / BGM 除去（audio-separator[gpu]）
  2. 話者分離（pyannote, CUDA）
  3. 文字起こし（faster-whisper, CUDA）

話者照合（声紋DB）と GUI は PC 側に残すため、ここでは cluster_id 付きの
セグメントと、照合用の処理済みボーカル WAV だけを返す。

非同期ジョブ方式:
  POST /jobs               → job_id を即返し（裏でスレッド処理）
  GET  /jobs/{id}          → 進捗 {status, error}
  GET  /jobs/{id}/result   → 完了後の JSON 本体
  GET  /jobs/{id}/vocals   → 照合用の処理済み WAV（audio/wav）

起動例（Spark の CUDA コンテナ内。リポジトリのルートで実行する）:
  uvicorn src.spark.server:app --host 0.0.0.0 --port 8000

環境変数:
  HF_TOKEN          Hugging Face トークン（pyannote モデル取得・利用規約同意済みのもの）
  SPARK_WORK_DIR    作業ディレクトリ（既定: /tmp/spark_jobs）
"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.common.logging import configure_logging, get_logger
from src.config import DENOISE_MODELS
from src.diarization import processor as audio_processor

configure_logging()
logger = get_logger(__name__)

# pyannote モデルはローカルキャッシュに無ければ取得する必要があるため、
# オフライン強制はしない（初回はオンラインでダウンロードさせる）。
os.environ.setdefault("HF_HUB_OFFLINE", "0")

WORK_DIR = Path(os.getenv("SPARK_WORK_DIR", "/tmp/spark_jobs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Spark Diarization/Transcription API")

# job_id -> {"status", "error", "result", "vocals"}
# 最小構成のためメモリ保持（プロセス再起動で消える）。
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class JobOptions:
    """ジョブ作成フォームの推論パラメータ。"""

    num_speakers: str = Form("")  # "" or 整数
    model: str = Form("large-v3")  # whisper サイズ
    pyannote_model_id: str = Form("pyannote/speaker-diarization-3.1")
    denoise: str = Form("fast")  # off / fast / high
    hf_token: str = Form("")  # 空なら環境変数を使用

    @property
    def speaker_count(self) -> Optional[int]:
        """話者数の指定。数値でなければ ``None``（自動推定）。"""
        text = str(self.num_speakers).strip()
        return int(text) if text.isdigit() else None


@dataclass(frozen=True)
class _JobRequest:
    """裏スレッドへ渡す 1 ジョブ分の入力。"""

    job_id: str
    wav_path: Path
    options: JobOptions
    hf_token: str


def _update_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        JOBS[job_id].update(fields)


def _run_job(job: _JobRequest) -> None:
    """1件のジョブを最後まで処理し、状態を :data:`JOBS` に反映する。"""
    try:
        _update_job(job.job_id, status="running")
        result = _transcribe(job)
        _update_job(
            job.job_id,
            status="done",
            result={
                "segments": result["segments"],
                "clusters": result["clusters"],
                "num_speakers": result["num_speakers"],
            },
            vocals=result.get("vocals_path"),
        )
        logger.info(
            f"[{job.job_id}] done: {result['num_speakers']} speakers, "
            f"{len(result['segments'])} segments"
        )
    except Exception as e:
        logger.error(f"[{job.job_id}] failed: {e}\n{traceback.format_exc()}")
        _update_job(job.job_id, status="error", error=str(e))
    finally:
        # アップロード元 WAV は不要になったら削除（vocals は DL 用に残す）
        try:
            job.wav_path.unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"[{job.job_id}] 入力WAVを削除できませんでした: {e}")


def _transcribe(job: _JobRequest) -> dict:
    """話者分離と文字起こしを GPU で実行して結果 dict を返す。"""
    separator_model = DENOISE_MODELS.get(job.options.denoise)
    with audio_processor.AudioProcessor(
        audio_file=job.wav_path,
        # API では CSV を書き出さないが、コンストラクタが要求するため渡す
        output_csv_path=WORK_DIR / f"{job.job_id}.csv",
        mlx_model_id=job.options.model,
        pyannote_model_id=job.options.pyannote_model_id,
        hf_token=job.hf_token,
        identifier=None,  # 照合は PC 側で行う
        registry_dir=None,
        interactive_unknown_resolve=False,
        denoise=separator_model is not None,
        separator_model=separator_model,
        whisper_backend="faster",  # Linux/CUDA は faster-whisper 固定
    ) as processor:
        return processor.process_for_api(
            known_num_speakers=job.options.speaker_count,
            vocals_out=WORK_DIR / f"{job.job_id}_vocals.wav",
        )


@app.get("/health")
def health():
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda, name = False, None
    return {"ok": True, "cuda": cuda, "gpu": name}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    options: JobOptions = Depends(),
):
    """音声を受け取ってジョブを登録し、``job_id`` を即返す。"""
    token = options.hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise HTTPException(
            400, "HF_TOKEN が未設定です（フォーム hf_token か環境変数で指定）"
        )

    job_id = uuid.uuid4().hex[:12]
    wav_path = WORK_DIR / f"{job_id}_input{Path(file.filename or '').suffix or '.wav'}"
    wav_path.write_bytes(await file.read())

    with _JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "error": None,
            "result": None,
            "vocals": None,
        }

    threading.Thread(
        target=_run_job,
        args=(_JobRequest(job_id, wav_path, options, token),),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return {"status": job["status"], "error": job["error"]}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job["status"] == "error":
        raise HTTPException(500, job["error"] or "job failed")
    if job["status"] != "done":
        raise HTTPException(409, f"not ready (status={job['status']})")
    return job["result"]


@app.get("/jobs/{job_id}/vocals")
def job_vocals(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job["status"] != "done":
        raise HTTPException(409, f"not ready (status={job['status']})")
    vocals = job.get("vocals")
    if not vocals or not Path(vocals).exists():
        raise HTTPException(404, "vocals not available（denoise=off だった可能性）")
    return FileResponse(vocals, media_type="audio/wav", filename=f"{job_id}_vocals.wav")
