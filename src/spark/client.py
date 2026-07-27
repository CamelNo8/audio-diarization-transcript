"""PC 側から Spark の推論 API を呼ぶクライアント。

使い方:
    from src.spark.client import SparkJobOptions, transcribe_on_spark

    result, vocals_wav = transcribe_on_spark(
        "path/to/audio.wav",
        SparkJobOptions(num_speakers=3, denoise="fast"),
    )
    # result["segments"]: [{"start","end","text","cluster_id"}, ...]
    # result["clusters"]: {cluster_id: {"rep_start","rep_end"}, ...}  ← 照合用代表区間
    # vocals_wav: 背景音除去済み 16kHz mono WAV のローカルパス（声紋照合に使う）

接続先は環境変数 SPARK_URL（既定 http://192.168.1.50:8000）で変更可能。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests

from src.common.logging import configure_logging, get_logger
from src.config import DEFAULT_DIARIZATION_MODEL, TEMP_DIR

logger = get_logger(__name__)

#: 接続先の既定値。環境変数 ``SPARK_URL`` で上書きできる。
SPARK_URL = os.getenv("SPARK_URL", "http://192.168.1.50:8000").rstrip("/")

#: 照合用 WAV の既定の保存先。
DEFAULT_VOCALS_OUT = TEMP_DIR / "vocals_from_spark.wav"

# HTTP ごとのタイムアウト（秒）。音声の往復は時間がかかるため個別に持つ。
_UPLOAD_TIMEOUT_SEC = 60
_STATUS_TIMEOUT_SEC = 30
_RESULT_TIMEOUT_SEC = 60
_DOWNLOAD_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class SparkJobOptions:
    """Spark に投げるジョブの設定。既定のままでも動く。"""

    #: 話者数。不明なら ``None``（自動推定）。
    num_speakers: Optional[int] = None
    #: whisper のサイズ。
    model: str = "large-v3"
    #: 話者分離パイプライン。
    pyannote_model_id: str = DEFAULT_DIARIZATION_MODEL
    #: 背景音除去の強度（``off`` / ``fast`` / ``high``）。
    denoise: str = "fast"
    #: 完了確認の間隔（秒）。
    poll_interval: float = 3.0
    #: 全体のタイムアウト（秒）。
    timeout: float = 1800.0
    #: 接続先。省略時は環境変数 ``SPARK_URL``。
    base_url: Optional[str] = None

    @property
    def base(self) -> str:
        """末尾のスラッシュを落とした接続先。"""
        return (self.base_url or SPARK_URL).rstrip("/")

    def as_form(self) -> dict[str, str]:
        """``POST /jobs`` へ送るフォーム値。"""
        return {
            "num_speakers": "" if self.num_speakers is None else str(self.num_speakers),
            "model": self.model,
            "pyannote_model_id": self.pyannote_model_id,
            "denoise": self.denoise,
        }


def transcribe_on_spark(
    audio_path: str | Path,
    options: Optional[SparkJobOptions] = None,
    vocals_out: str | Path = DEFAULT_VOCALS_OUT,
) -> Tuple[dict, str]:
    """音声を Spark に送り、文字起こし＋話者分離結果と処理済みWAVを取得する。

    Args:
        audio_path: 送信する音声ファイル。
        options: ジョブの設定。省略時は既定値。
        vocals_out: 照合用 WAV の保存先。

    Returns:
        ``(結果 dict, 保存した WAV のパス)``。

    Raises:
        RuntimeError: ジョブが失敗した、またはタイムアウトした場合。
        requests.HTTPError: HTTP エラーが返った場合。
    """
    options = options or SparkJobOptions()
    job_id = _create_job(Path(audio_path), options)
    _wait_until_done(job_id, options)

    result = requests.get(
        f"{options.base}/jobs/{job_id}/result", timeout=_RESULT_TIMEOUT_SEC
    ).json()
    return result, _download_vocals(job_id, options, Path(vocals_out))


def _create_job(audio_path: Path, options: SparkJobOptions) -> str:
    """音声をアップロードしてジョブを作り、``job_id`` を返す。"""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{options.base}/jobs",
            files={"file": (audio_path.name, f, "application/octet-stream")},
            data=options.as_form(),
            timeout=_UPLOAD_TIMEOUT_SEC,
        )
    resp.raise_for_status()
    return resp.json()["job_id"]


def _wait_until_done(job_id: str, options: SparkJobOptions) -> None:
    """ジョブが完了するまでポーリングする。

    Raises:
        RuntimeError: ジョブが失敗した、またはタイムアウトした場合。
    """
    deadline = time.monotonic() + options.timeout
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Spark ジョブがタイムアウトしました (job_id={job_id})")
        status = requests.get(
            f"{options.base}/jobs/{job_id}", timeout=_STATUS_TIMEOUT_SEC
        ).json()
        if status["status"] == "done":
            return
        if status["status"] == "error":
            raise RuntimeError(f"Spark ジョブが失敗しました: {status.get('error')}")
        time.sleep(options.poll_interval)


def _download_vocals(job_id: str, options: SparkJobOptions, dest: Path) -> str:
    """照合用の処理済み WAV を取得して保存する。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    wav = requests.get(
        f"{options.base}/jobs/{job_id}/vocals", timeout=_DOWNLOAD_TIMEOUT_SEC
    )
    wav.raise_for_status()
    dest.write_bytes(wav.content)
    return str(dest)


def _main(argv: list[str]) -> int:
    """簡易動作確認: ``python -m src.spark.client <audio> [num_speakers]``"""
    if len(argv) < 2:
        logger.error("usage: python -m src.spark.client <audio> [num_speakers]")
        return 1
    num_speakers = int(argv[2]) if len(argv) > 2 else None
    result, vocals = transcribe_on_spark(
        argv[1], SparkJobOptions(num_speakers=num_speakers)
    )
    logger.info(
        f"speakers={result['num_speakers']}  "
        f"segments={len(result['segments'])}  vocals={vocals}"
    )
    logger.info(json.dumps(result["segments"][:5], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(_main(sys.argv))
