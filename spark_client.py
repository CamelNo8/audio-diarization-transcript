"""PC 側から Spark の推論 API を呼ぶクライアント。

使い方:
    from spark_client import transcribe_on_spark

    result, vocals_wav = transcribe_on_spark(
        "path/to/audio.wav",
        num_speakers=3,        # 不明なら None
        denoise="fast",        # off / fast / high
    )
    # result["segments"]: [{"start","end","text","cluster_id"}, ...]
    # result["clusters"]: {cluster_id: {"rep_start","rep_end"}, ...}  ← 照合用代表区間
    # vocals_wav: 背景音除去済み 16kHz mono WAV のローカルパス（声紋照合に使う）

接続先は環境変数 SPARK_URL（既定 http://192.168.1.50:8000）で変更可能。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

SPARK_URL = os.getenv("SPARK_URL", "http://192.168.1.50:8000").rstrip("/")


def transcribe_on_spark(
    audio_path: str | Path,
    num_speakers: Optional[int] = None,
    model: str = "large-v3",
    pyannote_model_id: str = "pyannote/speaker-diarization-3.1",
    denoise: str = "fast",
    vocals_out: str | Path = "temp/vocals_from_spark.wav",
    poll_interval: float = 3.0,
    timeout: float = 1800.0,
    base_url: Optional[str] = None,
) -> Tuple[dict, str]:
    """音声を Spark に送り、文字起こし＋話者分離結果と処理済みWAVを取得する。

    戻り値: (result_dict, vocals_wav_path)
    例外:   ジョブが error / タイムアウト / HTTP エラー時に RuntimeError を送出。
    """
    base = (base_url or SPARK_URL).rstrip("/")
    audio_path = Path(audio_path)

    # 1. アップロード（ジョブ作成）
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{base}/jobs",
            files={"file": (audio_path.name, f, "application/octet-stream")},
            data={
                "num_speakers": "" if num_speakers is None else str(num_speakers),
                "model": model,
                "pyannote_model_id": pyannote_model_id,
                "denoise": denoise,
            },
            timeout=60,
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    # 2. 完了までポーリング
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Spark ジョブがタイムアウトしました (job_id={job_id})")
        st = requests.get(f"{base}/jobs/{job_id}", timeout=30).json()
        status = st["status"]
        if status == "done":
            break
        if status == "error":
            raise RuntimeError(f"Spark ジョブが失敗しました: {st.get('error')}")
        time.sleep(poll_interval)

    # 3. 結果 JSON を取得
    result = requests.get(f"{base}/jobs/{job_id}/result", timeout=60).json()

    # 4. 照合用 WAV を取得して保存
    vocals_out = Path(vocals_out)
    vocals_out.parent.mkdir(parents=True, exist_ok=True)
    wav = requests.get(f"{base}/jobs/{job_id}/vocals", timeout=300)
    wav.raise_for_status()
    vocals_out.write_bytes(wav.content)

    return result, str(vocals_out)


if __name__ == "__main__":
    # 簡易動作確認: python spark_client.py path/to/audio.wav [num_speakers]
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python spark_client.py <audio> [num_speakers]")
        raise SystemExit(1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res, vocals = transcribe_on_spark(sys.argv[1], num_speakers=n)
    print(f"speakers={res['num_speakers']}  segments={len(res['segments'])}  vocals={vocals}")
    print(json.dumps(res["segments"][:5], ensure_ascii=False, indent=2))
