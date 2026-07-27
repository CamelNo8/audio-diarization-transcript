"""Web レイヤのテストで共有するフィクスチャ。

差し替え対象のモジュールをこのファイルに集約している。参照先が移動しても、
書き換えるのはここだけで済むようにするため。
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

import app as app_module
import src.web.identification as identification_module
import src.web.jobs as _JOBS_MODULE
import src.web.routes.transcription as _TRANSCRIPTION_MODULE
import src.web.storage as _STORAGE_MODULE

#: 話者照合器の生成を差し替える対象。文字起こしとラベル付けの両方が経由する。
_IDENTIFICATION_MODULE = identification_module


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture
def 作業ディレクトリ(tmp_path, monkeypatch) -> Path:
    """アップロードと生成物の置き場を tmp_path 配下へ差し替える。

    ``../`` を数階層たどっても tmp_path の中に収まるよう深い位置に作る。
    こうするとテストの検査範囲が tmp_path 内で完結する。
    """
    work_dir = tmp_path / "a" / "b" / "temp"
    work_dir.mkdir(parents=True)
    monkeypatch.setattr(_STORAGE_MODULE, "TEMP_DIR", work_dir)
    return work_dir


@pytest.fixture
def ジョブ保存先(tmp_path, monkeypatch) -> Path:
    """実リポジトリの temp/clusters を汚さないよう保存先を差し替える。"""
    clusters_root = tmp_path / "clusters"
    monkeypatch.setattr(_JOBS_MODULE, "CLUSTERS_ROOT", clusters_root)
    return clusters_root


@pytest.fixture
def 声紋DBルート(tmp_path, monkeypatch) -> Path:
    """声紋DB のルートを tmp_path 配下へ差し替える。"""
    root = (tmp_path / "voice_databases").resolve()
    root.mkdir()
    monkeypatch.setenv("VOICE_DB_ROOT", str(root))
    return root


class FakeAudioProcessor:
    """:class:`AudioProcessor` の代役。

    重いモデルを読まずに、文字起こし CSV を1行だけ書いて成功を返す（規約2.5）。
    生成時の引数は :attr:`kwargs` に残るので、ルートが何を渡したかを検証できる。
    """

    #: 直近に生成されたインスタンス。ルートへ渡した引数の確認に使う。
    last: Optional["FakeAudioProcessor"] = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.known_num_speakers: Optional[int] = None
        FakeAudioProcessor.last = self

    def __enter__(self) -> "FakeAudioProcessor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        self.known_num_speakers = known_num_speakers
        csv_path = Path(self.kwargs["output_csv_path"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "speaker", "text", "cosine_distance"])
            writer.writerow(
                ["00:00:01:000", "00:00:02:000", "Unknown_1", "こんにちは", "1.000000"]
            )
        return True

    def persist_unknown_clusters(self, job_dir: Path) -> List[Dict[str, Any]]:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "clip_0.wav").write_bytes(b"RIFF----WAVE")
        return [
            {
                "cluster_id": "0",
                "unknown_label": "Unknown_1",
                "clip_filename": "clip_0.wav",
                "segment_start": 1.0,
                "segment_end": 2.0,
                "distance": 1.0,
                "candidate_distances": [],
            }
        ]


class FakeSpeakerIdentifier:
    """話者照合モデルの代役。登録は記録するだけ、照合は常に Unknown を返す。"""

    def __init__(self) -> None:
        self.registered: List[Tuple[str, Path]] = []

    def register_speaker(self, name: str, path: Path) -> None:
        self.registered.append((name, path))

    def identify_from_audio_path(
        self, path: Path
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        return "Unknown_1", 1.0, []


@pytest.fixture
def 文字起こしをモックにする(monkeypatch) -> FakeAudioProcessor:
    """重い文字起こし処理と ffmpeg 検出・HF トークンをテスト用に差し替える。

    ワーカースレッドも**同期実行**に差し替える。別スレッドのままだと
    「POST が返った時点で処理が終わっているか」がタイミング次第になり、
    テストが不安定になるため。
    """
    monkeypatch.setattr(_TRANSCRIPTION_MODULE, "AudioProcessor", FakeAudioProcessor)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    monkeypatch.setattr(_TRANSCRIPTION_MODULE, "start_worker", _start_worker_同期)
    return FakeAudioProcessor


def _start_worker_同期(run) -> None:
    """ワーカースレッドを起こさず、その場で処理を終わらせる。"""
    _TRANSCRIPTION_MODULE.jobs.save_job(
        run.job_id, _TRANSCRIPTION_MODULE._initial_job(run)
    )
    _TRANSCRIPTION_MODULE.run_job(run)


@pytest.fixture
def ワーカーを起動しない(monkeypatch) -> list:
    """ジョブ登録だけ行い、処理は走らせない。進捗表示の検証に使う。

    Returns:
        起動を要求された実行条件のリスト。
    """
    requested = []

    def 記録するだけ(run) -> None:
        requested.append(run)
        _TRANSCRIPTION_MODULE.jobs.save_job(
            run.job_id, _TRANSCRIPTION_MODULE._initial_job(run)
        )

    monkeypatch.setattr(_TRANSCRIPTION_MODULE, "start_worker", 記録するだけ)
    return requested


@pytest.fixture
def 話者照合をモックにする(monkeypatch) -> FakeSpeakerIdentifier:
    """声紋モデルの読み込みを避け、常に Unknown を返す照合器に差し替える。"""
    identifier = FakeSpeakerIdentifier()
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    monkeypatch.setattr(
        _IDENTIFICATION_MODULE,
        "get_cached_speaker_identifier",
        lambda **kwargs: identifier,
    )
    return identifier
