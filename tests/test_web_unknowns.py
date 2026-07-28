"""未知話者の事後ラベル付けの特性テスト。"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

import src.web.jobs as jobs
from src.common.audio import probe_duration_sec

WAV_BYTES = b"RIFF----WAVEfmt "


def _正弦波を書く(path: Path, seconds: int) -> None:
    """ffmpeg で扱える本物の WAV を作る（切り出しの検証に使う）。"""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", str(path),
        ],
        check=True,
    )


def _文字起こしCSV(path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker", "text", "cosine_distance"])
        writer.writerow(
            ["00:00:01:000", "00:00:02:000", "Unknown_1", "こんにちは", "1.000000"]
        )


@pytest.fixture
def 話者入りDB(声紋DBルート) -> Path:
    """話者ファイルを1つ持つ声紋DB を用意する。"""
    db_dir = 声紋DBルート / "テストDB"
    db_dir.mkdir()
    (db_dir / "太郎.wav").write_bytes(WAV_BYTES)
    return db_dir


class Test未知話者のラベル付け:
    @pytest.fixture
    def 未解決ジョブ(self, ジョブ保存先, 作業ディレクトリ) -> str:
        job_id = "20260727-120000-abc123"
        job_dir = ジョブ保存先 / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "clip_0.wav").write_bytes(WAV_BYTES)

        work_dir = 作業ディレクトリ
        csv_path = work_dir / "transcription.csv"
        _文字起こしCSV(csv_path)
        srt_path = work_dir / "transcription.srt"
        srt_path.write_text("1\n", encoding="utf-8")

        jobs.save_job(
            job_id,
            {
                "job_id": job_id,
                "csv_path": str(csv_path),
                "csv_filename": csv_path.name,
                "srt_path": str(srt_path),
                "srt_filename": srt_path.name,
                "db_name": None,
                "threshold": 0.5,
                "embedding_model": "pyannote/embedding",
                "clusters": [
                    {
                        "cluster_id": "0",
                        "unknown_label": "Unknown_1",
                        "clip_filename": "clip_0.wav",
                        "segment_start": 1.0,
                        "segment_end": 2.0,
                        "distance": 1.0,
                        "candidate_distances": [],
                        "resolved": False,
                        "resolved_name": None,
                    }
                ],
                "created_at": "2026-07-27T12:00:00",
            },
        )
        return job_id

    def test_ラベル付け画面が表示される(self, client, 未解決ジョブ):
        response = client.get(f"/unknowns/{未解決ジョブ}")

        assert response.status_code == 200
        assert "Unknown_1" in response.text

    def test_クラスタ音声を取得できる(self, client, 未解決ジョブ):
        response = client.get(f"/unknowns/{未解決ジョブ}/clip/0")

        assert response.status_code == 200
        assert response.content == WAV_BYTES

    def test_存在しないジョブのクラスタ音声は404になる(self, client, ジョブ保存先):
        assert client.get("/unknowns/20260727-000000-zzzzzz/clip/0").status_code == 404

    def test_パス区切りを含むクラスタIDは音声を返さない(self, client, 未解決ジョブ):
        # ルート側にも多層防御の検査（400）があるが、パス区切りは経路解決の
        # 段階で弾かれるためここまで届かない。
        response = client.get(f"/unknowns/{未解決ジョブ}/clip/..%2F..%2Fjob")

        assert response.status_code == 404

    def test_存在しないクラスタ音声は404になる(self, client, 未解決ジョブ):
        assert client.get(f"/unknowns/{未解決ジョブ}/clip/9").status_code == 404

    def test_ラベル付けするとCSVの話者名が置換される(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "テストDB"},
        )

        assert response.status_code == 200
        job = jobs.load_job(未解決ジョブ)
        rows = list(csv.reader(open(job["csv_path"], encoding="utf-8-sig", newline="")))
        assert rows[1][2] == "花子"
        assert (話者入りDB / "花子.wav").is_file()
        assert job["clusters"][0]["resolved"] is True
        assert job["db_name"] == "テストDB"

    def test_ラベル付け済みのクラスタは再度ラベル付けできない(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "テストDB"},
        )

        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "次郎", "db_name": "テストDB"},
        )

        assert "既にラベル付け済み" in response.text

    def test_存在しないクラスタIDはエラー表示になる(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/9",
            data={"speaker_name": "花子", "db_name": "テストDB"},
        )

        assert "クラスタが見つかりません" in response.text

    def test_無効な話者名はエラー表示になる(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "  ", "db_name": "テストDB"},
        )

        assert "話者名が無効" in response.text

    def test_同名DBの新規作成要求は確認バナーを返す(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={
                "speaker_name": "花子",
                "db_name": "__new__",
                "new_db_name": "テストDB",
            },
        )

        assert response.status_code == 200
        assert jobs.load_job(未解決ジョブ)["clusters"][0]["resolved"] is False

    def test_ラベル付け済みの試聴はDBに保存されたファイルを指す(
        self, client, 未解決ジョブ, 話者入りDB, 話者照合をモックにする
    ):
        """切り出しが効いたかを画面で確認できるようにするため（元クリップではない）。"""
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "テストDB"},
        )

        期待するURL = (
            f"/api/databases/{quote('テストDB')}/speakers/{quote('花子.wav')}/audio"
        )
        assert 期待するURL in response.text

    def test_新規DBを作ってラベル付けできる(
        self,
        client,
        未解決ジョブ,
        声紋DBルート,
        話者照合をモックにする,
    ):
        response = client.post(
            f"/unknowns/{未解決ジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "__new__", "new_db_name": "新DB"},
        )

        assert response.status_code == 200
        assert (声紋DBルート / "新DB" / "花子.wav").is_file()


class Test切り出し範囲の指定:
    """フォームの clip_start / clip_end が DB へ保存する音声に効くこと。

    ffmpeg を実際に呼ぶ。ここが壊れると「範囲を指定したのに全体が保存される」
    という気づきにくい不具合になるため、本物の音声で押さえる。
    """

    @pytest.fixture
    def 十秒のクラスタを持つジョブ(self, ジョブ保存先, 作業ディレクトリ) -> str:
        job_id = "20260728-120000-crop00"
        job_dir = ジョブ保存先 / job_id
        job_dir.mkdir(parents=True)
        _正弦波を書く(job_dir / "clip_0.wav", 10)

        csv_path = 作業ディレクトリ / "transcription.csv"
        _文字起こしCSV(csv_path)
        jobs.save_job(
            job_id,
            {
                "job_id": job_id,
                "csv_path": str(csv_path),
                "csv_filename": csv_path.name,
                "srt_path": None,
                "db_name": None,
                "threshold": 0.5,
                "embedding_model": "pyannote/embedding",
                "clusters": [
                    {
                        "cluster_id": "0",
                        "unknown_label": "Unknown_1",
                        "clip_filename": "clip_0.wav",
                        "segment_start": 0.0,
                        "segment_end": 10.0,
                        "distance": 1.0,
                        "candidate_distances": [],
                        "resolved": False,
                        "resolved_name": None,
                    }
                ],
                "created_at": "2026-07-28T12:00:00",
            },
        )
        return job_id

    def test_指定した範囲だけが保存される(
        self, client, 十秒のクラスタを持つジョブ, 声紋DBルート, 話者照合をモックにする
    ):
        db_dir = 声紋DBルート / "テストDB"
        db_dir.mkdir()

        response = client.post(
            f"/unknowns/{十秒のクラスタを持つジョブ}/label/0",
            data={
                "speaker_name": "花子",
                "db_name": "テストDB",
                "clip_start": "3.00",
                "clip_end": "6.00",
            },
        )

        assert response.status_code == 200
        assert probe_duration_sec(db_dir / "花子.wav") == pytest.approx(3.0, abs=0.2)

    def test_範囲が空欄なら全体が保存される(
        self, client, 十秒のクラスタを持つジョブ, 声紋DBルート, 話者照合をモックにする
    ):
        db_dir = 声紋DBルート / "テストDB"
        db_dir.mkdir()

        client.post(
            f"/unknowns/{十秒のクラスタを持つジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "テストDB",
                  "clip_start": "", "clip_end": ""},
        )

        assert probe_duration_sec(db_dir / "花子.wav") == pytest.approx(10.0, abs=0.2)

    def test_終了が開始より前ならエラーになる(
        self, client, 十秒のクラスタを持つジョブ, 声紋DBルート, 話者照合をモックにする
    ):
        (声紋DBルート / "テストDB").mkdir()

        response = client.post(
            f"/unknowns/{十秒のクラスタを持つジョブ}/label/0",
            data={"speaker_name": "花子", "db_name": "テストDB",
                  "clip_start": "6.00", "clip_end": "3.00"},
        )

        assert "終了時間は開始時間より後" in response.text
