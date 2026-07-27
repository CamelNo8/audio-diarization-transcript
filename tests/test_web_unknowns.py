"""未知話者の事後ラベル付けの特性テスト。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import src.web.jobs as jobs

WAV_BYTES = b"RIFF----WAVEfmt "


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
