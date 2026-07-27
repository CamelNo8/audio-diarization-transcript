"""Step 2（マッチング）ルートの特性テスト。

マッチング本体は別プロセスで動くため、プロセス起動をモックする（規約2.5）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import src.web.jobs as jobs

SCRIPT_TXT = "# 場面\n太郎:やあ\n".encode("utf-8")


class Testマッチング:
    """Step 2。マッチング本体は別プロセスなのでプロセス起動をモックする。"""

    @pytest.fixture
    def マッチングプロセスをモックにする(self, monkeypatch):
        呼び出し引数 = []

        class _FakeProcess:
            returncode = 0

            def __init__(self, output_path: str) -> None:
                self._output_path = output_path

            async def communicate(self):
                Path(self._output_path).write_text("id,text\n", encoding="utf-8")
                return b"matching done\n", None

        async def fake_exec(*args, **kwargs):
            呼び出し引数.append(args)
            return _FakeProcess(args[-1])

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        return 呼び出し引数

    @pytest.fixture
    def 済みジョブ(self, 作業ディレクトリ, ジョブ保存先):
        work_dir = 作業ディレクトリ
        srt_path = work_dir / "transcription.srt"
        srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\n[太郎] やあ\n", "utf-8")
        job_id = "20260727-120000-abc123"
        jobs.save_job(
            job_id,
            {"job_id": job_id, "srt_path": str(srt_path), "clusters": []},
        )
        return job_id

    def test_ジョブIDが無ければ422になる(self, client):
        # ルート側にも「Step 1 を先に」の案内はあるが、job_id は必須フォーム値
        # なので、値が届かない要求は FastAPI の検証で先に弾かれる。
        response = client.post(
            "/process/matching",
            files={"script_file": ("台本.txt", SCRIPT_TXT, "text/plain")},
        )

        assert response.status_code == 422

    def test_存在しないジョブIDはエラーになる(self, client, ジョブ保存先):
        response = client.post(
            "/process/matching",
            files={"script_file": ("台本.txt", SCRIPT_TXT, "text/plain")},
            data={"job_id": "20260727-120000-zzzzzz"},
        )

        assert response.status_code == 200
        assert "見つかりません" in response.text

    def test_テキスト台本は台本CSVへ変換されて渡される(
        self, client, 済みジョブ, マッチングプロセスをモックにする, 作業ディレクトリ
    ):
        work_dir = 作業ディレクトリ

        response = client.post(
            "/process/matching",
            files={"script_file": ("台本.txt", SCRIPT_TXT, "text/plain")},
            data={"job_id": 済みジョブ, "output_csv_name": "対応表.csv"},
        )

        assert response.status_code == 200
        assert (work_dir / "対応表.csv").is_file()
        変換後の台本 = work_dir / "upload_script_台本.csv"
        assert 変換後の台本.is_file()
        assert "dialogue" in 変換後の台本.read_text(encoding="utf-8-sig")

    def test_CSV台本はそのまま渡される(
        self, client, 済みジョブ, マッチングプロセスをモックにする, 作業ディレクトリ
    ):
        work_dir = 作業ディレクトリ
        中身 = "id,scene_id,type,speaker,contents\n1,,dialogue,,やあ\n".encode("utf-8")

        client.post(
            "/process/matching",
            files={"script_file": ("台本.csv", 中身, "text/csv")},
            data={"job_id": 済みジョブ},
        )

        assert (work_dir / "upload_script_台本.csv").read_bytes() == 中身

    def test_プロセスが異常終了するとエラー表示になる(
        self, client, 済みジョブ, monkeypatch, 作業ディレクトリ
    ):
        class _FailingProcess:
            returncode = 1

            async def communicate(self):
                return b"boom\n", None

        async def fake_exec(*args, **kwargs):
            return _FailingProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        response = client.post(
            "/process/matching",
            files={"script_file": ("台本.txt", b"a\n", "text/plain")},
            data={"job_id": 済みジョブ},
        )

        assert response.status_code == 200
        assert "rc=1" in response.text
