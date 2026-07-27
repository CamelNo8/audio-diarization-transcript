"""Spark サーバーのジョブ状態がファイルに残ることの特性テスト。

メモリ辞書だった頃はプロセス再起動でジョブが消え、複数ワーカーでは
リクエストごとに別プロセスへ振られて「unknown job_id」になっていた。
ファイルを唯一の真実にしたことをここで固定する。重い推論は呼ばない。
"""

from __future__ import annotations

import pytest

import src.spark.server as server


@pytest.fixture(autouse=True)
def 作業ディレクトリ(tmp_path, monkeypatch):
    """ジョブ状態の置き場を tmp_path 配下へ差し替える。"""
    work_dir = tmp_path / "spark_jobs"
    work_dir.mkdir()
    monkeypatch.setattr(server, "WORK_DIR", work_dir)
    return work_dir


class TestジョブIDの検証:
    def test_正当なIDは作業ディレクトリ直下に解決される(self, 作業ディレクトリ):
        path = server.job_path("abc123def456")

        assert path.parent == 作業ディレクトリ
        assert path.name == "abc123def456.json"

    @pytest.mark.parametrize(
        "invalid_id", ["../etc", "a/b", "/absolute", "..", "job id", "job.id", ""]
    )
    def test_不正なIDはValueErrorになる(self, invalid_id):
        with pytest.raises(ValueError, match="無効な job_id"):
            server.job_path(invalid_id)


class Testジョブ状態の保存:
    def test_更新した内容がファイルに残る(self, 作業ディレクトリ):
        server._update_job("job0001", status="running")

        assert (作業ディレクトリ / "job0001.json").is_file()
        assert server.read_job("job0001")["status"] == "running"

    def test_指定したフィールドだけが書き換わる(self):
        server._update_job("job0001", status="queued", error=None)

        server._update_job("job0001", status="done")

        job = server.read_job("job0001")
        assert job == {"status": "done", "error": None}

    def test_ファイルを直接書き換えると読み込み結果も変わる(self, 作業ディレクトリ):
        # メモリにキャッシュしていないこと（別プロセスが書いた内容が見えること）
        server._update_job("job0001", status="running")

        (作業ディレクトリ / "job0001.json").write_text(
            '{"status": "done"}', encoding="utf-8"
        )

        assert server.read_job("job0001") == {"status": "done"}

    def test_存在しないジョブはNoneを返す(self):
        assert server.read_job("job9999") is None

    def test_不正なジョブIDは例外ではなくNoneを返す(self):
        assert server.read_job("../etc") is None


class TestジョブAPIの応答:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        return TestClient(server.app)

    def test_未知のジョブIDは404になる(self, client):
        assert client.get("/jobs/job9999").status_code == 404
        assert client.get("/jobs/job9999/result").status_code == 404
        assert client.get("/jobs/job9999/vocals").status_code == 404

    def test_処理中のジョブの結果要求は409になる(self, client):
        server._update_job("job0001", status="running", error=None)

        assert client.get("/jobs/job0001/result").status_code == 409

    def test_失敗したジョブの結果要求は500になる(self, client):
        server._update_job("job0001", status="error", error="boom")

        assert client.get("/jobs/job0001/result").status_code == 500

    def test_完了したジョブの結果が返る(self, client):
        server._update_job(
            "job0001", status="done", error=None, result={"num_speakers": 2}
        )

        response = client.get("/jobs/job0001/result")

        assert response.status_code == 200
        assert response.json() == {"num_speakers": 2}

    def test_状態の問い合わせにstatusとerrorが返る(self, client):
        server._update_job("job0001", status="error", error="boom")

        assert client.get("/jobs/job0001").json() == {
            "status": "error",
            "error": "boom",
        }

    def test_denoise_offなら声だけのWAVは404になる(self, client):
        server._update_job("job0001", status="done", error=None, vocals=None)

        assert client.get("/jobs/job0001/vocals").status_code == 404
