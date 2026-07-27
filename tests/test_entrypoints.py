"""エントリポイントの特性テスト。

分割で参照が切れると、テストは緑のままアプリだけが起動しなくなる。
起動経路そのものをここで固定する。

``main.py --help`` はサブプロセスで確認する。CLI が引数解析だけで
``SystemExit`` するところまで含めて、実際の起動と同じ経路を通すため。
"""

from __future__ import annotations

import subprocess
import sys

from src.config import PROJECT_ROOT


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    """リポジトリのルートで Python を起動する。"""
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestWebアプリのエントリポイント:
    def test_ルートのapp_pyからappをimportできる(self):
        import app

        assert app.app.title

    def test_uvicornが解決する名前でappを取得できる(self):
        # `uvicorn app:app` と同じ解決をたどる
        import importlib

        module = importlib.import_module("app")

        assert getattr(module, "app", None) is not None

    def test_全ルートが登録されている(self):
        import app

        paths = {r.path for r in app.app.routes if hasattr(r, "methods")}

        assert {"/", "/databases", "/download/{filename}"} <= paths
        assert {
            "/process/transcription",
            "/process/matching",
            "/process/generation",
        } <= paths
        assert "/unknowns/{job_id}/label/{cluster_id}" in paths


class TestCLIのエントリポイント:
    def test_helpが終了コード0で表示される(self):
        result = _run_module("main.py", "--help")

        assert result.returncode == 0, result.stderr
        assert "usage" in result.stdout.lower()

    def test_引数なしは終了コード0以外になる(self):
        result = _run_module("main.py")

        assert result.returncode != 0


class TestSpark連携のエントリポイント:
    def test_サーバーモジュールをimportできる(self):
        from src.spark import server

        assert server.app.title

    def test_クライアントモジュールをimportできる(self):
        from src.spark import client

        assert callable(client.transcribe_on_spark)

    def test_クライアントは引数なしで使い方を示して終了する(self):
        result = _run_module("-m", "src.spark.client")

        assert result.returncode == 1
