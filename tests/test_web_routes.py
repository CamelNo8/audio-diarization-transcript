"""Web ルートの入力検証（app.py）。

出力ファイル名はフォームから自由に指定できるため、``../`` や絶対パスで
作業ディレクトリの外へ書き出せてはならない（規約7.1）。
"""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def 作業ディレクトリを一時ディレクトリにする(tmp_path, monkeypatch):
    # ``../`` を数階層たどっても tmp_path の中に収まるよう深い位置に作る。
    # こうするとテストの検査範囲が tmp_path 内で完結する。
    work_dir = tmp_path / "a" / "b" / "temp"
    work_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "TEMP_DIR", work_dir)
    return work_dir


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _対応表CSV() -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["start_time", "end_time", "speaker", "subtitle_text"])
    writer.writerow(["00:00:01,000", "00:00:02,000", "アイ", "こんにちは"])
    return buf.getvalue().encode("utf-8-sig")


class Test字幕生成の出力ファイル名:
    def test_通常の名前は作業ディレクトリに生成される(
        self, client, 作業ディレクトリを一時ディレクトリにする
    ):
        work_dir = 作業ディレクトリを一時ディレクトリにする

        response = client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": "subtitles.srt"},
        )

        assert response.status_code == 200
        assert (work_dir / "subtitles.srt").is_file()

    @pytest.mark.parametrize(
        "output_name", ["../escaped.srt", "../../escaped.srt", "sub/dir.srt"]
    )
    def test_相対パスを含む名前でも作業ディレクトリの外に書かれない(
        self, client, 作業ディレクトリを一時ディレクトリにする, output_name, tmp_path
    ):
        work_dir = 作業ディレクトリを一時ディレクトリにする

        client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": output_name},
        )

        # 作業ディレクトリ内に既定名で生成され、外には1つも出ていないこと
        assert list(work_dir.glob("*.srt"))
        外に出たファイル = [p for p in tmp_path.rglob("*.srt") if p.parent != work_dir]
        assert 外に出たファイル == []

    def test_絶対パスを指定しても作業ディレクトリの外に書かれない(
        self, client, 作業ディレクトリを一時ディレクトリにする, tmp_path
    ):
        work_dir = 作業ディレクトリを一時ディレクトリにする
        外部パス = tmp_path / "outside.srt"

        client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": str(外部パス)},
        )

        assert not 外部パス.exists()
        assert list(work_dir.glob("*.srt"))

    def test_空の名前でも既定名で生成される(
        self, client, 作業ディレクトリを一時ディレクトリにする
    ):
        work_dir = 作業ディレクトリを一時ディレクトリにする

        response = client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": ""},
        )

        assert response.status_code == 200
        assert list(work_dir.glob("*.srt"))


class Testダウンロード:
    def test_存在しないファイルは404になる(
        self, client, 作業ディレクトリを一時ディレクトリにする
    ):
        assert client.get("/download/missing.srt").status_code == 404

    @pytest.mark.parametrize(
        "filename", ["../app.py", "..%2Fapp.py", "%2e%2e%2fapp.py"]
    )
    def test_パス区切りを含む要求は成功しない(
        self, client, 作業ディレクトリを一時ディレクトリにする, filename
    ):
        assert client.get(f"/download/{filename}").status_code == 404

    def test_作業ディレクトリ内のファイルは取得できる(
        self, client, 作業ディレクトリを一時ディレクトリにする
    ):
        work_dir = 作業ディレクトリを一時ディレクトリにする
        (work_dir / "result.srt").write_text("1\n", encoding="utf-8")

        response = client.get("/download/result.srt")

        assert response.status_code == 200
        assert response.content == b"1\n"


class Test画面の表示:
    def test_トップページが表示される(self, client):
        assert client.get("/").status_code == 200

    def test_声紋DB一覧ページが表示される(self, client):
        assert client.get("/databases").status_code == 200

    def test_存在しないジョブはエラー表示になる(self, client):
        response = client.get("/unknowns/does-not-exist")

        assert response.status_code == 200
        assert "見つかりません" in response.text
