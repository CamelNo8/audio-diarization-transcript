"""画面表示・Step 3（字幕生成）・ダウンロードの特性テスト。

出力ファイル名はフォームから自由に指定できるため、``../`` や絶対パスで
作業ディレクトリの外へ書き出せないことを固定する（規約7.1）。
"""

from __future__ import annotations

import csv
import io

import pytest


def _対応表CSV() -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["start_time", "end_time", "speaker", "subtitle_text"])
    writer.writerow(["00:00:01,000", "00:00:02,000", "アイ", "こんにちは"])
    return buf.getvalue().encode("utf-8-sig")


class Test画面の表示:
    def test_トップページが表示される(self, client):
        assert client.get("/").status_code == 200

    def test_声紋DB一覧ページが表示される(self, client):
        assert client.get("/databases").status_code == 200

    def test_存在しないジョブはエラー表示になる(self, client):
        response = client.get("/unknowns/does-not-exist")

        assert response.status_code == 200
        assert "見つかりません" in response.text


class Test字幕生成の出力ファイル名:
    def test_通常の名前は作業ディレクトリに生成される(self, client, 作業ディレクトリ):
        work_dir = 作業ディレクトリ

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
        self, client, 作業ディレクトリ, output_name, tmp_path
    ):
        work_dir = 作業ディレクトリ

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
        self, client, 作業ディレクトリ, tmp_path
    ):
        work_dir = 作業ディレクトリ
        外部パス = tmp_path / "outside.srt"

        client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": str(外部パス)},
        )

        assert not 外部パス.exists()
        assert list(work_dir.glob("*.srt"))

    def test_空の名前でも既定名で生成される(self, client, 作業ディレクトリ):
        work_dir = 作業ディレクトリ

        response = client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", _対応表CSV(), "text/csv")},
            data={"output_srt_name": ""},
        )

        assert response.status_code == 200
        assert list(work_dir.glob("*.srt"))

    def test_字幕データが空ならエラー表示になる(self, client, 作業ディレクトリ):
        response = client.post(
            "/process/generation",
            files={"edited_csv": ("edited.csv", b"start_time,end_time\n", "text/csv")},
        )

        assert response.status_code == 200
        assert "読み込みに失敗" in response.text


class Testダウンロード:
    def test_存在しないファイルは404になる(self, client, 作業ディレクトリ):
        assert client.get("/download/missing.srt").status_code == 404

    @pytest.mark.parametrize(
        "filename", ["../app.py", "..%2Fapp.py", "%2e%2e%2fapp.py"]
    )
    def test_パス区切りを含む要求は成功しない(self, client, 作業ディレクトリ, filename):
        assert client.get(f"/download/{filename}").status_code == 404

    def test_作業ディレクトリ内のファイルは取得できる(self, client, 作業ディレクトリ):
        work_dir = 作業ディレクトリ
        (work_dir / "result.srt").write_text("1\n", encoding="utf-8")

        response = client.get("/download/result.srt")

        assert response.status_code == 200
        assert response.content == b"1\n"


class Test静的ファイル:
    def test_スタイルシートが配信される(self, client):
        response = client.get("/static/css/dads.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_スタイルシートにDADSのプライマリ色が含まれる(self, client):
        response = client.get("/static/css/dads.css")

        assert "#0017c1" in response.text


class Test手順ごとの画面:
    def test_トップにStep1の見出しが出る(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "文字起こし" in response.text

    def test_マッチング画面が表示される(self, client):
        assert client.get("/matching").status_code == 200

    def test_字幕生成画面が表示される(self, client):
        assert client.get("/generation").status_code == 200

    def test_ジョブIDが無いマッチング画面はStep1を促す(self, client):
        response = client.get("/matching")

        assert "Step 1" in response.text
        assert "disabled" in response.text

    def test_ジョブIDを渡すとStep1の出力名が表示される(
        self, client, ジョブ保存先, 作業ディレクトリ
    ):
        import src.web.jobs as jobs

        srt_path = 作業ディレクトリ / "転スラ_hard.srt"
        srt_path.write_text("1\n", encoding="utf-8")
        jobs.save_job(
            "test-job",
            {
                "job_id": "test-job",
                "status": jobs.STATUS_DONE,
                "srt_path": str(srt_path),
                "srt_filename": srt_path.name,
            },
        )

        response = client.get("/matching?job_id=test-job")

        assert response.status_code == 200
        assert "転スラ_hard.srt" in response.text

    def test_存在しないジョブIDでも画面は出る(self, client, ジョブ保存先):
        response = client.get("/matching?job_id=does-not-exist")

        assert response.status_code == 200
        assert "Step 1" in response.text

    @pytest.mark.parametrize("job_id", ["../../etc/passwd", "a/b", "."])
    def test_不正なジョブIDでもエラーにならない(self, client, ジョブ保存先, job_id):
        response = client.get("/matching", params={"job_id": job_id})

        assert response.status_code == 200

    def test_字幕生成画面はStep2の出力名を案内に出す(self, client):
        response = client.get("/generation", params={"csv": "対応表.csv"})

        assert response.status_code == 200
        assert "対応表.csv" in response.text
