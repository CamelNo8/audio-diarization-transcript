"""Web ルートの特性テスト（FastAPI TestClient によるスモーク）。

全エンドポイントについて、パス・メソッド・ステータス・返るテンプレートを
分割前の挙動として固定する。重いモデルとサブプロセスはモックする（規約2.5）。
出力ファイル名はフォームから自由に指定できるため、``../`` や絶対パスで
作業ディレクトリの外へ書き出せないことも併せて固定する（規約7.1）。
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path

import pytest

import app as app_module

WAV_BYTES = b"RIFF----WAVEfmt "
SCRIPT_TXT = "# 場面\n太郎:やあ\n".encode("utf-8")


def _対応表CSV() -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["start_time", "end_time", "speaker", "subtitle_text"])
    writer.writerow(["00:00:01,000", "00:00:02,000", "アイ", "こんにちは"])
    return buf.getvalue().encode("utf-8-sig")


def _文字起こしCSV(path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker", "text", "cosine_distance"])
        writer.writerow(
            ["00:00:01:000", "00:00:02:000", "Unknown_1", "こんにちは", "1.000000"]
        )


class Test画面の表示:
    def test_トップページが表示される(self, client):
        assert client.get("/").status_code == 200

    def test_声紋DB一覧ページが表示される(self, client):
        assert client.get("/databases").status_code == 200

    def test_存在しないジョブはエラー表示になる(self, client):
        response = client.get("/unknowns/does-not-exist")

        assert response.status_code == 200
        assert "見つかりません" in response.text


class Test文字起こし:
    """Step 1。重いモデルはモックし、入力検証と生成物だけを見る。"""

    def test_ffmpegが無ければエラー表示になる(
        self, client, 文字起こしをモックにする, monkeypatch
    ):
        monkeypatch.setattr(app_module.shutil, "which", lambda name: None)

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert response.status_code == 200
        assert "ffmpeg" in response.text

    def test_HFトークンが無ければエラー表示になる(
        self, client, 文字起こしをモックにする, monkeypatch
    ):
        monkeypatch.delenv("HF_TOKEN", raising=False)

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert response.status_code == 200
        assert "Hugging Face Token" in response.text

    def test_成功するとSRTとジョブが作られる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        work_dir = 作業ディレクトリ
        clusters_root = ジョブ保存先

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"output_srt_name": "結果.srt"},
        )

        assert response.status_code == 200
        assert (work_dir / "結果.srt").is_file()
        assert (work_dir / "結果.csv").is_file()
        job_files = list(clusters_root.glob("*/job.json"))
        assert len(job_files) == 1
        job = json.loads(job_files[0].read_text(encoding="utf-8"))
        assert job["srt_filename"] == "結果.srt"
        assert len(job["clusters"]) == 1

    def test_話者数の指定がプロセッサへ渡る(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"num_speakers": "3"},
        )

        assert 文字起こしをモックにする.last.known_num_speakers == 3

    def test_数値でない話者数は指定なしとして扱われる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"num_speakers": "たくさん"},
        )

        assert 文字起こしをモックにする.last.known_num_speakers is None

    @pytest.mark.parametrize(
        ("quality", "custom", "expected"),
        [
            ("medium", "", "medium"),
            ("__custom__", "my/whisper", "my/whisper"),
            ("__custom__", "", "mlx-community/whisper-large-v3-mlx"),
            ("", "", "mlx-community/whisper-large-v3-mlx"),
        ],
    )
    def test_Whisperモデルは品質指定とカスタム入力から決まる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
        quality,
        custom,
        expected,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"whisper_quality": quality, "whisper_custom_model": custom},
        )

        assert 文字起こしをモックにする.last.kwargs["mlx_model_id"] == expected

    def test_保存先DB未選択で声紋をアップロードするとエラーになる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        response = client.post(
            "/process/transcription",
            files=[
                ("audio_file", ("a.wav", WAV_BYTES, "audio/wav")),
                ("registry_files", ("太郎.wav", WAV_BYTES, "audio/wav")),
            ],
            data={"db_choice": "none"},
        )

        assert response.status_code == 200
        assert "保存先DB" in response.text

    def test_新規DBを指定すると声紋が登録される(
        self,
        client,
        文字起こしをモックにする,
        話者照合をモックにする,
        作業ディレクトリ,
        ジョブ保存先,
        声紋DBルート,
    ):
        db_root = 声紋DBルート

        response = client.post(
            "/process/transcription",
            files=[
                ("audio_file", ("a.wav", WAV_BYTES, "audio/wav")),
                ("registry_files", ("太郎.wav", WAV_BYTES, "audio/wav")),
            ],
            data={"db_choice": "new", "db_new_name": "新DB"},
        )

        assert response.status_code == 200
        assert (db_root / "新DB" / "太郎.wav").is_file()
        assert 話者照合をモックにする.registered == [
            ("太郎", db_root / "新DB" / "太郎.wav")
        ]

    def test_存在しない既存DBを選ぶとエラーになる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        声紋DBルート,
    ):
        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"db_choice": "existing", "db_existing_name": "無いDB"},
        )

        assert response.status_code == 200
        assert "DBエラー" in response.text


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
        app_module._save_job(
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


@pytest.fixture
def 話者入りDB(声紋DBルート) -> Path:
    """話者ファイルを1つ持つ声紋DB を用意する。"""
    db_dir = 声紋DBルート / "テストDB"
    db_dir.mkdir()
    (db_dir / "太郎.wav").write_bytes(WAV_BYTES)
    return db_dir


class Test声紋DBのAPI:
    def test_DB一覧がJSONで返る(self, client, 話者入りDB):
        response = client.get("/api/databases")

        assert response.status_code == 200
        assert [db["name"] for db in response.json()["databases"]] == ["テストDB"]

    def test_DBを作成できる(self, client, 声紋DBルート):
        response = client.post("/api/databases", data={"name": "新DB"})

        assert response.status_code == 200
        assert (声紋DBルート / "新DB").is_dir()

    def test_無効なDB名はエラー表示になる(self, client, 声紋DBルート):
        response = client.post("/api/databases", data={"name": "  "})

        assert response.status_code == 200
        assert "DB名が無効" in response.text

    def test_同名のDBは作成できない(self, client, 話者入りDB):
        response = client.post("/api/databases", data={"name": "テストDB"})

        assert response.status_code == 200
        assert "既に存在します" in response.text

    def test_DBを削除できる(self, client, 話者入りDB):
        response = client.delete("/api/databases/テストDB")

        assert response.status_code == 200
        assert not 話者入りDB.exists()

    def test_存在しないDBの削除はエラー表示になる(self, client, 声紋DBルート):
        response = client.delete("/api/databases/無いDB")

        assert response.status_code == 200
        assert "存在しません" in response.text

    def test_話者一覧が返る(self, client, 話者入りDB):
        response = client.get("/api/databases/テストDB/speakers")

        assert response.status_code == 200
        assert "太郎" in response.text

    def test_存在しないDBの話者一覧はエラー表示になる(self, client, 声紋DBルート):
        response = client.get("/api/databases/無いDB/speakers")

        assert response.status_code == 200
        assert "存在しません" in response.text

    def test_話者ファイルをアップロードできる(
        self, client, 話者入りDB, 作業ディレクトリ
    ):
        response = client.post(
            "/api/databases/テストDB/speakers/upload",
            files=[("files", ("花子.wav", WAV_BYTES, "audio/wav"))],
        )

        assert response.status_code == 200
        assert (話者入りDB / "花子.wav").is_file()

    def test_対応していない拡張子は取り込まれない(
        self, client, 話者入りDB, 作業ディレクトリ
    ):
        client.post(
            "/api/databases/テストDB/speakers/upload",
            files=[("files", ("メモ.txt", b"hello", "text/plain"))],
        )

        assert not (話者入りDB / "メモ.txt").exists()

    def test_話者名を変更できる(self, client, 話者入りDB):
        response = client.post(
            "/api/databases/テストDB/speakers/太郎.wav/rename",
            data={"new_name": "次郎"},
        )

        assert response.status_code == 200
        assert (話者入りDB / "次郎.wav").is_file()
        assert not (話者入りDB / "太郎.wav").exists()

    def test_話者ファイルを削除できる(self, client, 話者入りDB):
        response = client.delete("/api/databases/テストDB/speakers/太郎.wav")

        assert response.status_code == 200
        assert not (話者入りDB / "太郎.wav").exists()

    def test_話者音声を取得できる(self, client, 話者入りDB):
        response = client.get("/api/databases/テストDB/speakers/太郎.wav/audio")

        assert response.status_code == 200
        assert response.content == WAV_BYTES

    def test_存在しない話者音声は404になる(self, client, 話者入りDB):
        response = client.get("/api/databases/テストDB/speakers/居ない.wav/audio")

        assert response.status_code == 404

    def test_切り出し範囲が未指定ならエラー表示になる(
        self, client, 話者入りDB, 作業ディレクトリ
    ):
        response = client.post(
            "/api/databases/テストDB/speakers/太郎.wav/trim",
            data={"start": "", "end": ""},
        )

        assert response.status_code == 200
        assert "切り出し範囲" in response.text

    def test_終了が開始より前ならエラー表示になる(
        self, client, 話者入りDB, 作業ディレクトリ
    ):
        response = client.post(
            "/api/databases/テストDB/speakers/太郎.wav/trim",
            data={"start": "2", "end": "1"},
        )

        assert response.status_code == 200
        assert "終了時間は開始時間より後" in response.text

    def test_DB一覧フラグメントが返る(self, client, 話者入りDB):
        response = client.get("/api/databases/list")

        assert response.status_code == 200
        assert "テストDB" in response.text

    def test_選択肢フラグメントが返る(self, client, 話者入りDB):
        response = client.get(
            "/api/databases/select-options", params={"selected": "テストDB"}
        )

        assert response.status_code == 200
        assert "テストDB" in response.text


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

        app_module._save_job(
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
        job = app_module._load_job(未解決ジョブ)
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
        assert app_module._load_job(未解決ジョブ)["clusters"][0]["resolved"] is False

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
