"""声紋DB 管理 API の特性テスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

WAV_BYTES = b"RIFF----WAVEfmt "


@pytest.fixture
def 連結をモックにする(monkeypatch) -> list:
    """ffmpeg を起動せず、呼ばれた引数を記録して空の wav を置く。"""
    import src.voice_db.registry as vdb

    calls: list = []

    def fake_concat(src_paths, dst):
        calls.append((list(src_paths), dst))
        Path(dst).write_bytes(WAV_BYTES)

    monkeypatch.setattr(vdb, "concat_audio", fake_concat)
    return calls


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


class Test複数クリップをまとめて1人として登録する:
    URL = "/api/databases/テストDB/speakers/clips"

    def test_複数クリップが1人の話者として登録される(
        self, client, 話者入りDB, 作業ディレクトリ, 連結をモックにする
    ):
        response = client.post(
            self.URL,
            data={"speaker_name": "花子"},
            files=[
                ("files", ("a.wav", WAV_BYTES, "audio/wav")),
                ("files", ("b.wav", WAV_BYTES, "audio/wav")),
            ],
        )

        assert response.status_code == 200
        assert (話者入りDB / "花子.wav").is_file()

    def test_素材がクリップフォルダに残る(
        self, client, 話者入りDB, 作業ディレクトリ, 連結をモックにする
    ):
        client.post(
            self.URL,
            data={"speaker_name": "花子"},
            files=[
                ("files", ("a.wav", WAV_BYTES, "audio/wav")),
                ("files", ("b.wav", WAV_BYTES, "audio/wav")),
            ],
        )

        clips = sorted((話者入りDB / "_clips" / "花子").iterdir())
        assert [c.name for c in clips] == ["001_a.wav", "002_b.wav"]

    def test_話者名が空ならエラー表示になる(
        self, client, 話者入りDB, 作業ディレクトリ, 連結をモックにする
    ):
        response = client.post(
            self.URL,
            data={"speaker_name": "  "},
            files=[("files", ("a.wav", WAV_BYTES, "audio/wav"))],
        )

        assert response.status_code == 200
        assert "話者名" in response.text
        assert not (話者入りDB / "_clips").exists()

    def test_対応していない拡張子だけならエラー表示になる(
        self, client, 話者入りDB, 作業ディレクトリ, 連結をモックにする
    ):
        response = client.post(
            self.URL,
            data={"speaker_name": "花子"},
            files=[("files", ("メモ.txt", b"hello", "text/plain"))],
        )

        assert response.status_code == 200
        assert "クリップ" in response.text
        assert not (話者入りDB / "花子.wav").exists()

    def test_存在しないDBならエラー表示になる(
        self, client, 声紋DBルート, 作業ディレクトリ, 連結をモックにする
    ):
        response = client.post(
            "/api/databases/無いDB/speakers/clips",
            data={"speaker_name": "花子"},
            files=[("files", ("a.wav", WAV_BYTES, "audio/wav"))],
        )

        assert response.status_code == 200
        assert "存在しません" in response.text
