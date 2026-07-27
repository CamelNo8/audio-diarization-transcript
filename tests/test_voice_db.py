"""声紋データベース操作の特性テスト（現 voice_database.py）。

Phase 3 で src/voice_db/registry.py へ移設する。
DB名・ファイル名の検証（規約7.1）を含めてここで固定する。
"""

from __future__ import annotations

import pytest

import voice_database as vdb


@pytest.fixture(autouse=True)
def 声紋DBのルートを一時ディレクトリにする(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_DB_ROOT", str(tmp_path / "voice_databases"))


def _add_dummy_audio(db_name: str, filename: str, tmp_path) -> None:
    src = tmp_path / filename
    src.write_bytes(b"dummy audio")
    vdb.add_speaker_file(db_name, src)


class TestDB名の検証:
    @pytest.mark.parametrize("name", ["会議A", "project-1", " 前後空白あり "])
    def test_使える名前は前後の空白を除いて返される(self, name):
        assert vdb.sanitize_name(name) == name.strip()

    @pytest.mark.parametrize(
        "invalid", ["", "   ", ".", "..", "a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b']
    )
    def test_使えない名前はNoneになる(self, invalid):
        assert vdb.sanitize_name(invalid) is None


class TestDBの作成と一覧と削除:
    def test_作成したDBが一覧に現れる(self):
        vdb.create_database("会議A")

        databases = vdb.list_databases()

        assert [d["name"] for d in databases] == ["会議A"]
        assert databases[0]["speaker_count"] == 0

    def test_DBが無いとき一覧は空になる(self):
        assert vdb.list_databases() == []

    def test_同名のDBを作ろうとするとValueErrorになる(self):
        vdb.create_database("会議A")

        with pytest.raises(ValueError, match="既に存在します"):
            vdb.create_database("会議A")

    def test_無効な名前のDBは作成できない(self):
        with pytest.raises(ValueError, match="無効なデータベース名"):
            vdb.create_database("../脱出")

    def test_存在しないDBの参照はFileNotFoundErrorになる(self):
        with pytest.raises(FileNotFoundError):
            vdb.database_dir("無いDB")

    def test_削除したDBは一覧から消える(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        vdb.delete_database("会議A")

        assert vdb.list_databases() == []


class Test話者ファイルの登録と一覧:
    def test_登録した話者が一覧に現れる(self, tmp_path):
        vdb.create_database("会議A")

        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        speakers = vdb.list_speakers("会議A")
        assert len(speakers) == 1
        assert speakers[0]["filename"] == "太郎.wav"
        assert speakers[0]["speaker_name"] == "太郎"
        assert speakers[0]["size_bytes"] == len(b"dummy audio")

    def test_対応外の拡張子は登録できない(self, tmp_path):
        vdb.create_database("会議A")
        src = tmp_path / "メモ.txt"
        src.write_text("これは音声ではない", encoding="utf-8")

        with pytest.raises(ValueError, match="対応していない拡張子"):
            vdb.add_speaker_file("会議A", src)

    def test_音声以外のファイルは一覧に含まれない(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        (vdb.database_dir("会議A") / "メモ.txt").write_text(
            "無視される", encoding="utf-8"
        )

        assert [s["filename"] for s in vdb.list_speakers("会議A")] == ["太郎.wav"]

    def test_ディレクトリを跨ぐファイル名は登録できない(self, tmp_path):
        vdb.create_database("会議A")
        src = tmp_path / "太郎.wav"
        src.write_bytes(b"dummy audio")

        with pytest.raises(ValueError, match="無効なファイル名"):
            vdb.add_speaker_file("会議A", src, dest_filename="../脱出.wav")

    def test_同名で登録すると上書きされる(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        updated = tmp_path / "更新.wav"
        updated.write_bytes(b"new audio content")

        vdb.add_speaker_file("会議A", updated, dest_filename="太郎.wav")

        assert len(vdb.list_speakers("会議A")) == 1
        assert (
            vdb.speaker_path("会議A", "太郎.wav").read_bytes() == b"new audio content"
        )


class Test話者ファイルの参照と削除:
    def test_ディレクトリを跨ぐファイル名の参照は拒否される(self):
        vdb.create_database("会議A")

        with pytest.raises(ValueError, match="無効なファイル名"):
            vdb.speaker_path("会議A", "../../etc/passwd")

    def test_存在しない話者ファイルの参照はFileNotFoundErrorになる(self):
        vdb.create_database("会議A")

        with pytest.raises(FileNotFoundError):
            vdb.speaker_path("会議A", "居ない.wav")

    def test_削除した話者は一覧から消える(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        vdb.delete_speaker("会議A", "太郎.wav")

        assert vdb.list_speakers("会議A") == []


class Test話者名の変更:
    def test_拡張子を保ったままリネームされる(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        dst = vdb.rename_speaker("会議A", "太郎.wav", "花子")

        assert dst.name == "花子.wav"
        assert [s["speaker_name"] for s in vdb.list_speakers("会議A")] == ["花子"]

    def test_同じ名前へのリネームは何もしない(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        dst = vdb.rename_speaker("会議A", "太郎.wav", "太郎")

        assert dst.name == "太郎.wav"

    def test_既存の話者名へのリネームは拒否される(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        _add_dummy_audio("会議A", "花子.wav", tmp_path)

        with pytest.raises(ValueError, match="同名の話者が既に存在します"):
            vdb.rename_speaker("会議A", "太郎.wav", "花子")

    def test_無効な話者名へのリネームは拒否される(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        with pytest.raises(ValueError, match="無効な話者名"):
            vdb.rename_speaker("会議A", "太郎.wav", "../脱出")
