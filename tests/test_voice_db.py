"""声紋データベース操作の特性テスト。

DB名・ファイル名の検証（規約7.1）と、削除がゴミ箱への退避になることを固定する。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import src.voice_db.registry as vdb
from src.diarization.registry import collect_registry_files


@pytest.fixture(autouse=True)
def 声紋DBのルートを一時ディレクトリにする(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_DB_ROOT", str(tmp_path / "voice_databases"))


@pytest.fixture
def 連結をモックにする(monkeypatch) -> list:
    """ffmpeg を起動せず、呼ばれた引数を記録して空の wav を置く。

    連結そのものは tests/test_audio_concat.py で確認済みなので、ここでは
    「どの素材をどの順で渡したか」だけを見る。
    """
    calls: list = []

    def fake_concat(src_paths, dst):
        calls.append((list(src_paths), dst))
        Path(dst).write_bytes(b"merged audio")

    monkeypatch.setattr(vdb, "concat_audio", fake_concat)
    return calls


def _add_dummy_audio(db_name: str, filename: str, tmp_path) -> None:
    src = tmp_path / filename
    src.write_bytes(b"dummy audio")
    vdb.add_speaker_file(db_name, src)


def _make_clip_files(base_dir, filenames: list[str]) -> list[Path]:
    """テスト用のダミークリップを作り、そのパスを返す。"""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in filenames:
        path = base_dir / name
        path.write_bytes(b"dummy clip")
        paths.append(path)
    return paths


class TestDB名の検証:
    @pytest.mark.parametrize("name", ["会議A", "project-1", " 前後空白あり "])
    def test_使える名前は前後の空白を除いて返される(self, name):
        assert vdb.sanitize_name(name) == name.strip()

    @pytest.mark.parametrize(
        "invalid",
        [
            "",
            "   ",
            ".",
            "..",
            ".trash",
            ".隠し",
            "a/b",
            "a\\b",
            "a:b",
            "a*b",
            "a?b",
            'a"b',
        ],
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


class Testゴミ箱への退避:
    """削除は不可逆にしない。確認ダイアログを押し間違えても手で戻せること。"""

    def test_削除したDBはゴミ箱に残る(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        退避先 = vdb.delete_database("会議A")

        assert 退避先.is_dir()
        assert (退避先 / "太郎.wav").read_bytes() == b"dummy audio"
        assert 退避先.parent == vdb.get_root() / vdb.TRASH_DIR_NAME

    def test_退避先の名前は日時とDB名からなる(self):
        vdb.create_database("会議A")

        退避先 = vdb.delete_database("会議A")

        assert re.fullmatch(r"\d{8}-\d{6}_会議A", 退避先.name), (
            f"想定外の書式: {退避先.name}"
        )

    def test_削除した話者ファイルはゴミ箱に残る(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        退避先 = vdb.delete_speaker("会議A", "太郎.wav")

        assert 退避先.read_bytes() == b"dummy audio"
        assert re.fullmatch(r"\d{8}-\d{6}_会議A_太郎\.wav", 退避先.name)

    def test_同名を続けて削除しても上書きされない(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        一度目 = vdb.delete_speaker("会議A", "太郎.wav")

        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        二度目 = vdb.delete_speaker("会議A", "太郎.wav")

        assert 一度目 != 二度目
        assert 一度目.is_file() and 二度目.is_file()

    def test_ゴミ箱はDB一覧に出ない(self):
        vdb.create_database("会議A")
        vdb.delete_database("会議A")

        assert (vdb.get_root() / vdb.TRASH_DIR_NAME).is_dir()
        assert vdb.list_databases() == []

    def test_ゴミ箱をDBとして操作できない(self):
        vdb.create_database("会議A")
        vdb.delete_database("会議A")

        assert vdb.sanitize_name(vdb.TRASH_DIR_NAME) is None
        with pytest.raises(ValueError, match="無効なデータベース名"):
            vdb.database_dir(vdb.TRASH_DIR_NAME)
        with pytest.raises(ValueError, match="無効なデータベース名"):
            vdb.delete_database(vdb.TRASH_DIR_NAME)

    def test_ゴミ箱から戻せば復旧できる(self, tmp_path):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)
        退避先 = vdb.delete_database("会議A")

        # 利用者が手でやる復旧操作（ディレクトリを元の名前に戻すだけ）
        退避先.rename(vdb.get_root() / "会議A")

        assert [d["name"] for d in vdb.list_databases()] == ["会議A"]
        assert [s["filename"] for s in vdb.list_speakers("会議A")] == ["太郎.wav"]


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


class Test複数クリップをまとめて1人として登録する:
    """素材は ``_clips/<話者名>/`` に残し、繋いだ結果を ``<話者名>.wav`` にする。"""

    def test_素材が連番付きでクリップフォルダに保存される(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.wav", "b.wav"])

        vdb.add_speaker_clips("会議A", "太郎", clips)

        assert [p.name for p in vdb.list_clips("会議A", "太郎")] == [
            "001_a.wav",
            "002_b.wav",
        ]

    def test_繋いだ結果が話者名のwavとして登録される(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.wav", "b.wav"])

        dst = vdb.add_speaker_clips("会議A", "太郎", clips)

        assert dst.name == "太郎.wav"
        assert [s["speaker_name"] for s in vdb.list_speakers("会議A")] == ["太郎"]

    def test_素材の順番どおりに連結が呼ばれる(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.wav", "b.wav"])

        vdb.add_speaker_clips("会議A", "太郎", clips)

        呼ばれた素材, _ = 連結をモックにする[0]
        assert [p.name for p in 呼ばれた素材] == ["001_a.wav", "002_b.wav"]

    def test_同じ話者名で追加すると既存の素材込みで繋ぎ直される(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["a.wav"]))

        vdb.add_speaker_clips(
            "会議A", "太郎", _make_clip_files(tmp_path / "2回目", ["b.wav"])
        )

        呼ばれた素材, _ = 連結をモックにする[-1]
        assert [p.name for p in 呼ばれた素材] == ["001_a.wav", "002_b.wav"]

    def test_クリップが0本ならエラーになる(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")

        with pytest.raises(ValueError, match="クリップが指定されていません"):
            vdb.add_speaker_clips("会議A", "太郎", [])

    def test_無効な話者名は拒否される(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.wav"])

        with pytest.raises(ValueError, match="無効な話者名"):
            vdb.add_speaker_clips("会議A", "../脱出", clips)

    def test_対応していない拡張子は拒否される(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.txt"])

        with pytest.raises(ValueError, match="対応していない拡張子"):
            vdb.add_speaker_clips("会議A", "太郎", clips)

    def test_素材に残す名前を別に指定できる(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["tmp_xxxx.wav"])

        vdb.add_speaker_clips("会議A", "太郎", clips, ["元の名前.wav"])

        assert [p.name for p in vdb.list_clips("会議A", "太郎")] == ["001_元の名前.wav"]

    def test_名前の数がクリップの数と合わなければエラーになる(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        clips = _make_clip_files(tmp_path, ["a.wav", "b.wav"])

        with pytest.raises(ValueError, match="数が合いません"):
            vdb.add_speaker_clips("会議A", "太郎", clips, ["a.wav"])


class Test素材フォルダが照合や一覧に混ざらない:
    def test_クリップフォルダは話者一覧に出ない(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["a.wav"]))

        names = [s["speaker_name"] for s in vdb.list_speakers("会議A")]

        assert names == ["太郎"]

    def test_クリップフォルダは照合対象として読まれない(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["a.wav"]))

        registry_files = collect_registry_files(vdb.database_dir("会議A"))

        assert list(registry_files) == ["太郎"]


class Test改名と削除が素材フォルダにも追従する:
    def test_改名すると素材フォルダも一緒に移る(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["a.wav"]))

        vdb.rename_speaker("会議A", "太郎.wav", "花子")

        assert [p.name for p in vdb.list_clips("会議A", "花子")] == ["001_a.wav"]
        assert vdb.list_clips("会議A", "太郎") == []

    def test_削除すると素材フォルダも消える(self, tmp_path, 連結をモックにする):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["a.wav"]))

        vdb.delete_speaker("会議A", "太郎.wav")

        assert vdb.list_clips("会議A", "太郎") == []

    def test_削除後に同じ名前で登録しても古い素材が混ざらない(
        self, tmp_path, 連結をモックにする
    ):
        vdb.create_database("会議A")
        vdb.add_speaker_clips("会議A", "太郎", _make_clip_files(tmp_path, ["古い.wav"]))
        vdb.delete_speaker("会議A", "太郎.wav")

        vdb.add_speaker_clips(
            "会議A", "太郎", _make_clip_files(tmp_path / "2回目", ["新しい.wav"])
        )

        assert [p.name for p in vdb.list_clips("会議A", "太郎")] == ["001_新しい.wav"]


class Test話者一覧に長さと素材の本数が入る:
    """登録音声が短すぎないかを画面で確かめられるようにするための情報。"""

    @pytest.fixture
    def 音声長をモックにする(self, monkeypatch):
        monkeypatch.setattr(vdb, "probe_duration_sec", lambda path: 4.5)

    def test_登録音声の長さが入る(self, tmp_path, 音声長をモックにする):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        speakers = vdb.list_speakers("会議A")

        assert speakers[0]["duration_sec"] == 4.5

    def test_素材の本数が入る(self, tmp_path, 連結をモックにする, 音声長をモックにする):
        vdb.create_database("会議A")
        vdb.add_speaker_clips(
            "会議A", "太郎", _make_clip_files(tmp_path, ["a.wav", "b.wav"])
        )

        speakers = vdb.list_speakers("会議A")

        assert speakers[0]["clip_count"] == 2

    def test_素材を使っていない話者の本数は0になる(
        self, tmp_path, 音声長をモックにする
    ):
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        speakers = vdb.list_speakers("会議A")

        assert speakers[0]["clip_count"] == 0

    def test_長さが取れない音声はNoneになる(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vdb, "probe_duration_sec", lambda path: None)
        vdb.create_database("会議A")
        _add_dummy_audio("会議A", "太郎.wav", tmp_path)

        speakers = vdb.list_speakers("会議A")

        assert speakers[0]["duration_sec"] is None
