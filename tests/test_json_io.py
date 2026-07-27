"""アトミックな JSON 読み書きの特性テスト。

ジョブ状態はワーカースレッドが書いている最中に、別のリクエストが読む。
「書き途中の中身が読めてしまわない」ことがこのモジュールの存在理由なので、
そこを固定する。
"""

from __future__ import annotations

import json

import pytest

from src.common.json_io import read_json, write_json


class TestJSONの書き出し:
    def test_書いた内容がそのまま読み戻せる(self, tmp_path):
        path = tmp_path / "job.json"

        write_json(path, {"status": "running", "clusters": []})

        assert read_json(path) == {"status": "running", "clusters": []}

    def test_日本語がエスケープされずに保存される(self, tmp_path):
        path = tmp_path / "job.json"

        write_json(path, {"speaker": "太郎"})

        assert "太郎" in path.read_text(encoding="utf-8")

    def test_JSONにできない値は文字列化される(self, tmp_path):
        path = tmp_path / "job.json"

        write_json(path, {"path": tmp_path / "x.wav"})

        assert read_json(path)["path"] == str(tmp_path / "x.wav")

    def test_親ディレクトリが無ければ作られる(self, tmp_path):
        path = tmp_path / "a" / "b" / "job.json"

        write_json(path, {"ok": True})

        assert read_json(path) == {"ok": True}

    def test_既存ファイルは上書きされる(self, tmp_path):
        path = tmp_path / "job.json"
        write_json(path, {"status": "queued"})

        write_json(path, {"status": "done"})

        assert read_json(path) == {"status": "done"}

    def test_書き出し後に一時ファイルが残らない(self, tmp_path):
        path = tmp_path / "job.json"

        write_json(path, {"ok": True})

        assert [p.name for p in tmp_path.iterdir()] == ["job.json"]

    def test_書き出しに失敗しても元の内容が壊れない(self, tmp_path):
        path = tmp_path / "job.json"
        write_json(path, {"status": "done"})

        class _シリアライズできない値:
            pass

        # default=str も通らない値を作り、途中で失敗させる
        with pytest.raises(TypeError):
            write_json(path, {"bad": {_シリアライズできない値(): 1}})

        assert read_json(path) == {"status": "done"}, "元のファイルが残っていること"

    def test_失敗したときも一時ファイルが残らない(self, tmp_path):
        path = tmp_path / "job.json"
        write_json(path, {"status": "done"})

        class _シリアライズできない値:
            pass

        with pytest.raises(TypeError):
            write_json(path, {"bad": {_シリアライズできない値(): 1}})

        assert [p.name for p in tmp_path.iterdir()] == ["job.json"]


class TestJSONの読み込み:
    def test_存在しないファイルはNoneを返す(self, tmp_path):
        assert read_json(tmp_path / "missing.json") is None

    def test_壊れたJSONは例外になる(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{ではない", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            read_json(path)
