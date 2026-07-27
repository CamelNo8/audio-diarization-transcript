"""出力ファイル名の検証（src/common/filenames.py）。

Web のフォームから受け取る出力ファイル名は、そのままパス結合すると
``../`` や絶対パスで temp/ の外へ書き込めてしまう（規約7.1）。
安全な名前だけを通し、それ以外は既定名へ落とすことをここで固定する。
"""

from __future__ import annotations

import pytest

from src.common.filenames import safe_output_name

_DEFAULT = "output.srt"


class Test使えるファイル名:
    @pytest.mark.parametrize(
        "name",
        [
            "transcription.srt",
            "対応表.csv",
            "subtitles.srt",
            "my file with spaces.srt",
            "a.b.c.srt",
            "拡張子なし",
        ],
    )
    def test_通常のファイル名はそのまま返される(self, name):
        assert safe_output_name(name, _DEFAULT) == name

    def test_前後の空白は取り除かれる(self):
        assert safe_output_name("  result.csv  ", _DEFAULT) == "result.csv"


class Test既定名へ落とすファイル名:
    @pytest.mark.parametrize("name", ["", "   ", ".", "..", "  ..  "])
    def test_空や特殊なパス名は既定名になる(self, name):
        assert safe_output_name(name, _DEFAULT) == _DEFAULT

    @pytest.mark.parametrize(
        "name",
        [
            "../../evil.srt",
            "../evil.srt",
            "/tmp/abs.srt",
            "sub/dir.srt",
            "dir/",
            "a\\b.srt",
        ],
    )
    def test_パス区切りを含む名前は既定名になる(self, name):
        assert safe_output_name(name, _DEFAULT) == _DEFAULT


class Test検証後の名前でパスを組み立てると外へ出ない:
    @pytest.mark.parametrize(
        "name", ["../../evil.srt", "/tmp/abs.srt", "sub/dir.srt", ""]
    )
    def test_temp配下から出ない(self, name, tmp_path):
        base = tmp_path / "temp"
        base.mkdir()

        resolved = (base / safe_output_name(name, _DEFAULT)).resolve()

        assert resolved.parent == base.resolve()
