"""Web UI のファイル変換処理の特性テスト。

対象は現在 app.py のモジュールプライベート関数だが、Phase 4 で
src/web/converters.py の公開関数になる予定のため、ここでは
移設後の公開インターフェースとして検証している。
"""

from __future__ import annotations

import csv
import io

from app import _csv_to_srt_with_speaker, _txt_to_script_csv_bytes

TRANSCRIPT_HEADER = "start,end,speaker,text\n"


def _write_transcript_csv(path, body: str) -> None:
    path.write_text(TRANSCRIPT_HEADER + body, encoding="utf-8-sig")


def _read_script_rows(csv_bytes: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))))


class Test文字起こしCSVからSRTへの変換:
    def test_話者名が本文の先頭に付与される(self, tmp_path):
        csv_path = tmp_path / "transcript.csv"
        srt_path = tmp_path / "out.srt"
        _write_transcript_csv(
            csv_path,
            "00:00:01:000,00:00:02:500,太郎,こんにちは\n"
            "00:00:03:000,00:00:04:000,花子,はじめまして\n",
        )

        count = _csv_to_srt_with_speaker(csv_path, srt_path)

        assert count == 2
        assert srt_path.read_text(encoding="utf-8") == (
            "1\n00:00:01,000 --> 00:00:02,500\n[太郎] こんにちは\n"
            "\n"
            "2\n00:00:03,000 --> 00:00:04,000\n[花子] はじめまして\n"
        )

    def test_話者が空欄の行は本文だけが書き出される(self, tmp_path):
        csv_path = tmp_path / "transcript.csv"
        srt_path = tmp_path / "out.srt"
        _write_transcript_csv(csv_path, "00:00:01:000,00:00:02:000,,本文のみ\n")

        _csv_to_srt_with_speaker(csv_path, srt_path)

        assert "[" not in srt_path.read_text(encoding="utf-8")
        assert "本文のみ" in srt_path.read_text(encoding="utf-8")

    def test_開始終了本文のいずれかが欠けた行は飛ばされる(self, tmp_path):
        csv_path = tmp_path / "transcript.csv"
        srt_path = tmp_path / "out.srt"
        _write_transcript_csv(
            csv_path,
            ",00:00:02:000,太郎,開始なし\n"
            "00:00:03:000,,太郎,終了なし\n"
            "00:00:05:000,00:00:06:000,太郎,\n"
            "00:00:07:000,00:00:08:000,太郎,有効な行\n",
        )

        count = _csv_to_srt_with_speaker(csv_path, srt_path)

        assert count == 1, "有効な1行だけが書き出されるはず"
        assert srt_path.read_text(encoding="utf-8").startswith("1\n")

    def test_有効な行が無いとき空のSRTが作られる(self, tmp_path):
        csv_path = tmp_path / "transcript.csv"
        srt_path = tmp_path / "out.srt"
        _write_transcript_csv(csv_path, "")

        count = _csv_to_srt_with_speaker(csv_path, srt_path)

        assert count == 0
        assert srt_path.read_text(encoding="utf-8") == ""

    def test_入力CSVが存在しないとき0を返しSRTを作らない(self, tmp_path):
        srt_path = tmp_path / "out.srt"

        count = _csv_to_srt_with_speaker(tmp_path / "missing.csv", srt_path)

        assert count == 0
        assert not srt_path.exists()


class Testテキストから台本CSVへの変換:
    def test_ヘッダ行が固定の5列で出力される(self):
        rows = _read_script_rows(_txt_to_script_csv_bytes("台詞".encode("utf-8")))

        assert rows[0] == ["id", "scene_id", "type", "speaker", "contents"]

    def test_話者付きの行が対話行として分解される(self):
        rows = _read_script_rows(
            _txt_to_script_csv_bytes("太郎: こんにちは".encode("utf-8"))
        )[1:]

        assert rows == [["1", "", "dialogue", "太郎", "こんにちは"]]

    def test_全角コロンでも話者が分解される(self):
        rows = _read_script_rows(
            _txt_to_script_csv_bytes("花子：おはよう".encode("utf-8"))
        )[1:]

        assert rows == [["1", "", "dialogue", "花子", "おはよう"]]

    def test_シャープ始まりの行はシーンになる(self):
        rows = _read_script_rows(_txt_to_script_csv_bytes("# 教室".encode("utf-8")))[1:]

        assert rows == [["1", "", "scene", "", "教室"]]

    def test_括弧で囲まれた行はシーンになる(self):
        rows = _read_script_rows(
            _txt_to_script_csv_bytes("（夕方の校庭）".encode("utf-8"))
        )[1:]

        assert rows == [["1", "", "scene", "", "夕方の校庭"]]

    def test_話者のない行は話者空欄の対話行になる(self):
        rows = _read_script_rows(
            _txt_to_script_csv_bytes("ただの地の文".encode("utf-8"))
        )[1:]

        assert rows == [["1", "", "dialogue", "", "ただの地の文"]]

    def test_空行は飛ばしてidが連番になる(self):
        text = "太郎: 一行目\n\n   \n花子: 三行目"

        rows = _read_script_rows(_txt_to_script_csv_bytes(text.encode("utf-8")))[1:]

        assert [row[0] for row in rows] == ["1", "2"]

    def test_話者名が32文字以上の行は話者として分解されない(self):
        # 話者名として認識される上限は31文字（正規表現 [^...]{0,30} + 先頭1文字）
        long_name = "あ" * 32
        text = f"{long_name}: 本文"

        rows = _read_script_rows(_txt_to_script_csv_bytes(text.encode("utf-8")))[1:]

        assert rows == [["1", "", "dialogue", "", text]]

    def test_話者名が31文字ちょうどなら話者として分解される(self):
        boundary_name = "あ" * 31
        text = f"{boundary_name}: 本文"

        rows = _read_script_rows(_txt_to_script_csv_bytes(text.encode("utf-8")))[1:]

        assert rows == [["1", "", "dialogue", boundary_name, "本文"]]

    def test_入力のBOMが除去され出力にBOMが付く(self):
        result = _txt_to_script_csv_bytes("﻿太郎: こんにちは".encode("utf-8"))

        assert result.startswith(b"\xef\xbb\xbf"), "Excel 互換のため BOM 付きで出力する"
        rows = _read_script_rows(result)[1:]
        assert rows[0][3] == "太郎", "入力側の BOM が話者名に混入しないこと"

    def test_空のテキストはヘッダだけのCSVになる(self):
        result = _txt_to_script_csv_bytes(b"")

        assert _read_script_rows(result) == [
            ["id", "scene_id", "type", "speaker", "contents"]
        ]
