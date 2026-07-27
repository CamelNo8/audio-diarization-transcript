"""台本CSV・音声認識SRTの読み込みの特性テスト。

リファクタリング前の ``subtitle_matcher.py`` の振る舞いを固定する。
分割後の ``src/subtitle/loader.py`` に対して同じ期待値を保つ。

計画（テスト #6〜#8）には明示していなかったが、読み込み処理も
``src/subtitle/loader.py`` へ移すため、移設前に振る舞いを固定しておく。
"""

from __future__ import annotations

import csv

import pytest

from src.subtitle.loader import load_scripts_from_csv, load_stt_from_srt


def _台本CSVを書く(path, rows: list[dict[str, str]]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "speaker", "contents"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _SRTを書く(path, content: str):
    path.write_text(content, encoding="utf-8")
    return path


class Test台本CSVの読み込み:
    """load_scripts_from_csv — type が dialogue の行だけを台詞として拾う。"""

    def test_台詞行だけが読み込まれidが0から振り直される(self, tmp_path):
        path = _台本CSVを書く(
            tmp_path / "script.csv",
            [
                {"type": "scene", "speaker": "", "contents": "# 朝"},
                {"type": "dialogue", "speaker": "太郎", "contents": "おはよう"},
                {"type": "scene", "speaker": "", "contents": "（ト書き）"},
                {"type": "dialogue", "speaker": "花子", "contents": "おはよう"},
            ],
        )

        台本 = load_scripts_from_csv(str(path))

        assert [s["id"] for s in 台本] == [0, 1]
        assert [s["speaker"] for s in 台本] == ["太郎", "花子"]

    def test_心の声という表記は本文から取り除かれる(self, tmp_path):
        path = _台本CSVを書く(
            tmp_path / "script.csv",
            [{"type": "dialogue", "speaker": "太郎", "contents": "心の声また明日か"}],
        )

        assert load_scripts_from_csv(str(path))[0]["dialogue"] == "また明日か"

    def test_台詞行がなければ空リストになる(self, tmp_path):
        path = _台本CSVを書く(
            tmp_path / "script.csv",
            [{"type": "scene", "speaker": "", "contents": "# 朝"}],
        )

        assert load_scripts_from_csv(str(path)) == []

    def test_ファイルがない場合は空リストが返る(self, tmp_path):
        assert load_scripts_from_csv(str(tmp_path / "存在しない.csv")) == []


class Test音声認識SRTの読み込み:
    """load_stt_from_srt — 不正なセグメントと幻覚テキストを捨てる。"""

    def test_ブロックが順にidを振られて読み込まれる(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "stt.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nおはよう\n"
            "\n"
            "2\n00:00:03,000 --> 00:00:04,000\nいい天気だね\n",
        )

        字幕 = load_stt_from_srt(str(path))

        assert [s["id"] for s in 字幕] == [0, 1]
        assert [s["text"] for s in 字幕] == ["おはよう", "いい天気だね"]
        assert 字幕[0]["start"] == "00:00:01,000"
        assert 字幕[0]["end"] == "00:00:02,000"

    def test_先頭の角括弧は話者として分離される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "stt.srt",
            "1\n00:00:01,000 --> 00:00:02,000\n[太郎] おはよう\n",
        )

        字幕 = load_stt_from_srt(str(path))

        assert 字幕[0]["stt_speaker"] == "太郎"
        assert 字幕[0]["text"] == "おはよう"

    def test_話者表記がなければ話者は空文字になる(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "stt.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nおはよう\n",
        )

        assert load_stt_from_srt(str(path))[0]["stt_speaker"] == ""

    def test_3行目以降は半角空白で連結される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "stt.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nおはよう\nいい天気だね\n",
        )

        assert load_stt_from_srt(str(path))[0]["text"] == "おはよう いい天気だね"

    @pytest.mark.parametrize(
        ("start", "end"),
        [("00:00:02,000", "00:00:01,000"), ("00:00:01,000", "00:00:01,000")],
    )
    def test_終了が開始以下のブロックは捨てられる(self, tmp_path, start, end):
        path = _SRTを書く(tmp_path / "stt.srt", f"1\n{start} --> {end}\nおはよう\n")

        assert load_stt_from_srt(str(path)) == []

    @pytest.mark.parametrize("本文", ["!", "！？", "あ", "caus caus caus"])
    def test_意味のないテキストのブロックは捨てられる(self, tmp_path, 本文):
        path = _SRTを書く(
            tmp_path / "stt.srt", f"1\n00:00:01,000 --> 00:00:02,000\n{本文}\n"
        )

        assert load_stt_from_srt(str(path)) == []

    @pytest.mark.parametrize("本文", ["ああ", "ok", "caus caus me"])
    def test_意味のあるテキストは残る(self, tmp_path, 本文):
        path = _SRTを書く(
            tmp_path / "stt.srt", f"1\n00:00:01,000 --> 00:00:02,000\n{本文}\n"
        )

        assert len(load_stt_from_srt(str(path))) == 1

    def test_時刻行がないブロックは無視される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "stt.srt",
            "1\nおはよう\n\n2\n00:00:03,000 --> 00:00:04,000\nいい天気だね\n",
        )

        字幕 = load_stt_from_srt(str(path))

        assert len(字幕) == 1
        assert 字幕[0]["id"] == 0

    def test_ファイルがない場合は空リストが返る(self, tmp_path):
        assert load_stt_from_srt(str(tmp_path / "存在しない.srt")) == []
