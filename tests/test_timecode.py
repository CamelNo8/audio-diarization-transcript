"""時刻文字列と秒の相互変換の特性テスト。

リファクタリング前の現在の振る舞いを固定する。対象の関数は元々
app.py / audio_processor.py / subtitle_matcher.py に分散しており、
Phase 1 で src/common/timecode.py へ統合した。統合の前後で
ここの期待値は変えていない。
"""

from __future__ import annotations

import pytest

from src.common.timecode import (
    colon_ms_to_comma_ms,
    format_time,
    seconds_to_time_str,
    time_str_to_seconds,
)


class Test秒からHHMMSSms形式への変換:
    """format_time — 文字起こしCSVの時刻列に使われる。"""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00:00:000"),
            (1.234, "00:00:01:234"),
            (60, "00:01:00:000"),
            (3600, "01:00:00:000"),
            (3661.5, "01:01:01:500"),
        ],
    )
    def test_秒がHHMMSSmmm形式の文字列に変換される(self, seconds, expected):
        assert format_time(seconds) == expected, f"入力 {seconds} 秒の変換結果が不正"

    def test_負の秒はゼロ時刻として扱われる(self):
        assert format_time(-1.5) == "00:00:00:000"

    def test_ミリ秒未満は四捨五入される(self):
        assert format_time(1.2346) == "00:00:01:235"


class Testコロン区切りミリ秒からSRT形式への変換:
    """CSV(HH:MM:SS:ms) を SRT(HH:MM:SS,ms) にする。"""

    def test_コロン区切りのミリ秒がカンマ区切りに変換される(self):
        assert colon_ms_to_comma_ms("00:01:02:345") == "00:01:02,345"

    def test_時と分と秒がゼロ埋めされる(self):
        assert colon_ms_to_comma_ms("1:02:03.500") == "01:02:03,500"

    def test_すでにカンマ区切りならそのまま返される(self):
        assert colon_ms_to_comma_ms("00:01:02,345") == "00:01:02,345"

    def test_空文字は空文字のまま返される(self):
        assert colon_ms_to_comma_ms("") == ""

    def test_想定外の書式はそのまま返される(self):
        assert colon_ms_to_comma_ms("不正な時刻") == "不正な時刻"

    def test_コロンが3つ以上ある場合は最後のコロンだけがカンマになる(self):
        assert colon_ms_to_comma_ms("00:00:01:02:345") == "00:00:01:02,345"

    def test_ミリ秒が3桁未満のときは前ゼロ埋めされる(self):
        # 現仕様の記録: "5" は 500ms ではなく 005ms として解釈される。
        # 呼び出し元の CSV は常に3桁で出力するため実害は出ていない。
        assert colon_ms_to_comma_ms("00:00:01:5") == "00:00:01,005"


class Test字幕時刻文字列と秒の相互変換:
    """subtitle_matcher の時刻ユーティリティ。"""

    @pytest.mark.parametrize(
        ("time_str", "expected"),
        [
            ("00:00:00,000", 0.0),
            ("00:01:02,345", 62.345),
            ("01:01:01,500", 3661.5),
            ("1:2:3", 3723.0),
        ],
    )
    def test_時刻文字列が秒に変換される(self, time_str, expected):
        assert time_str_to_seconds(time_str) == pytest.approx(expected)

    @pytest.mark.parametrize("invalid", ["", "不正な時刻", "00:00"])
    def test_不正な時刻文字列は0秒として扱われる(self, invalid):
        assert time_str_to_seconds(invalid) == 0.0

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00:00,000"),
            (62.5, "00:01:02,500"),
            (3661.0, "01:01:01,000"),
        ],
    )
    def test_秒が字幕時刻文字列に変換される(self, seconds, expected):
        assert seconds_to_time_str(seconds) == expected

    def test_負の秒はゼロ時刻として扱われる(self):
        assert seconds_to_time_str(-5) == "00:00:00,000"

    def test_秒と文字列の往復で値が保たれる(self):
        original = "00:01:02,500"

        seconds = time_str_to_seconds(original)
        restored = seconds_to_time_str(seconds)

        assert restored == original
