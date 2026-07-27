"""マッチング結果のサマリ表示（src/subtitle/report.py）。

統計情報は母数が 0 でも例外を投げてはならない。呼び出し側のガードに
依存せず、この関数単体で成立させる。
"""

from __future__ import annotations

import pytest

from src.subtitle.report import display_summary


def _台本(id_: int, dialogue: str) -> dict:
    return {"id": id_, "speaker": "アイ", "dialogue": dialogue}


def _音声認識(id_: int, text: str) -> dict:
    return {
        "id": id_,
        "start": "00:00:01,000",
        "end": "00:00:02,000",
        "text": text,
        "stt_speaker": "アイ",
    }


class Test母数が0のとき:
    def test_台本も音声認識も空なら例外を投げない(self, caplog):
        with caplog.at_level("INFO"):
            display_summary([], [], [], set(), set())

        assert "0.0%" in caplog.text

    def test_台本だけ空でも例外を投げない(self):
        display_summary([], [_音声認識(0, "こんにちは")], [], set(), set())

    def test_音声認識だけ空でも例外を投げない(self):
        display_summary([_台本(0, "こんにちは")], [], [], set(), set())


class Test通常のデータ:
    def test_使用率が百分率で出力される(self, caplog):
        scripts = [_台本(0, "あ"), _台本(1, "い")]
        stt = [_音声認識(0, "あ"), _音声認識(1, "い"), _音声認識(2, "う")]

        with caplog.at_level("INFO"):
            display_summary(scripts, stt, [], {0}, {0, 1})

        assert "1/2 (50.0%)" in caplog.text
        assert "2/3 (66.7%)" in caplog.text

    @pytest.mark.parametrize(
        ("使用済み", "期待"), [(set(), "0/2 (0.0%)"), ({0, 1}, "2/2 (100.0%)")]
    )
    def test_全く使われない場合と全て使われる場合(self, caplog, 使用済み, 期待):
        scripts = [_台本(0, "あ"), _台本(1, "い")]

        with caplog.at_level("INFO"):
            display_summary(scripts, [_音声認識(0, "あ")], [], 使用済み, set())

        assert 期待 in caplog.text
