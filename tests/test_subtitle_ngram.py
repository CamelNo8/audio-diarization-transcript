"""テキスト正規化と n-gram 生成の特性テスト。

リファクタリング前の ``subtitle_matcher.py`` の振る舞いを固定する。
分割後の ``src/subtitle/ngram.py`` に対して同じ期待値を保つ。
"""

from __future__ import annotations

import pytest

from src.subtitle.ngram import create_ngrams, normalize_text


def _台本(items: list[tuple[int, str, str]]) -> list[dict]:
    """(id, 話者, 台詞) のタプル列を台本データの形に整える。"""
    return [
        {"id": item_id, "speaker": speaker, "dialogue": dialogue}
        for item_id, speaker, dialogue in items
    ]


def _字幕(items: list[tuple[int, str, str, str]]) -> list[dict]:
    """(id, 開始, 終了, 本文) のタプル列を音声認識データの形に整える。"""
    return [
        {"id": item_id, "start": start, "end": end, "text": text}
        for item_id, start, end, text in items
    ]


class Testテキスト正規化:
    """normalize_text — 比較用にゆらぎを取り除く。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("こんにちは。", "こんにちは"),
            ("あ、い。う", "あ い う"),
            ("Hello, world.", "Hello world"),
            ("全角．ピリオド", "全角 ピリオド"),
        ],
    )
    def test_句読点が空白に置き換わり前後の空白が落ちる(self, text, expected):
        assert normalize_text(text) == expected

    def test_全角括弧が空白に置き換わる(self):
        assert normalize_text("（心の声）こんにちは") == "心の声 こんにちは"

    def test_角括弧で囲まれた部分は丸ごと除去される(self):
        assert normalize_text("[SPEAKER_01] おはよう") == "おはよう"

    def test_角括弧は最短一致で除去される(self):
        assert normalize_text("[a]残す[b]") == "残す"

    def test_連続した空白は1つにまとめられる(self):
        assert normalize_text("あ　 い\t\nう") == "あ い う"

    @pytest.mark.parametrize("text", ["", "   ", "、。"])
    def test_中身が空になる入力は空文字になる(self, text):
        assert normalize_text(text) == ""

    def test_角括弧が閉じていない場合は除去されない(self):
        assert normalize_text("[未閉鎖") == "[未閉鎖"


class TestNgram生成:
    """create_ngrams — 連続する要素をつないだ検索単位を作る。"""

    def test_1gramのみのとき要素数と同じ数が生成される(self):
        data = _台本([(0, "太郎", "おはよう"), (1, "太郎", "元気？")])

        ngrams = create_ngrams(data, text_key="dialogue", max_n=1)

        assert len(ngrams) == 2
        assert [ng["n"] for ng in ngrams] == [1, 1]

    def test_連結テキストは半角空白でつながれる(self):
        data = _台本([(0, "太郎", "おはよう"), (1, "太郎", "元気？")])

        ngrams = create_ngrams(data, text_key="dialogue", max_n=2)

        bigram = ngrams[-1]
        assert bigram["text"] == "おはよう 元気？"
        assert bigram["normalized_text"] == "おはよう 元気？"

    def test_話者をまたぐngramは生成されない(self):
        data = _台本(
            [(0, "太郎", "おはよう"), (1, "太郎", "元気？"), (2, "花子", "うん")]
        )

        ngrams = create_ngrams(data, text_key="dialogue", max_n=2)

        # 1-gram 3件 + 太郎の 2-gram 1件（太郎→花子はスキップ）
        assert len(ngrams) == 4
        bigrams = [ng for ng in ngrams if ng["n"] == 2]
        assert len(bigrams) == 1
        assert bigrams[0]["original_ids"] == [0, 1]

    def test_idはスキップ分を詰めて連番になる(self):
        data = _台本(
            [(0, "太郎", "おはよう"), (1, "太郎", "元気？"), (2, "花子", "うん")]
        )

        ngrams = create_ngrams(data, text_key="dialogue", max_n=2)

        assert [ng["id"] for ng in ngrams] == [0, 1, 2, 3]

    def test_話者付きデータにはspeakerが先頭要素から引き継がれる(self):
        data = _台本([(0, "太郎", "おはよう"), (1, "太郎", "元気？")])

        ngrams = create_ngrams(data, text_key="dialogue", max_n=2)

        assert all(ng["speaker"] == "太郎" for ng in ngrams)

    def test_話者キーがないデータにはspeakerが付かない(self):
        data = _字幕([(0, "00:00:01,000", "00:00:02,000", "おはよう")])

        ngrams = create_ngrams(data, text_key="text", max_n=1)

        assert "speaker" not in ngrams[0]

    def test_開始終了インデックスと元idが記録される(self):
        data = _台本([(10, "太郎", "あ"), (11, "太郎", "い"), (12, "太郎", "う")])

        ngrams = create_ngrams(data, text_key="dialogue", max_n=3)

        trigram = ngrams[-1]
        assert trigram["start_index"] == 0
        assert trigram["end_index"] == 2
        assert trigram["original_ids"] == [10, 11, 12]

    def test_時刻付きのとき先頭の開始と末尾の終了が秒で入る(self):
        data = _字幕(
            [
                (0, "00:00:01,500", "00:00:02,000", "あ"),
                (1, "00:00:03,000", "00:00:04,250", "い"),
            ]
        )

        ngrams = create_ngrams(data, text_key="text", max_n=2, has_time=True)

        bigram = ngrams[-1]
        assert bigram["start_time"] == pytest.approx(1.5)
        assert bigram["end_time"] == pytest.approx(4.25)

    def test_時刻なしのとき時刻キーは付かない(self):
        data = _字幕([(0, "00:00:01,000", "00:00:02,000", "あ")])

        ngrams = create_ngrams(data, text_key="text", max_n=1, has_time=False)

        assert "start_time" not in ngrams[0]
        assert "end_time" not in ngrams[0]

    def test_空リストからは何も生成されない(self):
        assert create_ngrams([], text_key="dialogue", max_n=3) == []

    def test_max_nが0のとき何も生成されない(self):
        data = _台本([(0, "太郎", "おはよう")])

        assert create_ngrams(data, text_key="dialogue", max_n=0) == []

    def test_max_nが要素数を超えても超過分は生成されない(self):
        data = _台本([(0, "太郎", "あ"), (1, "太郎", "い")])

        ngrams = create_ngrams(data, text_key="dialogue", max_n=5)

        # 1-gram 2件 + 2-gram 1件のみ
        assert len(ngrams) == 3
        assert max(ng["n"] for ng in ngrams) == 2
