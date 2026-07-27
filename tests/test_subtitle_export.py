"""対応表CSVの出力と字幕SRT生成の特性テスト。

リファクタリング前の ``subtitle_matcher.export_results_to_csv`` と
``subtitle_exporter.py`` の振る舞いを固定する。分割後は
``src/subtitle/report.py`` / ``src/subtitle/exporter.py`` に対して同じ期待値を保つ。
"""

from __future__ import annotations

import csv

import pytest

from src.subtitle.exporter import (
    format_subtitle_text,
    generate_srt_content,
    load_subtitle_data,
    write_srt_file,
)
from src.subtitle.report import export_results_to_csv

#: 対応表CSVの列。順序も含めて後段（字幕生成・Web UI）が依存する。
対応表の列 = [
    "type",
    "script_start_id",
    "script_end_id",
    "script_speaker",
    "script_dialogue",
    "stt_start_id",
    "stt_end_id",
    "stt_speaker",
    "stt_text",
    "start_time",
    "end_time",
    "speaker",
    "subtitle_text",
]


def _対応表を読む(path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def 台本():
    return [
        {"id": 0, "speaker": "太郎", "dialogue": "おはよう"},
        {"id": 1, "speaker": "花子", "dialogue": "おはようございます"},
        {"id": 2, "speaker": "太郎", "dialogue": "いい天気だね"},
    ]


@pytest.fixture
def 音声認識():
    return [
        {
            "id": 0,
            "start": "00:00:01,000",
            "end": "00:00:02,000",
            "text": "おはよう",
            "stt_speaker": "SPEAKER_00",
        },
        {
            "id": 1,
            "start": "00:00:05,000",
            "end": "00:00:06,000",
            "text": "いい天気だね",
            "stt_speaker": "SPEAKER_01",
        },
        {
            "id": 2,
            "start": "00:00:09,000",
            "end": "00:00:10,000",
            "text": "ええ",
            "stt_speaker": "SPEAKER_01",
        },
    ]


@pytest.fixture
def マッチ結果():
    return [
        {
            "script_ng": {
                "id": 0,
                "text": "おはよう",
                "original_ids": [0],
                "n": 1,
                "speaker": "太郎",
            },
            "stt_ng": {"id": 0, "text": "おはよう", "original_ids": [0], "n": 1},
            "similarity": 0.95,
        },
        {
            "script_ng": {
                "id": 2,
                "text": "いい天気だね",
                "original_ids": [2],
                "n": 1,
                "speaker": "太郎",
            },
            "stt_ng": {"id": 1, "text": "いい天気だね", "original_ids": [1], "n": 1},
            "similarity": 0.90,
        },
    ]


@pytest.fixture
def 対応表(tmp_path, 台本, 音声認識, マッチ結果):
    出力先 = tmp_path / "対応表.csv"
    export_results_to_csv(
        台本, 音声認識, マッチ結果, {0, 2}, {0, 1}, filename=str(出力先)
    )
    return _対応表を読む(出力先)


class Test対応表CSVの出力:
    """export_results_to_csv — 時系列に並べた対応表を書き出す。"""

    def test_列構成が固定の順序で出力される(self, 対応表):
        assert list(対応表[0].keys()) == 対応表の列

    def test_マッチと未使用の行がすべて出力される(self, 対応表):
        # Matched 2件 + Unmatched_Script 1件 + Unmatched_STT 1件
        assert len(対応表) == 4
        assert [row["type"] for row in 対応表] == [
            "Matched",
            "Unmatched_Script",
            "Matched",
            "Unmatched_STT",
        ]

    def test_未使用の台本はマッチ行の台本idの直前に差し込まれる(self, 対応表):
        未使用台本 = 対応表[1]

        assert 未使用台本["script_start_id"] == "1"
        assert 未使用台本["script_dialogue"] == "おはようございます"

    def test_マッチ行の時刻は元の音声認識セグメントから引かれる(self, 対応表):
        assert 対応表[0]["start_time"] == "00:00:01,000"
        assert 対応表[0]["end_time"] == "00:00:02,000"

    def test_未使用台本の時刻は前後の空きに等分で補完される(self, 対応表):
        # 直前の Matched の終了 2 秒 〜 直後の Matched の開始 5 秒 を 1 件で埋める
        assert 対応表[1]["start_time"] == "00:00:02,000"
        assert 対応表[1]["end_time"] == "00:00:05,000"

    def test_台本に話者があるときは台本の話者が採用される(self, 対応表):
        assert 対応表[0]["speaker"] == "太郎"
        assert 対応表[0]["subtitle_text"] == "おはよう"

    def test_未使用の音声認識には音声認識の話者と本文が入る(self, 対応表):
        未使用字幕 = 対応表[3]

        assert 未使用字幕["speaker"] == "SPEAKER_01"
        assert 未使用字幕["subtitle_text"] == "ええ"
        assert 未使用字幕["script_start_id"] == ""

    def test_台本に話者がないときは音声認識の話者で補われる(self, tmp_path, 音声認識):
        台本 = [{"id": 0, "speaker": "", "dialogue": "おはよう"}]
        マッチ = [
            {
                "script_ng": {
                    "id": 0,
                    "text": "おはよう",
                    "original_ids": [0],
                    "n": 1,
                    "speaker": "",
                },
                "stt_ng": {"id": 0, "text": "おはよう", "original_ids": [0], "n": 1},
                "similarity": 0.9,
            }
        ]
        出力先 = tmp_path / "対応表.csv"

        export_results_to_csv(
            台本, 音声認識[:1], マッチ, {0}, {0}, filename=str(出力先)
        )

        assert _対応表を読む(出力先)[0]["speaker"] == "SPEAKER_00"

    def test_複数idのngramは開始idと終了idが両端になる(self, tmp_path, 音声認識):
        台本 = [
            {"id": 0, "speaker": "太郎", "dialogue": "おはよう"},
            {"id": 1, "speaker": "太郎", "dialogue": "いい天気だね"},
        ]
        マッチ = [
            {
                "script_ng": {
                    "id": 0,
                    "text": "おはよう いい天気だね",
                    "original_ids": [0, 1],
                    "n": 2,
                    "speaker": "太郎",
                },
                "stt_ng": {
                    "id": 0,
                    "text": "おはよう いい天気だね",
                    "original_ids": [0, 1],
                    "n": 2,
                },
                "similarity": 0.9,
            }
        ]
        出力先 = tmp_path / "対応表.csv"

        export_results_to_csv(
            台本, 音声認識[:2], マッチ, {0, 1}, {0, 1}, filename=str(出力先)
        )
        行 = _対応表を読む(出力先)[0]

        assert (行["script_start_id"], 行["script_end_id"]) == ("0", "1")
        assert (行["stt_start_id"], 行["stt_end_id"]) == ("0", "1")
        assert 行["start_time"] == "00:00:01,000"
        assert 行["end_time"] == "00:00:06,000"

    def test_マッチが1件もなくても未使用行だけで出力される(
        self, tmp_path, 台本, 音声認識
    ):
        出力先 = tmp_path / "対応表.csv"

        export_results_to_csv(台本, 音声認識, [], set(), set(), filename=str(出力先))
        行 = _対応表を読む(出力先)

        assert len(行) == 6
        assert {row["type"] for row in 行} == {"Unmatched_STT", "Unmatched_Script"}

    def test_BOM付きUTF8で書き出される(self, tmp_path, 台本, 音声認識, マッチ結果):
        出力先 = tmp_path / "対応表.csv"

        export_results_to_csv(
            台本, 音声認識, マッチ結果, {0, 2}, {0, 1}, filename=str(出力先)
        )

        assert 出力先.read_bytes().startswith(b"\xef\xbb\xbf")


class Test字幕テキストの整形:
    """format_subtitle_text — 話者が変わったときだけ話者名を前置する。"""

    def test_直前と話者が変わったら話者名が前置される(self):
        assert format_subtitle_text("太郎", "おはよう", "花子") == "(太郎)おはよう"

    def test_最初の字幕には話者名が前置される(self):
        assert format_subtitle_text("太郎", "おはよう", None) == "(太郎)おはよう"

    def test_直前と話者が同じなら本文だけになる(self):
        assert format_subtitle_text("太郎", "おはよう", "太郎") == "おはよう"

    def test_話者が空欄なら本文だけになる(self):
        assert format_subtitle_text("", "おはよう", "太郎") == "おはよう"


class TestSRTコンテンツの生成:
    """generate_srt_content — 対応表データから SRT 本文を組み立てる。"""

    def test_ブロックが1から連番で採番される(self):
        データ = [
            {
                "start_time": "00:00:01,000",
                "end_time": "00:00:02,000",
                "speaker": "太郎",
                "subtitle_text": "おはよう",
            },
            {
                "start_time": "00:00:03,000",
                "end_time": "00:00:04,000",
                "speaker": "花子",
                "subtitle_text": "おはようございます",
            },
        ]

        srt = generate_srt_content(データ)

        assert srt == (
            "1\n00:00:01,000 --> 00:00:02,000\n(太郎)おはよう\n"
            "\n"
            "2\n00:00:03,000 --> 00:00:04,000\n(花子)おはようございます\n"
        )

    def test_同じ話者が続くと2件目以降は話者名が省かれる(self):
        データ = [
            {
                "start_time": "00:00:01,000",
                "end_time": "00:00:02,000",
                "speaker": "太郎",
                "subtitle_text": "おはよう",
            },
            {
                "start_time": "00:00:03,000",
                "end_time": "00:00:04,000",
                "speaker": "太郎",
                "subtitle_text": "いい天気だね",
            },
        ]

        srt = generate_srt_content(データ)

        assert "(太郎)おはよう" in srt
        assert "\nいい天気だね\n" in srt

    def test_話者名の前後の空白は除去される(self):
        データ = [
            {
                "start_time": "00:00:01,000",
                "end_time": "00:00:02,000",
                "speaker": " 太郎 ",
                "subtitle_text": "おはよう",
            }
        ]

        assert "(太郎)おはよう" in generate_srt_content(データ)

    def test_話者が空欄の次の行は必ず話者名が付く(self):
        データ = [
            {
                "start_time": "00:00:01,000",
                "end_time": "00:00:02,000",
                "speaker": "太郎",
                "subtitle_text": "おはよう",
            },
            {
                "start_time": "00:00:03,000",
                "end_time": "00:00:04,000",
                "speaker": "",
                "subtitle_text": "……",
            },
            {
                "start_time": "00:00:05,000",
                "end_time": "00:00:06,000",
                "speaker": "太郎",
                "subtitle_text": "いい天気だね",
            },
        ]

        srt = generate_srt_content(データ)

        assert "\n……\n" in srt
        assert "(太郎)いい天気だね" in srt

    def test_空のデータからは空文字が返る(self):
        assert generate_srt_content([]) == ""


class Test対応表からの字幕データ読み込み:
    """load_subtitle_data — 字幕生成に必要な4列だけを取り出す。"""

    def _対応表を書く(self, path, rows, header=None):
        header = header or ["start_time", "end_time", "speaker", "subtitle_text"]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_必要な4列だけが取り出される(self, tmp_path):
        path = self._対応表を書く(
            tmp_path / "in.csv",
            [
                {
                    "type": "Matched",
                    "start_time": "00:00:01,000",
                    "end_time": "00:00:02,000",
                    "speaker": "太郎",
                    "subtitle_text": "おはよう",
                }
            ],
            header=["type", "start_time", "end_time", "speaker", "subtitle_text"],
        )

        データ = load_subtitle_data(str(path))

        assert データ == [
            {
                "start_time": "00:00:01,000",
                "end_time": "00:00:02,000",
                "speaker": "太郎",
                "subtitle_text": "おはよう",
            }
        ]

    def test_話者が空欄の行は残る(self, tmp_path):
        path = self._対応表を書く(
            tmp_path / "in.csv",
            [
                {
                    "start_time": "00:00:01,000",
                    "end_time": "00:00:02,000",
                    "speaker": "",
                    "subtitle_text": "……",
                }
            ],
        )

        assert len(load_subtitle_data(str(path))) == 1

    def test_話者が空欄でも警告は出ない(self, tmp_path, caplog):
        # 話者名の空欄は仕様どおりの正常系。採用するのに「スキップします」と
        # 警告すると、本当の警告が埋もれる。
        path = self._対応表を書く(
            tmp_path / "in.csv",
            [
                {
                    "start_time": "00:00:01,000",
                    "end_time": "00:00:02,000",
                    "speaker": "",
                    "subtitle_text": "……",
                }
            ],
        )

        with caplog.at_level("WARNING"):
            load_subtitle_data(str(path))

        assert caplog.records == []

    @pytest.mark.parametrize("欠損列", ["start_time", "end_time", "subtitle_text"])
    def test_時刻か本文が空の行は捨てられる(self, tmp_path, 欠損列):
        行 = {
            "start_time": "00:00:01,000",
            "end_time": "00:00:02,000",
            "speaker": "太郎",
            "subtitle_text": "おはよう",
        }
        行[欠損列] = ""
        path = self._対応表を書く(tmp_path / "in.csv", [行])

        assert load_subtitle_data(str(path)) == []

    @pytest.mark.parametrize("欠損列", ["start_time", "end_time", "subtitle_text"])
    def test_時刻か本文が空の行はスキップを警告する(self, tmp_path, caplog, 欠損列):
        行 = {
            "start_time": "00:00:01,000",
            "end_time": "00:00:02,000",
            "speaker": "太郎",
            "subtitle_text": "おはよう",
        }
        行[欠損列] = ""
        path = self._対応表を書く(tmp_path / "in.csv", [行])

        with caplog.at_level("WARNING"):
            load_subtitle_data(str(path))

        assert "スキップ" in caplog.text

    def test_必要な列がないCSVからは何も読めない(self, tmp_path):
        path = self._対応表を書く(
            tmp_path / "in.csv",
            [{"start_time": "00:00:01,000", "end_time": "00:00:02,000"}],
            header=["start_time", "end_time"],
        )

        assert load_subtitle_data(str(path)) == []

    def test_ファイルがない場合は空リストが返る(self, tmp_path):
        assert load_subtitle_data(str(tmp_path / "存在しない.csv")) == []


class TestSRTファイルの書き出し:
    """write_srt_file — BOM なし UTF-8 で書き出す。"""

    def test_内容がそのまま書き出される(self, tmp_path):
        出力先 = tmp_path / "out.srt"

        write_srt_file(str(出力先), "1\n00:00:01,000 --> 00:00:02,000\n本文\n")

        assert 出力先.read_text(encoding="utf-8") == (
            "1\n00:00:01,000 --> 00:00:02,000\n本文\n"
        )
        assert not 出力先.read_bytes().startswith(b"\xef\xbb\xbf")
