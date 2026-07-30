"""正解字幕SRTの解析と説明変数の算出のテスト。

実験用（予稿 4.3 / 4.5）のクリップ選定と評価が共通して使う土台。
話者ラベルの拾い方は納品字幕の実物に合わせてあり、ここで振る舞いを固定する。
"""

from __future__ import annotations

import pytest

from src.evaluation.srt_stats import (
    LabelRules,
    compute_overlap_time_ratio,
    compute_variables,
    parse_rttm,
    parse_srt,
)


def _SRTを書く(path, content: str, encoding: str = "utf-8-sig"):
    path.write_text(content, encoding=encoding)
    return path


class Test字幕の読み込み:
    """parse_srt — BOM や表記ゆれを吸収して字幕を読む。"""

    def test_BOM付きで複数行の字幕が読み込まれる(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n"
            "00:00:02,190 --> 00:00:04,820\n"
            "（三上）そういえば\n"
            "（三上）あの時の\n"
            "\n"
            "2\n"
            "00:01:05,000 --> 00:01:07,500\n"
            "（大賢者）解。記録は残っています\n",
        )

        entries = parse_srt(path)

        assert len(entries) == 2
        assert entries[0].start == pytest.approx(2.190)
        assert entries[0].end == pytest.approx(4.820)
        assert entries[1].start == pytest.approx(65.0)
        assert entries[1].end == pytest.approx(67.5)

    def test_ミリ秒の区切りがピリオドでも読み込める(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:01.500 --> 00:00:03.250\n（三上）やあ\n",
        )

        entries = parse_srt(path)

        assert entries[0].start == pytest.approx(1.5)
        assert entries[0].end == pytest.approx(3.25)

    def test_存在しないファイルはFileNotFoundErrorになる(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_srt(tmp_path / "missing.srt")


class Test話者ラベルの抽出:
    """parse_srt — 話者は （） だけから拾い、効果音・音楽は話者に数えない。"""

    def test_行頭の話者ラベルが抽出され本文から除かれる(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:02,000\n（三上）そういえば あの時の\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("三上",)
        assert entry.body == "そういえば あの時の"

    def test_同じ話者が各行に繰り返されても話者は1人と数える(self, tmp_path):
        """納品字幕は複数行のとき各行に同じ話者名を前置する。重なりと誤認しない。"""
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "（佐藤）ありがたいことに\n"
            "（佐藤）5回目になりました\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("佐藤",)
        assert entry.has_overlap is False

    def test_空白の後ろの話者ラベルも抽出される(self, tmp_path):
        """1行に2話者が並ぶ表記が、重なり・話者交替の手がかりになる。"""
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "（吉村）あ、そうなんだ （松尾）そうですね\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("吉村", "松尾")
        assert entry.has_overlap is True

    def test_単語の直後の括弧は話者ラベルとみなさない(self, tmp_path):
        """``Claude（クロード）`` のような言い換えを話者として拾わない。"""
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:03,000\n（岸谷）あとはClaude（クロード）\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("岸谷",)
        assert entry.body == "あとはClaude（クロード）"

    def test_角括弧の効果音は話者にも本文にも含まれない(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:05,000\n[荘厳な音楽]\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ()
        assert entry.body == ""
        assert entry.is_speech is False

    @pytest.mark.parametrize("label", ["軽快なBGM", "ドアの作動音", "笑", "笑い"])
    def test_括弧内の効果音や音楽は話者から除外される(self, tmp_path, label):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            f"1\n00:00:00,000 --> 00:00:02,000\n（{label}）\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ()
        assert entry.non_speech_labels == (label,)
        assert entry.is_speech is False

    def test_効果音と話者が同じエントリにあれば発話として扱う(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "（軽快なBGM）\n"
            "（磯貝）今、世界のリーダーは\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("磯貝",)
        assert entry.non_speech_labels == ("軽快なBGM",)
        assert entry.is_speech is True

    def test_入れ子の括弧を含む話者名がそのまま保持される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "（マクロン大統領（VTR））いつ遮断されるか\n",
        )

        entry = parse_srt(path)[0]

        assert entry.speakers == ("マクロン大統領（VTR）",)
        assert entry.body == "いつ遮断されるか"

    def test_接尾辞を落とす指定で注記付きの話者名が同一視される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:02,000\n（中嶋／手話・音声通訳）はい\n"
            "\n2\n00:00:02,000 --> 00:00:04,000\n（中嶋・手話）どうぞ\n",
        )

        entries = parse_srt(path, LabelRules(should_strip_suffix=True))

        assert entries[0].speakers == ("中嶋",)
        assert entries[1].speakers == ("中嶋",)

    def test_話者として明示指定したラベルはキーワードより優先される(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:02,000\n（笑い袋）へへへ\n",
        )

        entries = parse_srt(path, LabelRules(speaker_labels=("笑い袋",)))

        assert entries[0].speakers == ("笑い袋",)

    def test_角括弧を話者として読む指定では話者に数える(self, tmp_path):
        """アプリの仮字幕（src/web/converters.py）は ``[話者] 本文`` 形式で出る。"""
        path = _SRTを書く(
            tmp_path / "app.srt",
            "1\n00:00:00,000 --> 00:00:02,000\n[若林] ん?\n",
        )

        entries = parse_srt(path, LabelRules(should_read_square_brackets=True))

        assert entries[0].speakers == ("若林",)
        assert entries[0].body == "ん?"

    def test_非発話として追加指定したラベルは話者に数えない(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:02,000\n（驚く声）ええっ\n",
        )

        entries = parse_srt(path, LabelRules(extra_non_speech_labels=("驚く声",)))

        assert entries[0].speakers == ()
        assert entries[0].non_speech_labels == ("驚く声",)


class Test説明変数:
    """compute_variables — 予稿 4.3 の記録項目を数える。"""

    def _字幕を作る(self, tmp_path):
        return _SRTを書く(
            tmp_path / "gt.srt",
            "1\n00:00:00,000 --> 00:00:10,000\n（三上）あのさ\n"
            "\n2\n00:00:10,000 --> 00:00:20,000\n（三上）そうそう\n"
            "\n3\n00:00:20,000 --> 00:00:30,000\n（大賢者）解 （三上）まじか\n"
            "\n4\n00:00:30,000 --> 00:00:40,000\n[荘厳な音楽]\n"
            "\n5\n00:00:40,000 --> 00:00:50,000\n（大賢者）記録は残っています\n",
        )

    def test_発話数と話者交替回数が数えられる(self, tmp_path):
        entries = parse_srt(self._字幕を作る(tmp_path))

        variables = compute_variables(entries, duration_sec=60.0)

        # 音楽だけの4番は発話に数えない
        assert variables.utterance_count == 4
        assert variables.speaker_count == 2
        # 三上 → 三上 → 大賢者 → 大賢者 なので交替は1回
        assert variables.speaker_change_count == 1
        assert variables.speaker_change_per_min == pytest.approx(1.0)

    def test_複数話者を含むエントリだけが重なりに数えられる(self, tmp_path):
        entries = parse_srt(self._字幕を作る(tmp_path))

        variables = compute_variables(entries, duration_sec=60.0)

        assert variables.overlap_entry_count == 1
        assert variables.overlap_entry_ratio == pytest.approx(0.25)

    def test_発話時間の割合が算出される(self, tmp_path):
        entries = parse_srt(self._字幕を作る(tmp_path))

        variables = compute_variables(entries, duration_sec=60.0)

        # 発話4件×10秒 ÷ 60秒
        assert variables.speech_time_ratio == pytest.approx(40.0 / 60.0)

    def test_話者ごとの発話数が発話の多い順に記録される(self, tmp_path):
        """登場人数を揃えず記録する方針のため、誰が何回喋ったかを残す。"""
        entries = parse_srt(self._字幕を作る(tmp_path))

        variables = compute_variables(entries, duration_sec=60.0)

        # 三上は1・2・3番、大賢者は3・5番に登場する
        assert variables.utterance_counts == (("三上", 3), ("大賢者", 2))

    def test_話者ごとの発話数がCSVの1列に収まる(self, tmp_path):
        entries = parse_srt(self._字幕を作る(tmp_path))

        row = compute_variables(entries, duration_sec=60.0).as_row()

        assert row["speaker_utterances"] == "三上:3 大賢者:2"

    def test_発話が無ければ割合は0になる(self):
        variables = compute_variables([], duration_sec=60.0)

        assert variables.utterance_count == 0
        assert variables.speaker_change_count == 0
        assert variables.overlap_entry_ratio == 0.0
        assert variables.speech_time_ratio == 0.0

    def test_区間長が0でも0除算しない(self):
        variables = compute_variables([], duration_sec=0.0)

        assert variables.speaker_change_per_min == 0.0


class Test話者分離結果からの重なり時間:
    """parse_rttm / compute_overlap_time_ratio — pyannote の出力から重なりを測る。"""

    def _RTTMを書く(self, path, lines: list[str]):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_2人以上が同時に喋っている時間の割合が算出される(self, tmp_path):
        # 三上 0.0-4.0 / 大賢者 3.0-5.0 → 重なり 1秒、発話 5秒
        path = self._RTTMを書く(
            tmp_path / "d.rttm",
            [
                "SPEAKER clip 1 0.000 4.000 <NA> <NA> 三上 <NA> <NA>",
                "SPEAKER clip 1 3.000 2.000 <NA> <NA> 大賢者 <NA> <NA>",
            ],
        )

        segments = parse_rttm(path)

        assert len(segments) == 2
        assert compute_overlap_time_ratio(segments) == pytest.approx(1.0 / 5.0)

    def test_3人が重なっても重複して数えない(self, tmp_path):
        path = self._RTTMを書く(
            tmp_path / "d.rttm",
            [
                "SPEAKER clip 1 0.000 3.000 <NA> <NA> A <NA> <NA>",
                "SPEAKER clip 1 1.000 2.000 <NA> <NA> B <NA> <NA>",
                "SPEAKER clip 1 1.000 2.000 <NA> <NA> C <NA> <NA>",
            ],
        )

        # 1.0-3.0 の2秒が重なり、発話区間は 0.0-3.0 の3秒
        assert compute_overlap_time_ratio(parse_rttm(path)) == pytest.approx(2.0 / 3.0)

    def test_重なりが無ければ0になる(self, tmp_path):
        path = self._RTTMを書く(
            tmp_path / "d.rttm",
            [
                "SPEAKER clip 1 0.000 1.000 <NA> <NA> A <NA> <NA>",
                "SPEAKER clip 1 2.000 1.000 <NA> <NA> B <NA> <NA>",
            ],
        )

        assert compute_overlap_time_ratio(parse_rttm(path)) == 0.0

    def test_区間が無ければ0になる(self):
        assert compute_overlap_time_ratio([]) == 0.0

    def test_列が足りない行や数値でない行は読み飛ばす(self, tmp_path):
        path = self._RTTMを書く(
            tmp_path / "d.rttm",
            [
                "SPEAKER clip 1 0.000 1.000 <NA> <NA> A <NA> <NA>",
                "SPEAKER clip 1 0.000",
                "SPEAKER clip 1 abc 1.000 <NA> <NA> B <NA> <NA>",
                "",
            ],
        )

        segments = parse_rttm(path)

        assert len(segments) == 1
        assert segments[0].speaker == "A"
