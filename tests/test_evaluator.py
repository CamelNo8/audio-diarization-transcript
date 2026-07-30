"""アプリ生成字幕と正解字幕の突き合わせ評価のテスト（予稿 4.5）。

in点は許容ずれ以内かで、話者は発話区間単位で一致するかで評価する。
話者の誤りは false negative / false positive に分類する（予稿 3.2）。
"""

from __future__ import annotations

import pytest

from src.evaluation.evaluator import (
    EvaluationOptions,
    align_entries,
    evaluate,
    fill_omitted_speakers,
    main,
)
from src.evaluation.srt_stats import parse_srt


def _SRTを書く(path, rows: list[tuple[float, float, str]]):
    blocks = []
    for number, (start, end, text) in enumerate(rows, start=1):
        blocks.append(f"{number}\n{_時刻(start)} --> {_時刻(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _時刻(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3600 * 1000)
    minutes, rest = divmod(rest, 60 * 1000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class Test省略された話者名の補完:
    """fill_omitted_speakers — アプリ生成SRTは話者が変わったときだけ前置する。"""

    def test_話者名が無い字幕は直前の話者を引き継ぐ(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "app.srt",
            [
                (0.0, 2.0, "（三上）そういえば、あの時の"),
                (2.0, 4.0, "どうなったんだろうな"),
                (4.0, 6.0, "（大賢者）解。記録は残っています"),
                (6.0, 8.0, "以下に示します"),
            ],
        )

        speakers = fill_omitted_speakers(parse_srt(path))

        assert speakers == ["三上", "三上", "大賢者", "大賢者"]

    def test_先頭の字幕に話者名が無ければ空欄のまま(self, tmp_path):
        path = _SRTを書く(
            tmp_path / "app.srt",
            [(0.0, 2.0, "話者が付かなかった行"), (2.0, 4.0, "（三上）やあ")],
        )

        speakers = fill_omitted_speakers(parse_srt(path))

        assert speakers == ["", "三上"]


class Test発話単位の対応付け:
    """align_entries — 同じ人手文字起こし文が元なので本文で対応付ける。"""

    def test_本文が一致する行同士が対応付けられる(self, tmp_path):
        rows = [(0.0, 2.0, "あのさ"), (2.0, 4.0, "そうだね"), (4.0, 6.0, "まじか")]
        ground_truth = parse_srt(_SRTを書く(tmp_path / "gt.srt", rows))
        app = parse_srt(_SRTを書く(tmp_path / "app.srt", rows))

        pairs = align_entries(ground_truth, app, min_similarity=0.6)

        assert pairs == [(0, 0), (1, 1), (2, 2)]

    def test_話者名の有無は対応付けに影響しない(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(
                tmp_path / "gt.srt",
                [(0.0, 2.0, "（三上）あのさ"), (2.0, 4.0, "（三上）そうだね")],
            )
        )
        app = parse_srt(
            _SRTを書く(
                tmp_path / "app.srt",
                [(0.0, 2.0, "（三上）あのさ"), (2.0, 4.0, "そうだね")],
            )
        )

        pairs = align_entries(ground_truth, app, min_similarity=0.6)

        assert pairs == [(0, 0), (1, 1)]

    def test_アプリ側に余分な行があっても後続の対応付けが崩れない(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(
                tmp_path / "gt.srt",
                [(0.0, 2.0, "あのさ"), (2.0, 4.0, "そうだね"), (4.0, 6.0, "まじか")],
            )
        )
        app = parse_srt(
            _SRTを書く(
                tmp_path / "app.srt",
                [
                    (0.0, 2.0, "あのさ"),
                    (2.0, 3.0, "ご視聴ありがとうございました"),
                    (3.0, 4.0, "そうだね"),
                    (4.0, 6.0, "まじか"),
                ],
            )
        )

        pairs = align_entries(ground_truth, app, min_similarity=0.6)

        assert pairs == [(0, 0), (None, 1), (1, 2), (2, 3)]

    def test_少し文字が違っても類似していれば対応付ける(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, "そういえば、あの時の話")])
        )
        app = parse_srt(
            _SRTを書く(tmp_path / "app.srt", [(0.0, 2.0, "そういえばあの時の話")])
        )

        pairs = align_entries(ground_truth, app, min_similarity=0.6)

        assert pairs == [(0, 0)]


class Testin点の評価:
    """evaluate — 許容ずれ以内かで判定する。閾値は複数まとめて渡せる。"""

    def _1組を作る(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(
                tmp_path / "gt.srt",
                [
                    (0.0, 2.0, "（三上）あのさ"),
                    (10.0, 12.0, "（三上）そうだね"),
                    (20.0, 22.0, "（三上）まじか"),
                ],
            )
        )
        app = parse_srt(
            _SRTを書く(
                tmp_path / "app.srt",
                [
                    (0.2, 2.0, "（三上）あのさ"),
                    (10.5, 12.0, "そうだね"),
                    (25.0, 27.0, "まじか"),
                ],
            )
        )
        return ground_truth, app

    def test_閾値ごとにin点の正解率が出る(self, tmp_path):
        ground_truth, app = self._1組を作る(tmp_path)

        result = evaluate(
            ground_truth, app, EvaluationOptions(tolerances=(0.3, 1.0))
        )

        assert result.in_point_correct_counts[0.3] == 1
        assert result.in_point_correct_counts[1.0] == 2
        assert result.in_point_accuracies[1.0] == pytest.approx(2 / 3)

    def test_in点のずれの統計が出る(self, tmp_path):
        ground_truth, app = self._1組を作る(tmp_path)

        result = evaluate(ground_truth, app, EvaluationOptions())

        assert result.in_point_mean_error == pytest.approx((0.2 + 0.5 + 5.0) / 3)
        assert result.in_point_median_error == pytest.approx(0.5)
        assert result.in_point_max_error == pytest.approx(5.0)


class Test話者の評価:
    """evaluate — 話者の一致と、誤りの2分類（予稿 3.2）。"""

    def _評価する(self, tmp_path, gt_speaker, app_speaker, registered):
        ground_truth = parse_srt(
            _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, f"（{gt_speaker}）あのさ")])
        )
        app = parse_srt(
            _SRTを書く(tmp_path / "app.srt", [(0.0, 2.0, f"（{app_speaker}）あのさ")])
        )
        return evaluate(
            ground_truth, app, EvaluationOptions(registered_speakers=registered)
        )

    def test_話者が一致すれば正解に数える(self, tmp_path):
        result = self._評価する(tmp_path, "三上", "三上", ("三上", "大賢者"))

        assert result.speaker_correct_count == 1
        assert result.speaker_accuracy == pytest.approx(1.0)
        assert result.false_negative_count == 0
        assert result.false_positive_count == 0

    def test_登録話者がUnknownになった場合はfalse_negativeに数える(self, tmp_path):
        result = self._評価する(tmp_path, "三上", "Unknown_01", ("三上",))

        assert result.false_negative_count == 1
        assert result.false_positive_count == 0
        assert result.speaker_correct_count == 0

    def test_話者が空欄でもfalse_negativeに数える(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, "（三上）あのさ")])
        )
        app = parse_srt(_SRTを書く(tmp_path / "app.srt", [(0.0, 2.0, "あのさ")]))

        result = evaluate(
            ground_truth, app, EvaluationOptions(registered_speakers=("三上",))
        )

        assert result.false_negative_count == 1

    def test_別の話者名を付けた場合はfalse_positiveに数える(self, tmp_path):
        result = self._評価する(tmp_path, "三上", "大賢者", ("三上", "大賢者"))

        assert result.false_positive_count == 1
        assert result.false_negative_count == 0

    def test_未登録話者にUnknownを付けた場合は誤りに数えない(self, tmp_path):
        result = self._評価する(tmp_path, "群衆", "Unknown_02", ("三上",))

        assert result.correct_unknown_count == 1
        assert result.false_negative_count == 0
        assert result.false_positive_count == 0

    def test_未登録話者に登録話者名を付けた場合はfalse_positiveに数える(self, tmp_path):
        result = self._評価する(tmp_path, "群衆", "三上", ("三上",))

        assert result.false_positive_count == 1

    def test_混同行列に正解話者とアプリ話者の組が記録される(self, tmp_path):
        result = self._評価する(tmp_path, "三上", "大賢者", ("三上", "大賢者"))

        assert result.confusion[("三上", "大賢者")] == 1


class Test対応付かなかった行:
    """evaluate — 対応が取れなかった行は評価対象から外し、件数を報告する。"""

    def test_対応付かなかった行は件数として報告される(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(
                tmp_path / "gt.srt",
                [
                    (0.0, 2.0, "（三上）あのさ"),
                    (2.0, 4.0, "（三上）聞き取れなかった行"),
                ],
            )
        )
        app = parse_srt(
            _SRTを書く(
                tmp_path / "app.srt",
                [(0.0, 2.0, "（三上）あのさ"), (2.0, 4.0, "全然違う幻聴の行")],
            )
        )

        result = evaluate(ground_truth, app, EvaluationOptions())

        assert result.matched_count == 1
        assert result.unmatched_ground_truth_count == 1
        assert result.unmatched_app_count == 1

    def test_1件も対応付かなくても0除算しない(self, tmp_path):
        ground_truth = parse_srt(
            _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, "（三上）あのさ")])
        )
        app = parse_srt(_SRTを書く(tmp_path / "app.srt", [(0.0, 2.0, "全然違う行")]))

        result = evaluate(ground_truth, app, EvaluationOptions())

        assert result.matched_count == 0
        assert result.speaker_accuracy == 0.0
        assert result.in_point_mean_error == 0.0


class TestCLIとアプリ字幕の話者表記:
    """アプリ生成字幕には2つの話者表記があり、どちらでも評価できる必要がある。"""

    def _評価を実行する(self, tmp_path, app_text: str, capsys):
        ground_truth = _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, "（若林）ん?")])
        app = _SRTを書く(tmp_path / "app.srt", [(0.0, 2.0, app_text)])

        code = main(["--gt", str(ground_truth), "--app", str(app)])

        assert code == 0
        return capsys.readouterr().out

    def test_丸括弧形式のアプリ字幕で話者が一致する(self, tmp_path, capsys):
        """最終字幕（src/subtitle/exporter.py）の形式。"""
        出力 = self._評価を実行する(tmp_path, "(若林)ん?", capsys)

        assert "話者一致: 1 件（100.0%）" in 出力

    def test_角括弧形式のアプリ字幕でも話者が一致する(self, tmp_path, capsys):
        """仮字幕（src/web/converters.py）の形式。ここを外すと全件が誤りになる。"""
        出力 = self._評価を実行する(tmp_path, "[若林] ん?", capsys)

        assert "話者一致: 1 件（100.0%）" in 出力

    def test_字幕ファイルが無ければ終了コード1を返す(self, tmp_path):
        ground_truth = _SRTを書く(tmp_path / "gt.srt", [(0.0, 2.0, "（若林）ん?")])

        code = main(["--gt", str(ground_truth), "--app", str(tmp_path / "missing.srt")])

        assert code == 1
