"""アプリ生成字幕を正解字幕と突き合わせて評価する（予稿 4.5）。

- **in点**: 正解との差が許容ずれ以内かで判定する。閾値は複数まとめて渡せる。
- **話者**: 発話区間単位で正解と一致するかで判定し、誤りを2種類に分類する（予稿 3.2）。
  - false negative … 登録話者なのに ``Unknown_NN``（または空欄）になった
  - false positive … 正解とは別の話者名を付けた

対応付けは**本文テキスト**で行う。同じ人手文字起こし文が元なので本文が一致するため。
ただしアプリ生成SRTは**話者が変わったときだけ話者名を前置する**
（:mod:`src.subtitle.exporter` の仕様）ので、剥がしたあと直前の話者で補完する。

実行例::

    uv run python -m src.evaluation.evaluator \\
        --gt docs/experiment/gt/転スラ_hard.srt --app temp/subtitles.srt \\
        --voice-db voice_databases/転スラ --rttm temp/transcription.rttm \\
        --tolerance 0.3 0.5 1.0 \\
        --out-csv docs/experiment/results.csv --append
"""

from __future__ import annotations

import argparse
import difflib
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.common.csv_io import write_dict_rows
from src.common.logging import configure_logging, get_logger
from src.evaluation.srt_stats import (
    LabelRules,
    SubtitleEntry,
    compute_overlap_time_ratio,
    compute_variables,
    match_key,
    parse_rttm,
    parse_srt,
)

logger = get_logger(__name__)

#: 未登録話者に付くラベル。空欄も「話者を付けられなかった」として同じ扱いにする。
_UNKNOWN_SPEAKER_RE = re.compile(r"^unknown(_\d+)?$", re.IGNORECASE)

#: 既定の許容ずれ（秒）。予稿 4.5 の「許容できるずれ」は事前の試行で決める。
DEFAULT_TOLERANCE_SEC = 0.5

#: 本文が一致しなくても対応付けてよいとみなす類似度の下限。
DEFAULT_MIN_SIMILARITY = 0.6

#: 声紋DBのファイル名から話者名を取り出す区切り（``田村_叫び声`` → ``田村``）。
_VOICE_FILE_SEPARATOR = "_"

#: 話者の判定結果。
JUDGEMENT_CORRECT = "correct"
JUDGEMENT_FALSE_NEGATIVE = "false_negative"
JUDGEMENT_FALSE_POSITIVE = "false_positive"
JUDGEMENT_CORRECT_UNKNOWN = "correct_unknown"


@dataclass(frozen=True)
class EvaluationOptions:
    """評価の設定。

    Attributes:
        tolerances: in点の許容ずれ（秒）。複数指定すると閾値ごとに集計する。
        min_similarity: 本文の類似度がこれ以上なら対応付ける。
        registered_speakers: 声紋データベースに登録済みの話者名。
            空の場合は「正解に出てくる話者はすべて登録済み」と仮定する。
    """

    tolerances: tuple[float, ...] = (DEFAULT_TOLERANCE_SEC,)
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    registered_speakers: tuple[str, ...] = ()

    def is_registered(self, speaker: str) -> bool:
        """正解の話者が声紋データベースに登録されているか。"""
        if not self.registered_speakers:
            return True
        return speaker in self.registered_speakers


@dataclass(frozen=True)
class MatchedUtterance:
    """対応が取れた1発話の突き合わせ結果。"""

    ground_truth: SubtitleEntry
    app: SubtitleEntry
    ground_truth_speaker: str
    app_speaker: str
    in_point_error: float
    judgement: str

    @property
    def is_speaker_correct(self) -> bool:
        """話者が正解と一致したか。"""
        return self.judgement == JUDGEMENT_CORRECT


@dataclass(frozen=True)
class EvaluationResult:
    """クリップ1本の評価結果。"""

    matched: tuple[MatchedUtterance, ...]
    unmatched_ground_truth_count: int
    unmatched_app_count: int
    in_point_correct_counts: dict[float, int]
    in_point_accuracies: dict[float, float]
    in_point_mean_error: float
    in_point_median_error: float
    in_point_max_error: float
    speaker_correct_count: int
    speaker_accuracy: float
    false_negative_count: int
    false_positive_count: int
    correct_unknown_count: int
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def matched_count(self) -> int:
        """対応が取れた発話の件数。"""
        return len(self.matched)


def fill_omitted_speakers(entries: list[SubtitleEntry]) -> list[str]:
    """話者名が省略された字幕に、直前の話者を引き継いで補完する。

    アプリ生成SRTは話者が変わったときだけ ``（話者名）`` を前置するため、
    そのまま比較すると「話者を付けられなかった」と誤って読んでしまう。

    Args:
        entries: 字幕エントリ。

    Returns:
        エントリと同じ順・同じ長さの話者名リスト。先頭が省略されている場合は空文字。
    """
    speakers = []
    previous = ""
    for entry in entries:
        speaker = entry.speaker or previous
        speakers.append(speaker)
        previous = speaker
    return speakers


def align_entries(
    ground_truth: list[SubtitleEntry],
    app: list[SubtitleEntry],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[tuple[int | None, int | None]]:
    """本文テキストで正解字幕とアプリ生成字幕を対応付ける。

    順序は保ったまま 1:1 で対応させる。完全一致で対応しなかった行は、同じ
    ずれの範囲にある未対応行の中から類似度が最大のものへ割り当てる。

    Args:
        ground_truth: 正解字幕のエントリ。
        app: アプリ生成字幕のエントリ。
        min_similarity: 類似度がこれ未満なら対応付けない。

    Returns:
        ``(正解の添字, アプリの添字)`` のリスト。対応が無い側は None。
    """
    ground_truth_keys = [match_key(entry.body) for entry in ground_truth]
    app_keys = [match_key(entry.body) for entry in app]
    matcher = difflib.SequenceMatcher(
        a=ground_truth_keys, b=app_keys, autojunk=False
    )

    pairs: list[tuple[int | None, int | None]] = []
    for tag, gt_from, gt_to, app_from, app_to in matcher.get_opcodes():
        if tag == "equal":
            pairs += [
                (gt_index, app_from + offset)
                for offset, gt_index in enumerate(range(gt_from, gt_to))
            ]
            continue
        pairs += _pair_by_similarity(
            ground_truth_keys[gt_from:gt_to],
            app_keys[app_from:app_to],
            (gt_from, app_from),
            min_similarity,
        )
    return pairs


def _pair_by_similarity(
    ground_truth_keys: list[str],
    app_keys: list[str],
    offsets: tuple[int, int],
    min_similarity: float,
) -> list[tuple[int | None, int | None]]:
    """一致しなかった塊の中で、順序を保ったまま似ている行同士を対応付ける。"""
    gt_offset, app_offset = offsets
    pairs: list[tuple[int | None, int | None]] = []
    used_app_indexes: set[int] = set()
    next_app_index = 0

    for gt_index, gt_key in enumerate(ground_truth_keys):
        best_index = _find_similar(gt_key, app_keys, next_app_index, min_similarity)
        if best_index is None:
            pairs.append((gt_offset + gt_index, None))
            continue
        pairs.append((gt_offset + gt_index, app_offset + best_index))
        used_app_indexes.add(best_index)
        next_app_index = best_index + 1

    pairs += [
        (None, app_offset + index)
        for index in range(len(app_keys))
        if index not in used_app_indexes
    ]
    return sorted(pairs, key=lambda pair: (pair[0] is None, pair[0], pair[1] or 0))


def _find_similar(
    key: str, candidates: list[str], start: int, min_similarity: float
) -> int | None:
    """``start`` 以降で最も似ている候補の添字を返す。閾値未満なら None。"""
    best_index = None
    best_ratio = min_similarity
    for index in range(start, len(candidates)):
        ratio = difflib.SequenceMatcher(a=key, b=candidates[index]).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best_index = index
    return best_index


def evaluate(
    ground_truth: list[SubtitleEntry],
    app: list[SubtitleEntry],
    options: EvaluationOptions,
) -> EvaluationResult:
    """正解字幕とアプリ生成字幕を突き合わせ、in点と話者を評価する。

    Args:
        ground_truth: 正解字幕のエントリ。
        app: アプリ生成字幕のエントリ。
        options: 評価の設定。

    Returns:
        評価結果。対応が取れなかった行は評価対象から外し、件数として残す。
    """
    ground_truth = [entry for entry in ground_truth if entry.is_speech]
    app = [entry for entry in app if entry.is_speech]
    ground_truth_speakers = fill_omitted_speakers(ground_truth)
    app_speakers = fill_omitted_speakers(app)

    matched = []
    unmatched_ground_truth = 0
    unmatched_app = 0
    for gt_index, app_index in align_entries(
        ground_truth, app, options.min_similarity
    ):
        if gt_index is None:
            unmatched_app += 1
            continue
        if app_index is None:
            unmatched_ground_truth += 1
            continue
        matched.append(
            _build_matched(
                (ground_truth[gt_index], app[app_index]),
                (ground_truth_speakers[gt_index], app_speakers[app_index]),
                options,
            )
        )

    return _summarize(tuple(matched), (unmatched_ground_truth, unmatched_app), options)


def _build_matched(
    entries: tuple[SubtitleEntry, SubtitleEntry],
    speakers: tuple[str, str],
    options: EvaluationOptions,
) -> MatchedUtterance:
    """対応が取れた1組から突き合わせ結果を作る。"""
    ground_truth, app = entries
    ground_truth_speaker, app_speaker = speakers
    return MatchedUtterance(
        ground_truth=ground_truth,
        app=app,
        ground_truth_speaker=ground_truth_speaker,
        app_speaker=app_speaker,
        in_point_error=abs(app.start - ground_truth.start),
        judgement=_judge_speaker(ground_truth_speaker, app_speaker, options),
    )


def _judge_speaker(
    ground_truth_speaker: str, app_speaker: str, options: EvaluationOptions
) -> str:
    """話者の付与結果を4種類に分類する（予稿 3.2）。"""
    if app_speaker == ground_truth_speaker:
        return JUDGEMENT_CORRECT
    if _is_unknown(app_speaker):
        if options.is_registered(ground_truth_speaker):
            return JUDGEMENT_FALSE_NEGATIVE
        return JUDGEMENT_CORRECT_UNKNOWN
    return JUDGEMENT_FALSE_POSITIVE


def _is_unknown(speaker: str) -> bool:
    """``Unknown_NN`` または空欄か。"""
    return not speaker or bool(_UNKNOWN_SPEAKER_RE.match(speaker))


def _summarize(
    matched: tuple[MatchedUtterance, ...],
    unmatched_counts: tuple[int, int],
    options: EvaluationOptions,
) -> EvaluationResult:
    """突き合わせ結果を集計する。"""
    errors = [utterance.in_point_error for utterance in matched]
    judgements = Counter(utterance.judgement for utterance in matched)
    correct_counts = {
        tolerance: sum(1 for error in errors if error <= tolerance)
        for tolerance in options.tolerances
    }
    confusion = Counter(
        (utterance.ground_truth_speaker, utterance.app_speaker)
        for utterance in matched
    )

    return EvaluationResult(
        matched=matched,
        unmatched_ground_truth_count=unmatched_counts[0],
        unmatched_app_count=unmatched_counts[1],
        in_point_correct_counts=correct_counts,
        in_point_accuracies={
            tolerance: _ratio(count, len(matched))
            for tolerance, count in correct_counts.items()
        },
        in_point_mean_error=statistics.fmean(errors) if errors else 0.0,
        in_point_median_error=statistics.median(errors) if errors else 0.0,
        in_point_max_error=max(errors) if errors else 0.0,
        speaker_correct_count=judgements[JUDGEMENT_CORRECT],
        speaker_accuracy=_ratio(judgements[JUDGEMENT_CORRECT], len(matched)),
        false_negative_count=judgements[JUDGEMENT_FALSE_NEGATIVE],
        false_positive_count=judgements[JUDGEMENT_FALSE_POSITIVE],
        correct_unknown_count=judgements[JUDGEMENT_CORRECT_UNKNOWN],
        confusion=dict(confusion),
    )


def _ratio(numerator: float, denominator: float) -> float:
    """0 除算を避けて割合を求める。"""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _app_rules(rules: LabelRules) -> LabelRules:
    """アプリ生成字幕を読むためのラベル設定を作る。

    アプリの出力には2形式ある。

    - ``(話者名)本文`` … 最終字幕（:mod:`src.subtitle.exporter`）
    - ``[話者名] 本文`` … 仮字幕（:mod:`src.web.converters`）

    アプリ側の ``[…]`` に効果音が入ることはないので、話者ラベルとして読む。
    これを外すと ``[話者] 本文`` 形式のとき全発話が false negative になる。
    """
    return LabelRules(
        extra_non_speech_labels=rules.extra_non_speech_labels,
        speaker_labels=rules.speaker_labels,
        should_strip_suffix=rules.should_strip_suffix,
        should_read_square_brackets=True,
    )


def load_registered_speakers(voice_db_dir: Path) -> tuple[str, ...]:
    """声紋データベースのディレクトリから登録話者名を集める。

    ``田村_叫び声.wav`` のような派生ファイルは ``田村`` に寄せる。

    Args:
        voice_db_dir: 声紋データベースのディレクトリ。

    Returns:
        登録話者名（重複なし・名前順）。
    """
    names = {
        path.stem.split(_VOICE_FILE_SEPARATOR)[0]
        for path in Path(voice_db_dir).glob("*.wav")
    }
    return tuple(sorted(names))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="アプリ生成字幕を正解字幕と突き合わせて評価する"
    )
    parser.add_argument("--gt", required=True, type=Path, help="正解字幕SRT")
    parser.add_argument(
        "--app", required=True, type=Path, help="アプリが生成した字幕SRT"
    )
    parser.add_argument("--name", help="結果に付ける名前。既定は正解SRTのファイル名")
    parser.add_argument("--voice-db", type=Path, help="声紋データベースのディレクトリ")
    parser.add_argument(
        "--speakers", nargs="*", default=[], help="登録話者名（--voice-db の代わり）"
    )
    parser.add_argument(
        "--rttm", type=Path, help="話者分離結果のRTTM（重なり時間の算出用）"
    )
    parser.add_argument(
        "--transcript", type=Path, help="人手文字起こしテキスト（件数の照合用）"
    )
    parser.add_argument(
        "--tolerance",
        nargs="+",
        type=float,
        default=[DEFAULT_TOLERANCE_SEC],
        help="in点の許容ずれ（秒）。複数指定できる",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="本文の類似度がこれ未満なら対応付けない",
    )
    parser.add_argument(
        "--non-speech", nargs="*", default=[], help="非発話として扱うラベル（完全一致）"
    )
    parser.add_argument(
        "--speaker", nargs="*", default=[], help="話者として扱うラベル（完全一致）"
    )
    parser.add_argument(
        "--strip-suffix",
        action="store_true",
        help="（中嶋／手話）の注記を落として寄せる",
    )
    parser.add_argument("--out-csv", type=Path, help="指標と説明変数の出力先CSV")
    parser.add_argument("--append", action="store_true", help="--out-csv へ追記する")
    parser.add_argument(
        "--out-detail", type=Path, help="1行1発話の突き合わせ結果の出力先CSV"
    )
    parser.add_argument("--out-confusion", type=Path, help="混同行列の出力先CSV")
    return parser.parse_args(argv)


def _print_summary(result: EvaluationResult, name: str) -> None:
    """評価結果を人が読める形で出す。"""
    print(f"=== {name} ===")
    print(
        f"対応付け: {result.matched_count} 件"
        f"（正解側の未対応 {result.unmatched_ground_truth_count} 件 / "
        f"アプリ側の未対応 {result.unmatched_app_count} 件）"
    )
    for tolerance, accuracy in result.in_point_accuracies.items():
        count = result.in_point_correct_counts[tolerance]
        print(f"in点 ±{tolerance:g}秒 以内: {count} 件（{accuracy:.1%}）")
    print(
        f"in点のずれ: 平均 {result.in_point_mean_error:.3f} 秒 / "
        f"中央 {result.in_point_median_error:.3f} 秒 / "
        f"最大 {result.in_point_max_error:.3f} 秒"
    )
    print(
        f"話者一致: {result.speaker_correct_count} 件（{result.speaker_accuracy:.1%}）"
        f" / false negative {result.false_negative_count} 件"
        f" / false positive {result.false_positive_count} 件"
        f" / 正しくUnknown {result.correct_unknown_count} 件"
    )


def _print_transcript_check(transcript: Path | None, utterance_count: int) -> None:
    """人手文字起こしの行数と正解字幕の発話数を照合する。"""
    if transcript is None:
        return
    lines = [
        line
        for line in Path(transcript).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) == utterance_count:
        print(
            f"文字起こしの行数と正解字幕の発話数は一致（{utterance_count} 件）"
        )
        return
    print(
        f"注意: 文字起こしは {len(lines)} 行ですが、"
        f"正解字幕の発話は {utterance_count} 件です。"
        "正解データの取りこぼしがないか確認してください。"
    )


def _build_summary_row(
    result: EvaluationResult,
    name: str,
    ground_truth: list[SubtitleEntry],
    rttm: Path | None,
) -> dict[str, object]:
    """CSV へ書き出す1行を組み立てる。"""
    speech = [entry for entry in ground_truth if entry.is_speech]
    duration = max((entry.end for entry in speech), default=0.0)
    variables = compute_variables(ground_truth, duration_sec=duration)

    row: dict[str, object] = {
        "name": name,
        "matched_count": result.matched_count,
        "unmatched_gt_count": result.unmatched_ground_truth_count,
        "unmatched_app_count": result.unmatched_app_count,
    }
    for tolerance, accuracy in result.in_point_accuracies.items():
        row[f"in_point_accuracy_{tolerance:g}s"] = round(accuracy, 4)
    row.update(
        {
            "in_point_mean_error": round(result.in_point_mean_error, 3),
            "in_point_median_error": round(result.in_point_median_error, 3),
            "in_point_max_error": round(result.in_point_max_error, 3),
            "speaker_accuracy": round(result.speaker_accuracy, 4),
            "speaker_correct_count": result.speaker_correct_count,
            "false_negative_count": result.false_negative_count,
            "false_positive_count": result.false_positive_count,
            "correct_unknown_count": result.correct_unknown_count,
            **variables.as_row(),
            "overlap_time_ratio": _overlap_time_ratio(rttm),
        }
    )
    return row


def _overlap_time_ratio(rttm: Path | None) -> object:
    """RTTM から重なり時間の割合を求める。RTTM が無ければ空欄。"""
    if rttm is None:
        return ""
    try:
        segments = parse_rttm(rttm)
    except OSError as e:
        logger.warning(f"RTTM を読めません（重なり時間は空欄にします）: {e}")
        return ""
    return round(compute_overlap_time_ratio(segments), 4)


def _write_detail(path: Path, result: EvaluationResult) -> None:
    """1行1発話の突き合わせ結果を書き出す（誤りの目視分析用）。"""
    rows = [
        {
            "gt_index": utterance.ground_truth.index,
            "app_index": utterance.app.index,
            "gt_start": round(utterance.ground_truth.start, 3),
            "app_start": round(utterance.app.start, 3),
            "in_point_error": round(utterance.in_point_error, 3),
            "gt_speaker": utterance.ground_truth_speaker,
            "app_speaker": utterance.app_speaker,
            "judgement": utterance.judgement,
            "text": utterance.ground_truth.body.replace("\n", " "),
        }
        for utterance in result.matched
    ]
    write_dict_rows(path, rows)
    print(f"突き合わせ結果を書き出しました: {path}")


def _write_confusion(path: Path, result: EvaluationResult) -> None:
    """「正解話者 × アプリ話者」のクロス表を書き出す。"""
    rows = [
        {"gt_speaker": pair[0], "app_speaker": pair[1], "count": count}
        for pair, count in sorted(result.confusion.items(), key=lambda kv: -kv[1])
    ]
    write_dict_rows(path, rows)
    print(f"混同行列を書き出しました: {path}")


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。"""
    configure_logging()
    args = _parse_args(argv)

    rules = LabelRules(
        extra_non_speech_labels=tuple(args.non_speech),
        speaker_labels=tuple(args.speaker),
        should_strip_suffix=args.strip_suffix,
    )
    try:
        ground_truth = parse_srt(args.gt, rules)
        # アプリ生成字幕は `(話者)本文`（src/subtitle/exporter.py）と
        # `[話者] 本文`（src/web/converters.py）の2形式があるため、両方を読む
        app = parse_srt(args.app, _app_rules(rules))
    except FileNotFoundError as e:
        logger.error(f"字幕ファイルが見つかりません: {e}")
        return 1

    registered = tuple(args.speakers)
    if args.voice_db:
        registered = load_registered_speakers(args.voice_db)
        print(f"登録話者（{len(registered)}名）: {' '.join(registered)}")

    options = EvaluationOptions(
        tolerances=tuple(args.tolerance),
        min_similarity=args.min_similarity,
        registered_speakers=registered,
    )
    result = evaluate(ground_truth, app, options)

    name = args.name or Path(args.gt).stem
    _print_summary(result, name)
    _print_transcript_check(
        args.transcript, sum(1 for entry in ground_truth if entry.is_speech)
    )

    if args.out_csv:
        row = _build_summary_row(result, name, ground_truth, args.rttm)
        write_dict_rows(args.out_csv, [row], should_append=args.append)
        print(f"評価結果を書き出しました: {args.out_csv}")
    if args.out_detail:
        _write_detail(args.out_detail, result)
    if args.out_confusion:
        _write_confusion(args.out_confusion, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
