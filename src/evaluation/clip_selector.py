"""正解字幕から実験用の5分クリップ2本を選ぶ（予稿 4.3）。

「重なり・話者交替が多い区間（hard）」と「落ち着いた区間（calm）」を、
**発話境界に合わせて**自動で選ぶ。入力は正解字幕SRTだけで、声紋データベースは見ない。

手順

1. 各発話の in 点を窓の開始候補とし、``--length`` 秒以上になる最初の発話境界で閉じる。
2. 窓ごとに説明変数を出し、話者交替率と重なり率を z 正規化して足したスコアを付ける。
3. 時間が重ならない2窓の全組み合わせから、スコア差が最大になるペアを選ぶ。

音声・動画の切り出しは行わず、ffmpeg コマンドを表示するだけに留める。

実行例::

    uv run python -m src.evaluation.clip_selector \\
        --srt "字幕/転スラ.srt" --video 転スラ1話.mp4 \\
        --target-speakers 5 \\
        --out-csv docs/experiment/clips.csv --append \\
        --out-srt-dir docs/experiment/gt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from src.common.audio import build_extract_command
from src.common.csv_io import write_dict_rows
from src.common.logging import configure_logging, get_logger
from src.common.timecode import seconds_to_time_str
from src.evaluation.srt_stats import (
    ClipVariables,
    LabelRules,
    SubtitleEntry,
    compute_variables,
    parse_srt,
)

logger = get_logger(__name__)

#: 既定のクリップ長（秒）。予稿 4.3 の「5分のクリップ」。
DEFAULT_LENGTH_SEC = 300.0

#: 発話境界に合わせるために許す、クリップ長のずれ（秒）。
DEFAULT_TOLERANCE_SEC = 15.0

#: 目標話者数から1人ずれるごとに難易度スコアから引く値。
DEFAULT_SPEAKER_PENALTY = 0.5

#: 浮動小数の比較で境界を取りこぼさないための許容誤差（秒）。
_EPSILON_SEC = 1e-6


@dataclass(frozen=True)
class SelectionOptions:
    """クリップ選定の設定。

    Attributes:
        length_sec: 目標のクリップ長（秒）。
        tolerance_sec: 発話境界に合わせるために許すクリップ長のずれ（秒）。
        weight_change: 難易度スコアでの話者交替率の重み。
        weight_overlap: 難易度スコアでの重なり率の重み。
        target_speakers: 揃えたい話者数。None なら揃えない。
        speaker_penalty: 目標話者数から1人ずれるごとに引く値。
    """

    length_sec: float = DEFAULT_LENGTH_SEC
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC
    weight_change: float = 1.0
    weight_overlap: float = 1.0
    target_speakers: int | None = None
    speaker_penalty: float = DEFAULT_SPEAKER_PENALTY


@dataclass(frozen=True)
class ClipWindow:
    """クリップの候補区間。

    Attributes:
        start: 開始秒（ある発話の in 点）。
        end: 終了秒（ある発話の out 点）。
        entries: 区間に含まれる字幕エントリ。
        variables: 区間の説明変数。
    """

    start: float
    end: float
    entries: tuple[SubtitleEntry, ...]
    variables: ClipVariables


def enumerate_windows(
    entries: list[SubtitleEntry], options: SelectionOptions
) -> list[ClipWindow]:
    """発話境界に合わせた候補区間をすべて列挙する。

    Args:
        entries: 正解字幕の全エントリ。
        options: 選定の設定。``length_sec`` と ``tolerance_sec`` を使う。

    Returns:
        開始時刻の昇順に並んだ候補区間。1本も作れない場合は空リスト。
    """
    speech = [entry for entry in entries if entry.is_speech]
    boundaries = sorted({entry.end for entry in speech})

    windows = []
    for entry in speech:
        end = _first_boundary_at_least(boundaries, entry.start + options.length_sec)
        if end is None:
            continue
        if abs(end - entry.start - options.length_sec) > options.tolerance_sec:
            continue
        windows.append(_build_window(entries, entry.start, end))
    return windows


def _first_boundary_at_least(boundaries: list[float], lower: float) -> float | None:
    """``lower`` 以上で最も早い発話境界を返す。無ければ None。"""
    for boundary in boundaries:
        if boundary >= lower - _EPSILON_SEC:
            return boundary
    return None


def _build_window(
    entries: list[SubtitleEntry], start: float, end: float
) -> ClipWindow:
    """区間に収まる字幕を集め、説明変数を付けた候補区間を作る。"""
    included = tuple(
        entry
        for entry in entries
        if entry.start >= start - _EPSILON_SEC and entry.end <= end + _EPSILON_SEC
    )
    return ClipWindow(
        start=start,
        end=end,
        entries=included,
        variables=compute_variables(list(included), duration_sec=end - start),
    )


def score_windows(
    windows: list[ClipWindow], options: SelectionOptions
) -> list[float]:
    """候補区間に難易度スコアを付ける。

    話者交替率と重なり率をそれぞれ z 正規化して重み付きで足す。ばらつきが無い
    指標は寄与 0 とする（例: 重なり表記が無い番組では話者交替だけでスコアが決まる）。

    Args:
        windows: 候補区間。
        options: 重みと話者数ペナルティの設定。

    Returns:
        ``windows`` と同じ順のスコア。
    """
    changes = _standardize([w.variables.speaker_change_per_min for w in windows])
    overlaps = _standardize([w.variables.overlap_entry_ratio for w in windows])

    scores = []
    for window, change, overlap in zip(windows, changes, overlaps):
        score = options.weight_change * change + options.weight_overlap * overlap
        scores.append(score - _speaker_penalty(window, options))
    return scores


def _standardize(values: list[float]) -> list[float]:
    """値を z 正規化する。ばらつきが無ければすべて 0 にする。"""
    if len(values) < 2:
        return [0.0] * len(values)
    deviation = statistics.pstdev(values)
    if deviation == 0:
        return [0.0] * len(values)
    mean = statistics.fmean(values)
    return [(value - mean) / deviation for value in values]


def _speaker_penalty(window: ClipWindow, options: SelectionOptions) -> float:
    """目標話者数からのずれに応じたペナルティを返す。"""
    if options.target_speakers is None:
        return 0.0
    difference = abs(window.variables.speaker_count - options.target_speakers)
    return difference * options.speaker_penalty


def select_clip_pair(
    windows: list[ClipWindow], options: SelectionOptions
) -> tuple[ClipWindow, ClipWindow]:
    """時間が重ならない2本を、難易度スコアの差が最大になるように選ぶ。

    Args:
        windows: :func:`enumerate_windows` が返した候補区間。
        options: 選定の設定。

    Returns:
        ``(hard, calm)``。hard がスコアの高い方。

    Raises:
        ValueError: 候補が無い、または重ならない2本が取れない場合。
    """
    if not windows:
        raise ValueError(
            "クリップの候補が1本もありません。"
            "--length を短くするか --tolerance を広げてください。"
        )

    scores = score_windows(windows, options)
    best_pair: tuple[int, int] | None = None
    best_difference = -1.0

    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            if _is_overlapping(windows[i], windows[j]):
                continue
            difference = abs(scores[i] - scores[j])
            if difference > best_difference:
                best_difference = difference
                best_pair = (i, j) if scores[i] >= scores[j] else (j, i)

    if best_pair is None:
        raise ValueError(
            "時間が重ならない2本のクリップが取れません。"
            f"クリップ長 {options.length_sec:.0f} 秒の2倍より動画が短いようです。"
        )
    return windows[best_pair[0]], windows[best_pair[1]]


def _is_overlapping(left: ClipWindow, right: ClipWindow) -> bool:
    """2つの候補区間が時間的に重なっているか。"""
    return (
        left.start < right.end - _EPSILON_SEC
        and right.start < left.end - _EPSILON_SEC
    )


def shift_entries(
    entries: tuple[SubtitleEntry, ...], offset: float
) -> list[SubtitleEntry]:
    """字幕の時刻を ``offset`` 秒だけ手前へずらし、番号を振り直す。

    切り出したクリップの先頭を 0 秒に合わせ、クリップ単体で評価に使えるようにする。

    Args:
        entries: 元の字幕エントリ。
        offset: 差し引く秒数（クリップの開始秒）。

    Returns:
        時刻をずらした字幕エントリ。番号は 1 起点で振り直す。
    """
    shifted = []
    for number, entry in enumerate(entries, start=1):
        shifted.append(
            SubtitleEntry(
                index=number,
                start=max(0.0, entry.start - offset),
                end=max(0.0, entry.end - offset),
                speakers=entry.speakers,
                body=entry.body,
                non_speech_labels=entry.non_speech_labels,
                raw_text=entry.raw_text,
            )
        )
    return shifted


def write_srt(entries: list[SubtitleEntry], path: Path) -> None:
    """字幕エントリを SRT として書き出す。

    本文は元の SRT の記述をそのまま使う（話者ラベルの位置も保つ）。

    Args:
        entries: 書き出す字幕エントリ。
        path: 出力先。親ディレクトリが無ければ作る。

    Raises:
        OSError: 書き込みに失敗した場合。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for entry in entries:
        text = entry.raw_text or _rebuild_text(entry)
        start = seconds_to_time_str(entry.start)
        end = seconds_to_time_str(entry.end)
        blocks.append(f"{entry.index}\n{start} --> {end}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def _rebuild_text(entry: SubtitleEntry) -> str:
    """元の記述が無いエントリの本文を、話者ラベル付きで組み立てる。"""
    labels = "".join(f"（{speaker}）" for speaker in entry.speakers)
    return f"{labels}{entry.body}"


def build_ffmpeg_command(video: Path, window: ClipWindow, output: Path) -> list[str]:
    """クリップ切り出し用の ffmpeg コマンドを組み立てる（実行はしない）。"""
    return build_extract_command(
        Path(video),
        Path(output),
        start=window.start,
        end=window.end,
        to_wav16k=False,
        quiet=False,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="正解字幕から実験用の5分クリップ2本（hard / calm）を選ぶ"
    )
    parser.add_argument("--srt", required=True, type=Path, help="正解字幕SRTのパス")
    parser.add_argument("--name", help="クリップ名の接頭辞。既定はSRTのファイル名")
    parser.add_argument("--video", type=Path, help="ffmpeg コマンドに使う動画のパス")
    parser.add_argument(
        "--length", type=float, default=DEFAULT_LENGTH_SEC, help="クリップ長（秒）"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_SEC,
        help="発話境界に合わせるために許すクリップ長のずれ（秒）",
    )
    parser.add_argument(
        "--weight-change", type=float, default=1.0, help="話者交替率の重み"
    )
    parser.add_argument(
        "--weight-overlap", type=float, default=1.0, help="重なり率の重み"
    )
    parser.add_argument(
        "--target-speakers",
        type=int,
        help="揃えたい話者数。指定するとずれをペナルティにする",
    )
    parser.add_argument(
        "--speaker-penalty",
        type=float,
        default=DEFAULT_SPEAKER_PENALTY,
        help="目標話者数から1人ずれるごとに引く値",
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
        help="（中嶋／手話）のような注記を落として同一話者に寄せる",
    )
    parser.add_argument("--out-csv", type=Path, help="説明変数の出力先CSV")
    parser.add_argument(
        "--append",
        action="store_true",
        help="--out-csv へ追記する（複数番組の集約用）",
    )
    parser.add_argument("--out-srt-dir", type=Path, help="切り出した正解SRTの出力先")
    return parser.parse_args(argv)


def _to_options(args: argparse.Namespace) -> SelectionOptions:
    return SelectionOptions(
        length_sec=args.length,
        tolerance_sec=args.tolerance,
        weight_change=args.weight_change,
        weight_overlap=args.weight_overlap,
        target_speakers=args.target_speakers,
        speaker_penalty=args.speaker_penalty,
    )


def _print_label_judgement(entries: list[SubtitleEntry]) -> None:
    """話者・非発話の判定結果を一覧で出す（キーワード判定の目視確認用）。"""
    speakers = sorted({s for entry in entries for s in entry.speakers})
    non_speech = sorted(
        {label for entry in entries for label in entry.non_speech_labels}
    )
    print(f"話者と判定したラベル（{len(speakers)}件）: {' / '.join(speakers)}")
    print(f"非発話と判定したラベル（{len(non_speech)}件）: {' / '.join(non_speech)}")
    print("誤りがあれば --speaker / --non-speech で上書きしてください。\n")


def _print_window(kind: str, window: ClipWindow, score: float) -> None:
    """選ばれたクリップの内容を人が読める形で出す。"""
    variables = window.variables
    start = seconds_to_time_str(window.start)
    end = seconds_to_time_str(window.end)
    print(f"[{kind}] {start} --> {end}")
    print(f"  実長 {window.end - window.start:.1f} 秒 / 難易度スコア {score:+.3f}")
    print(f"  登場人数 {variables.speaker_count} 名（発話数の多い順）")
    print("    " + " / ".join(
        f"{speaker} {count}" for speaker, count in variables.utterance_counts
    ))
    print(
        f"  発話数 {variables.utterance_count} / "
        f"話者交替 {variables.speaker_change_count} 回"
        f"（{variables.speaker_change_per_min:.2f} 回/分）"
    )
    print(
        f"  重なり {variables.overlap_entry_count} 件"
        f"（{variables.overlap_entry_ratio:.1%}） / "
        f"発話時間の割合 {variables.speech_time_ratio:.1%}"
    )


def _warn_if_no_overlap_signal(windows: list[ClipWindow]) -> None:
    """重なり表記が全く無い場合に、難易度が話者交替だけで決まることを警告する。"""
    if any(window.variables.overlap_entry_count > 0 for window in windows):
        return
    print(
        "警告: この字幕には重なり（1エントリ内の別話者）が1件もありません。"
        "難易度は話者交替回数だけで決まります。\n"
    )


def _write_outputs(
    args: argparse.Namespace, clips: dict[str, ClipWindow], name: str
) -> None:
    """切り出しSRTと CSV を書き出す。

    SRT を先に書く。CSV は説明変数なので後から作り直せるが、切り出した正解SRT は
    以降の評価の土台になるため、片方が失敗してももう片方を落とさない順序にする。
    """
    if args.out_srt_dir:
        for kind, window in clips.items():
            output = Path(args.out_srt_dir) / f"{name}_{kind}.srt"
            write_srt(shift_entries(window.entries, offset=window.start), output)
            print(f"正解SRTを切り出しました: {output}")

    if args.out_csv:
        rows = [
            {
                "name": name,
                "clip": kind,
                "start_time": seconds_to_time_str(window.start),
                "end_time": seconds_to_time_str(window.end),
                "start_sec": round(window.start, 3),
                "end_sec": round(window.end, 3),
                **window.variables.as_row(),
            }
            for kind, window in clips.items()
        ]
        write_dict_rows(args.out_csv, rows, should_append=args.append)
        print(f"説明変数を書き出しました: {args.out_csv}")


def _print_ffmpeg_commands(
    video: Path | None, clips: dict[str, ClipWindow], name: str
) -> None:
    """切り出し用の ffmpeg コマンドを表示する（実行はしない）。"""
    if video is None:
        return
    print("\n切り出しコマンド（確認してから実行してください）:")
    for kind, window in clips.items():
        output = Path(f"{name}_{kind}{Path(video).suffix}")
        command = build_ffmpeg_command(video, window, output)
        print("  " + " ".join(command))


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
        entries = parse_srt(args.srt, rules)
    except FileNotFoundError:
        logger.error(f"字幕ファイルが見つかりません: {args.srt}")
        return 1

    if not entries:
        logger.error(f"字幕を1件も読み込めませんでした: {args.srt}")
        return 1

    name = args.name or Path(args.srt).stem
    options = _to_options(args)

    _print_label_judgement(entries)
    windows = enumerate_windows(entries, options)
    _warn_if_no_overlap_signal(windows)

    try:
        hard, calm = select_clip_pair(windows, options)
    except ValueError as e:
        logger.error(str(e))
        return 1

    scores = dict(zip(windows, score_windows(windows, options)))
    _print_window("hard（重なり・話者交替が多い）", hard, scores[hard])
    _print_window("calm（落ち着いた）", calm, scores[calm])

    clips = {"hard": hard, "calm": calm}
    _write_outputs(args, clips, name)
    _print_ffmpeg_commands(args.video, clips, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
