"""正解字幕SRTの解析と、実験の説明変数（予稿 4.3）の算出。

クリップ選定（:mod:`src.evaluation.clip_selector`）と評価
（:mod:`src.evaluation.evaluator`）が共通して使う土台。

既存の :mod:`src.subtitle.loader` は音声認識SRT（``[話者名]`` 前置）専用で、
BOM や1エントリ内の複数話者に対応していないため流用せず、この用途の
パーサを持つ。

**話者ラベルの拾い方**（納品字幕の実物に合わせている）

- 話者は ``（…）`` からのみ拾う。``[…]`` は効果音・音楽の表記なので話者に数えない。
- ラベルとみなすのは**行頭・空白の直後・直前のラベルの直後**から始まる括弧だけ。
  ``Claude（クロード）`` のような言い換えを話者と誤認しないため。
- ``（）`` の中の効果音・音楽（``（軽快なBGM）`` など）はキーワードで除外する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.subtitle.ngram import normalize_text

#: 括弧の開き。全角・半角の両方を受ける。
_OPEN_PARENS = "（("

#: 括弧の閉じ。開きと同じ順で対応する。
_CLOSE_PARENS = "）)"

#: 角括弧の開き。既定では効果音・音楽の表記なのでラベル候補にしない。
_OPEN_SQUARE_BRACKETS = "[［"

#: 角括弧の閉じ。開きと同じ順で対応する。
_CLOSE_SQUARE_BRACKETS = "]］"

#: 効果音・音楽の表記。話者ではないので ``[…]`` は最初からラベル候補にしない。
_SQUARE_BRACKET_RE = re.compile(r"[\[［][^\]］]*[\]］]")

#: 字幕エントリの区切り（空行）。
_ENTRY_SEPARATOR_RE = re.compile(r"\n\s*\n")

#: ``00:00:02,190 --> 00:00:04,820`` の時刻行。ミリ秒はカンマ・ピリオドのどちらでも。
_TIME_LINE_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*"
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)

#: この語を含むラベルは非発話（効果音・音楽）とみなす。部分一致で判定する。
DEFAULT_NON_SPEECH_WORDS = (
    "BGM",
    "音楽",
    "効果音",
    "ジングル",
    "笑い",
    "拍手",
    "ざわめき",
    "歓声",
)

#: この語で終わるラベルは非発話とみなす（``ドアの作動音`` など）。
DEFAULT_NON_SPEECH_SUFFIX = "音"

#: 完全一致でだけ非発話とみなすラベル。部分一致だと人名を巻き込む（``笑福亭``）。
DEFAULT_NON_SPEECH_LABELS = ("笑",)

#: ``中嶋／手話・音声通訳`` を ``中嶋`` に寄せるときの区切り。
_SUFFIX_SEPARATORS = "／・"

_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60
_MS_PER_SECOND = 1000.0

#: RTTM の列位置（1行1区間・空白区切り）。開始秒・継続秒・話者ラベル。
_RTTM_START_INDEX = 3
_RTTM_DURATION_INDEX = 4
_RTTM_SPEAKER_INDEX = 7
_RTTM_MIN_FIELDS = 8


@dataclass(frozen=True)
class LabelRules:
    """話者ラベルの判定を上書きするための設定。

    既定のキーワード判定は完全ではない（``（謎の声）`` は話者だが
    ``（驚く声）`` は非発話、のように語では割り切れない）。そのため
    CLI から個別に上書きできるようにしている。

    Attributes:
        extra_non_speech_labels: 非発話として扱うラベル（完全一致）。
        speaker_labels: キーワードに関わらず話者として扱うラベル（完全一致）。
        should_strip_suffix: True なら ``中嶋／手話`` を ``中嶋`` に寄せる。
        should_read_square_brackets: True なら ``[…]`` も話者ラベルとして読む。
            **アプリ生成字幕の評価用**。アプリの仮字幕（:mod:`src.web.converters`）は
            ``[話者] 本文`` 形式で、``[…]`` に効果音が入ることはないため。
            正解字幕（人の手で ``[荘厳な音楽]`` が入る）では False のままにする。
    """

    extra_non_speech_labels: tuple[str, ...] = ()
    speaker_labels: tuple[str, ...] = ()
    should_strip_suffix: bool = False
    should_read_square_brackets: bool = False

    @property
    def open_chars(self) -> str:
        """ラベルの開き括弧として扱う文字。"""
        if self.should_read_square_brackets:
            return _OPEN_PARENS + _OPEN_SQUARE_BRACKETS
        return _OPEN_PARENS

    @property
    def close_chars(self) -> str:
        """ラベルの閉じ括弧として扱う文字（開きと同じ並び順）。"""
        if self.should_read_square_brackets:
            return _CLOSE_PARENS + _CLOSE_SQUARE_BRACKETS
        return _CLOSE_PARENS

    def is_speaker(self, label: str) -> bool:
        """ラベルが話者名か（＝効果音・音楽でないか）を判定する。"""
        if label in self.speaker_labels:
            return True
        if label in self.extra_non_speech_labels:
            return False
        if label in DEFAULT_NON_SPEECH_LABELS:
            return False
        if any(word in label for word in DEFAULT_NON_SPEECH_WORDS):
            return False
        return not label.endswith(DEFAULT_NON_SPEECH_SUFFIX)

    def speaker_name(self, label: str) -> str:
        """話者ラベルを、話者を数えるときの名前へ整える。"""
        if not self.should_strip_suffix:
            return label
        name = _strip_parenthesized(label)
        for separator in _SUFFIX_SEPARATORS:
            name = name.split(separator)[0]
        return name.strip() or label


@dataclass(frozen=True)
class SubtitleEntry:
    """字幕1エントリ。

    Attributes:
        index: SRT 内の通し番号（1 起点）。
        start: 開始秒。
        end: 終了秒。
        speakers: 話者名（出現順・重複なし）。非発話ラベルは含まない。
        body: 話者ラベルと ``[…]`` を除いた本文。
        non_speech_labels: 効果音・音楽と判定したラベル（出現順・重複あり）。
        raw_text: SRT に書かれていた本文そのまま。切り出した SRT を書き戻すときに使う。
    """

    index: int
    start: float
    end: float
    speakers: tuple[str, ...]
    body: str
    non_speech_labels: tuple[str, ...] = ()
    raw_text: str = ""

    @property
    def duration(self) -> float:
        """表示時間（秒）。終了が開始より前なら 0。"""
        return max(0.0, self.end - self.start)

    @property
    def is_speech(self) -> bool:
        """発話を含むエントリか。音楽・効果音だけのエントリは False。"""
        return bool(self.speakers) or bool(self.body.strip())

    @property
    def speaker(self) -> str:
        """代表話者（最初に現れた話者）。話者ラベルが無ければ空文字。"""
        return self.speakers[0] if self.speakers else ""

    @property
    def has_overlap(self) -> bool:
        """1エントリに2人以上の話者が入っているか（重なりの手がかり）。"""
        return len(self.speakers) >= 2


@dataclass(frozen=True)
class ClipVariables:
    """クリップ1本の説明変数（予稿 4.3 の記録項目）。

    ``speaker_count`` は**その区間に登場した人数**。クリップ間で人数を揃えるのは
    番組の性質上できないため（対談番組は常に2人、バラエティは最小でも8人など）、
    揃えずに説明変数として記録する方針を採る。誰が何回喋ったかは
    ``utterance_counts`` に残す。
    """

    duration_sec: float
    speaker_count: int
    utterance_count: int
    speaker_change_count: int
    speaker_change_per_min: float
    overlap_entry_count: int
    overlap_entry_ratio: float
    speech_time_ratio: float
    speakers: tuple[str, ...] = field(default=())
    utterance_counts: tuple[tuple[str, int], ...] = field(default=())

    def as_row(self) -> dict[str, object]:
        """CSV へ書き出すための1行分の辞書を返す。"""
        return {
            "duration_sec": round(self.duration_sec, 3),
            "speaker_count": self.speaker_count,
            "utterance_count": self.utterance_count,
            "speaker_change_count": self.speaker_change_count,
            "speaker_change_per_min": round(self.speaker_change_per_min, 3),
            "overlap_entry_count": self.overlap_entry_count,
            "overlap_entry_ratio": round(self.overlap_entry_ratio, 4),
            "speech_time_ratio": round(self.speech_time_ratio, 4),
            "speaker_utterances": " ".join(
                f"{speaker}:{count}" for speaker, count in self.utterance_counts
            ),
        }


@dataclass(frozen=True)
class SpeakerSegment:
    """話者分離が出した1区間（RTTM の1行）。"""

    speaker: str
    start: float
    end: float


def parse_srt(path: Path, rules: LabelRules | None = None) -> list[SubtitleEntry]:
    """SRT を読み込み、話者ラベルを切り離した字幕エントリのリストを返す。

    Args:
        path: 読み込む SRT のパス。BOM 付き UTF-8 も受け付ける。
        rules: 話者ラベルの判定設定。None なら既定のキーワード判定。

    Returns:
        時刻の昇順に並んだ字幕エントリ。時刻行が無いブロックは読み飛ばす。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    rules = rules or LabelRules()
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    entries = []
    for block in _ENTRY_SEPARATOR_RE.split(text.strip()):
        entry = _parse_block(block, len(entries) + 1, rules)
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda e: e.start)


def _parse_block(block: str, index: int, rules: LabelRules) -> SubtitleEntry | None:
    """SRT の1ブロックを字幕エントリへ変換する。時刻行が無ければ None。"""
    lines = [line for line in block.split("\n") if line.strip()]
    time_line_position = _find_time_line(lines)
    if time_line_position is None:
        return None

    start, end = _parse_time_line(lines[time_line_position])
    text_lines = lines[time_line_position + 1 :]
    speakers, non_speech, body = _split_labels(text_lines, rules)
    return SubtitleEntry(
        index=index,
        start=start,
        end=end,
        speakers=speakers,
        body=body,
        non_speech_labels=non_speech,
        raw_text="\n".join(text_lines),
    )


def _find_time_line(lines: list[str]) -> int | None:
    """時刻行の位置を返す。見つからなければ None。"""
    for position, line in enumerate(lines):
        if _TIME_LINE_RE.search(line):
            return position
    return None


def _parse_time_line(line: str) -> tuple[float, float]:
    """時刻行から開始秒・終了秒を取り出す。"""
    matched = _TIME_LINE_RE.search(line)
    assert matched is not None  # _find_time_line で存在を確認済み
    values = [int(v) for v in matched.groups()]
    return _to_seconds(values[:4]), _to_seconds(values[4:])


def _to_seconds(parts: list[int]) -> float:
    """``[時, 分, 秒, ミリ秒]`` を秒へ変換する。"""
    hours, minutes, seconds, millis = parts
    return (
        hours * _SECONDS_PER_HOUR
        + minutes * _SECONDS_PER_MINUTE
        + seconds
        + millis / _MS_PER_SECOND
    )


def _split_labels(
    lines: list[str], rules: LabelRules
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """本文行から話者ラベルを切り離す。

    Returns:
        ``(話者名, 非発話ラベル, 本文)``。話者名は出現順で重複を除く。
    """
    speakers: list[str] = []
    non_speech: list[str] = []
    body_lines: list[str] = []

    for line in lines:
        if rules.should_read_square_brackets:
            labels, remainder = _extract_labels(line, rules)
        else:
            labels, remainder = _extract_labels(_SQUARE_BRACKET_RE.sub("", line), rules)
            non_speech += _SQUARE_BRACKET_RE.findall(line)
        for label in labels:
            if not rules.is_speaker(label):
                non_speech.append(label)
                continue
            name = rules.speaker_name(label)
            if name not in speakers:
                speakers.append(name)
        if remainder.strip():
            body_lines.append(remainder.strip())

    return tuple(speakers), tuple(_unwrap_brackets(non_speech)), "\n".join(body_lines)


def _extract_labels(line: str, rules: LabelRules) -> tuple[list[str], str]:
    """行頭・空白の直後・直前のラベルの直後にある ``（…）`` をラベルとして切り出す。

    Returns:
        ``(ラベルの中身のリスト, ラベルを除いた残りの文字列)``。
    """
    labels: list[str] = []
    remainder: list[str] = []
    position = 0
    is_label_position = True

    while position < len(line):
        char = line[position]
        if is_label_position and char in rules.open_chars:
            label, next_position = _read_parenthesized(
                line, position, rules.open_chars, rules.close_chars
            )
            if label is not None:
                labels.append(label)
                position = next_position
                continue
        remainder.append(char)
        is_label_position = char.isspace()
        position += 1

    return labels, "".join(remainder)


def _read_parenthesized(
    line: str,
    start: int,
    open_chars: str = _OPEN_PARENS,
    close_chars: str = _CLOSE_PARENS,
) -> tuple[str | None, int]:
    """``start`` の括弧に対応する閉じ括弧までを読む（入れ子に対応）。

    Returns:
        ``(括弧の中身, 閉じ括弧の次の位置)``。閉じ括弧が無ければ ``(None, start)``。
    """
    depth = 0
    for position in range(start, len(line)):
        if line[position] in open_chars:
            depth += 1
        elif line[position] in close_chars:
            depth -= 1
            if depth == 0:
                return line[start + 1 : position], position + 1
    return None, start


def _strip_parenthesized(label: str) -> str:
    """ラベルから入れ子の括弧ごと注記を落とす（``大統領（VTR）`` → ``大統領``）。"""
    stripped, _ = _extract_labels_from_tail(label)
    return stripped


def _extract_labels_from_tail(label: str) -> tuple[str, list[str]]:
    """ラベル中の括弧を取り除く。"""
    remainder: list[str] = []
    removed: list[str] = []
    position = 0
    while position < len(label):
        if label[position] in _OPEN_PARENS:
            inner, next_position = _read_parenthesized(label, position)
            if inner is not None:
                removed.append(inner)
                position = next_position
                continue
        remainder.append(label[position])
        position += 1
    return "".join(remainder), removed


def _unwrap_brackets(labels: list[str]) -> list[str]:
    """``[荘厳な音楽]`` のような角括弧付きのラベルから括弧を外す。"""
    return [label.strip("[]［］") for label in labels]


def match_key(text: str) -> str:
    """本文を、字幕どうしの対応付けに使う比較キーへ変換する。

    括弧の注記（``（笑）`` ``[音楽]``）を落としてから
    :func:`src.subtitle.ngram.normalize_text` で句読点と空白のゆらぎを吸収する。

    Args:
        text: 比較したい本文。

    Returns:
        比較キー。記号だけだった場合は空文字。
    """
    without_square_brackets = _SQUARE_BRACKET_RE.sub(" ", text)
    return normalize_text(_strip_parenthesized(without_square_brackets))


def compute_variables(
    entries: list[SubtitleEntry], duration_sec: float
) -> ClipVariables:
    """字幕エントリから説明変数（予稿 4.3）を算出する。

    Args:
        entries: 対象区間の字幕エントリ。非発話のエントリを含んでよい。
        duration_sec: 区間長（秒）。割合の分母に使う。

    Returns:
        算出した説明変数。発話が無い場合も 0 埋めした値を返す（0 除算しない）。
    """
    speech = [entry for entry in entries if entry.is_speech]
    utterance_counts = _count_utterances_by_speaker(speech)
    speakers = [speaker for speaker, _ in utterance_counts]
    overlap_count = sum(1 for entry in speech if entry.has_overlap)
    change_count = _count_speaker_changes(speech)
    speech_time = sum(entry.duration for entry in speech)

    return ClipVariables(
        duration_sec=duration_sec,
        speaker_count=len(speakers),
        utterance_count=len(speech),
        speaker_change_count=change_count,
        speaker_change_per_min=_ratio(change_count, duration_sec / _SECONDS_PER_MINUTE),
        overlap_entry_count=overlap_count,
        overlap_entry_ratio=_ratio(overlap_count, len(speech)),
        speech_time_ratio=_ratio(speech_time, duration_sec),
        speakers=tuple(speakers),
        utterance_counts=utterance_counts,
    )


def _count_utterances_by_speaker(
    entries: list[SubtitleEntry],
) -> tuple[tuple[str, int], ...]:
    """話者ごとの発話数を、多い順（同数なら出現順）に数える。

    1エントリに2人以上いる場合は、その全員に1件ずつ数える。
    """
    counts: dict[str, int] = {}
    for entry in entries:
        for speaker in entry.speakers:
            counts[speaker] = counts.get(speaker, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    return tuple(ordered)


def _count_speaker_changes(entries: list[SubtitleEntry]) -> int:
    """代表話者が直前と変わった回数を数える。"""
    changes = 0
    previous = ""
    for entry in entries:
        if entry.speaker and entry.speaker != previous:
            if previous:
                changes += 1
            previous = entry.speaker
    return changes


def _ratio(numerator: float, denominator: float) -> float:
    """0 除算を避けて割合を求める。分母が 0 なら 0.0。"""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def parse_rttm(path: Path) -> list[SpeakerSegment]:
    """RTTM を読み込み、話者区間のリストを返す。

    RTTM は空白区切りの9列。4列目が開始秒、5列目が**継続秒**、8列目が話者ラベル。

    Args:
        path: 読み込む RTTM のパス。

    Returns:
        話者区間のリスト。列が足りない行・数値として読めない行は読み飛ばす。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        segment = _parse_rttm_line(line)
        if segment is not None:
            segments.append(segment)
    return segments


def _parse_rttm_line(line: str) -> SpeakerSegment | None:
    """RTTM の1行を話者区間へ変換する。読めない行は None。"""
    fields = line.split()
    if len(fields) < _RTTM_MIN_FIELDS:
        return None
    try:
        start = float(fields[_RTTM_START_INDEX])
        duration = float(fields[_RTTM_DURATION_INDEX])
    except ValueError:
        return None
    return SpeakerSegment(
        speaker=fields[_RTTM_SPEAKER_INDEX], start=start, end=start + duration
    )


def compute_overlap_time_ratio(segments: list[SpeakerSegment]) -> float:
    """2人以上が同時に喋っている時間の割合を求める。

    区間の端点でイベントを作って走査し、同時に立っている区間の数を数える。

    Args:
        segments: 話者区間のリスト（重なっていてよい）。

    Returns:
        ``2人以上が喋っている時間 / 1人以上が喋っている時間``。区間が無ければ 0.0。
    """
    events = []
    for segment in segments:
        if segment.end <= segment.start:
            continue
        events.append((segment.start, 1))
        events.append((segment.end, -1))
    if not events:
        return 0.0

    events.sort()
    speech_time = 0.0
    overlap_time = 0.0
    active = 0
    previous_time = events[0][0]

    for time, delta in events:
        if active >= 1:
            speech_time += time - previous_time
        if active >= 2:
            overlap_time += time - previous_time
        active += delta
        previous_time = time

    return _ratio(overlap_time, speech_time)
