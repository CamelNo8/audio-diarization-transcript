"""時刻文字列と秒の相互変換。

本プロジェクトでは2種類の時刻表記を扱う。

- 文字起こしCSV: ``HH:MM:SS:mmm``（ミリ秒をコロンで区切る）
- SRT字幕:       ``HH:MM:SS,mmm``（ミリ秒をカンマで区切る）
"""

from __future__ import annotations

import re

#: 文字起こしCSVの時刻（ミリ秒区切りはコロン / ピリオド / カンマを許容）
_COLON_MS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[:.,](\d{1,3})$")

_MS_PER_SECOND = 1000
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def format_time(seconds: float) -> str:
    """秒数を文字起こしCSVの時刻表記 ``HH:MM:SS:mmm`` に変換する。

    Args:
        seconds: 変換する秒数。負値は 0 秒として扱う。

    Returns:
        ``HH:MM:SS:mmm`` 形式の文字列。ミリ秒未満は四捨五入する。
    """
    if seconds < 0:
        return "00:00:00:000"
    total_ms = int(round(seconds * _MS_PER_SECOND))
    hours, remainder_ms = divmod(total_ms, _SECONDS_PER_HOUR * _MS_PER_SECOND)
    minutes, remainder_ms = divmod(remainder_ms, _SECONDS_PER_MINUTE * _MS_PER_SECOND)
    secs, millis = divmod(remainder_ms, _MS_PER_SECOND)
    return f"{hours:02}:{minutes:02}:{secs:02}:{millis:03}"


def colon_ms_to_comma_ms(time_str: str) -> str:
    """文字起こしCSVの時刻表記を SRT の時刻表記へ変換する。

    Args:
        time_str: ``HH:MM:SS:mmm`` 形式の文字列。空文字や既に
            ``HH:MM:SS,mmm`` 形式の文字列も受け付ける。

    Returns:
        ``HH:MM:SS,mmm`` 形式の文字列。解釈できない書式は入力をそのまま返す。

    Note:
        **ミリ秒部は小数ではなく整数として解釈する。** 3桁未満の場合は
        前ゼロ埋めになるため ``"00:00:01:5"`` は 500ms ではなく 005ms を指す。
        :func:`format_time` が出力する CSV は常に3桁のため実害はない。
        これは仕様であり、変更するとリファクタリング前の出力と食い違う。
    """
    if not time_str:
        return ""
    if "," in time_str:
        return time_str
    matched = _COLON_MS_RE.match(time_str)
    if matched:
        hours, minutes, seconds, millis = (int(v) for v in matched.groups())
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
    # 想定外に区切りが多い場合は、最後の区切りだけをミリ秒とみなす
    if time_str.count(":") >= 3:
        head, _, tail = time_str.rpartition(":")
        return f"{head},{tail}"
    return time_str


def time_str_to_seconds(time_str: str) -> float:
    """SRT の時刻表記 ``HH:MM:SS,mmm`` を秒へ変換する。

    Args:
        time_str: 時刻文字列。ミリ秒の区切りはカンマ・ピリオドのどちらでもよい。

    Returns:
        秒数。空文字や不正な書式の場合は 0.0（字幕の並べ替えを止めないため）。
    """
    try:
        parts = time_str.replace(",", ".").split(":")
        return (
            int(parts[0]) * _SECONDS_PER_HOUR
            + int(parts[1]) * _SECONDS_PER_MINUTE
            + float(parts[2])
        )
    except (ValueError, IndexError):
        return 0.0


def seconds_to_time_str(seconds: float) -> str:
    """秒を SRT の時刻表記 ``HH:MM:SS,mmm`` へ変換する。

    Args:
        seconds: 変換する秒数。負値は 0 秒として扱う。

    Returns:
        ``HH:MM:SS,mmm`` 形式の文字列。
    """
    if seconds < 0:
        seconds = 0
    try:
        millis = int((seconds * _MS_PER_SECOND) % _MS_PER_SECOND)
        total_seconds = int(seconds)
    except (ValueError, OverflowError):
        # NaN / inf を渡されても字幕生成を止めないための既存挙動を踏襲する
        return "00:00:00,000"
    secs = total_seconds % _SECONDS_PER_MINUTE
    total_minutes = total_seconds // _SECONDS_PER_MINUTE
    minutes = total_minutes % _SECONDS_PER_MINUTE
    hours = total_minutes // _SECONDS_PER_MINUTE
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
