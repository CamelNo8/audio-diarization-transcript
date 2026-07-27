"""テキストの正規化と N-gram（連続する台詞・字幕のまとまり）の生成。

台本と音声認識では文の区切り方が一致しないため、連続する要素を最大
``max_n`` 件までつないだ N-gram を作り、その単位で対応付けを探す。
"""

from __future__ import annotations

import re
from typing import Any

from src.common.timecode import time_str_to_seconds

#: 比較の妨げになる記号。空白に置き換えて無視する。
_PUNCTUATION_RE = re.compile(r"[、,。．.]")

#: SRT に埋め込まれた ``[話者名]`` などの注記。
_BRACKETED_RE = re.compile(r"\[.*?\]")

#: 連続する空白（全角空白・改行・タブを含む）。
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """比較しやすいようにテキストのゆらぎを取り除く。

    括弧・角括弧の注記と句読点を空白に置き換え、連続する空白を1つにまとめる。

    Args:
        s: 正規化する文字列。

    Returns:
        正規化後の文字列。中身が記号だけだった場合は空文字。
    """
    s = s.replace("（", " ").replace("）", " ")
    s = _BRACKETED_RE.sub(" ", s)
    s = _PUNCTUATION_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def create_ngrams(
    data_list: list[dict],
    text_key: str,
    max_n: int,
    has_time: bool = False,
) -> list[dict[str, Any]]:
    """連続する要素をつないだ N-gram を生成する。

    ``speaker`` を持つデータ（台本）では、話者をまたぐ N-gram は作らない。

    Args:
        data_list: ``id`` と ``text_key`` を持つ辞書のリスト。
        text_key: 本文が入っているキー名。
            台本なら ``dialogue``、音声認識なら ``text``。
        max_n: つなぐ要素数の上限。0 以下なら何も生成しない。
        has_time: ``start`` / ``end`` から秒の時刻を付けるか。

    Returns:
        N-gram の辞書のリスト。``id`` は生成順の連番で、
        話者違いでスキップした分は詰める。
    """
    ngrams: list[dict[str, Any]] = []
    for n in range(1, max_n + 1):
        for i in range(len(data_list) - n + 1):
            chunk = data_list[i : i + n]
            if _spans_multiple_speakers(chunk):
                continue
            ngrams.append(_build_ngram(chunk, len(ngrams), text_key, i, n, has_time))
    return ngrams


def _spans_multiple_speakers(chunk: list[dict]) -> bool:
    """まとまりの中に複数の話者が混ざっているか。話者を持たないデータは常に False。"""
    if "speaker" not in chunk[0]:
        return False
    return not all(item["speaker"] == chunk[0]["speaker"] for item in chunk)


def _build_ngram(
    chunk: list[dict],
    ngram_id: int,
    text_key: str,
    start_index: int,
    n: int,
    has_time: bool,
) -> dict[str, Any]:
    """1つの N-gram を組み立てる。"""
    combined_text = " ".join(item[text_key] for item in chunk)
    ngram: dict[str, Any] = {
        "id": ngram_id,
        "text": combined_text,
        "normalized_text": normalize_text(combined_text),
        "start_index": start_index,
        "end_index": start_index + n - 1,
        "original_ids": [item["id"] for item in chunk],
        "n": n,
    }
    if "speaker" in chunk[0]:
        ngram["speaker"] = chunk[0]["speaker"]
    if has_time:
        ngram["start_time"] = time_str_to_seconds(chunk[0]["start"])
        ngram["end_time"] = time_str_to_seconds(chunk[-1]["end"])
    return ngram
