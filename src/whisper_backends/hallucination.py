"""Whisper が生み出す「幻聴」テキストの判定。

無音や音楽だけの区間に対して、Whisper は学習データ由来の定型句
（「ご視聴ありがとうございました」など）や、同じ音の繰り返しを出すことがある。
これらは実際には誰も発話していないため、字幕に残ると邪魔になる。

判定はテキストだけを見る。文字起こしバックエンドは ``{start, end, text}`` しか
返しておらず、``no_speech_prob`` のような信頼度が手元に無いため。
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

from src.common.logging import get_logger
from src.config import DEFAULT_HALLUCINATION_PHRASES

logger = get_logger(__name__)

#: 追加のフレーズを列挙したファイルを指す環境変数。1行1フレーズ。
#: ``#`` で始まる行と空行は無視する。
PHRASES_FILE_ENV = "HALLUCINATION_PHRASES_FILE"

#: 反復とみなす繰り返し単位の最大文字数。これより長い単位の繰り返しは
#: 意味のある発話（「ありがとうありがとう」など）の可能性が高いので残す。
MAX_REPEAT_UNIT_LEN = 4

#: 反復とみなす最小の繰り返し回数。「そうそう」（2回）は相槌として残す。
MIN_REPEAT_COUNT = 3

#: 正規化で取り除く文字（句読点・記号・空白）。
_NOISE_CHARS = re.compile(r"[\s、。，．,\.!！?？…‥・「」『』（）\(\)\-―ー~〜:：;；]+")


def normalize(text: str) -> str:
    """比較用にテキストを正規化する。

    全角・半角を統一し、句読点や記号・空白を取り除く。
    「ご視聴ありがとうございました。」と「ご視聴ありがとうございました」を
    同じものとして扱うため。

    Note:
        長音符（``ー``）も落とすため、「あーあーあー」は「あああ」になる。
        反復判定はこの正規化後の文字列に対して行う。
    """
    normalized = unicodedata.normalize("NFKC", text)
    return _NOISE_CHARS.sub("", normalized)


def load_phrases() -> List[str]:
    """幻聴とみなす定型フレーズの一覧を返す。

    :data:`~src.config.DEFAULT_HALLUCINATION_PHRASES` に、環境変数
    :data:`PHRASES_FILE_ENV` が指すファイルの内容を追記したもの。
    素材ごとの調整をコード変更なしで行えるようにするため。

    Returns:
        フレーズの一覧（正規化前の表記）。ファイルが無い・読めない場合は既定のみ。
    """
    phrases = list(DEFAULT_HALLUCINATION_PHRASES)

    path_str = os.environ.get(PHRASES_FILE_ENV)
    if not path_str:
        return phrases

    path = Path(path_str)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning(f"{PHRASES_FILE_ENV} を読めませんでした ({path}): {e}")
        return phrases

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            phrases.append(stripped)
    return phrases


class HallucinationFilter:
    """文字起こしの行を順に見て、幻聴とみなせる行を報告する。

    直前の行との比較を行うため状態を持つ。**1回の文字起こしにつき1インスタンス**を
    作り、セグメントを先頭から順に渡すこと。

    Attributes:
        phrases: 幻聴とみなす定型フレーズ（正規化済み）。
    """

    def __init__(self, phrases: Optional[Iterable[str]] = None) -> None:
        """
        Args:
            phrases: 既定のフレーズに加えて幻聴とみなす表記。
                省略すると :func:`load_phrases` の結果だけを使う。
        """
        すべて = list(load_phrases()) + list(phrases or [])
        self.phrases = {normalize(p) for p in すべて if normalize(p)}
        self._previous: Optional[str] = None

    def reason_to_drop(self, text: str) -> Optional[str]:
        """行を落とすべきなら理由を、残すべきなら ``None`` を返す。

        採用した行だけを「直前の行」として記憶する。幻聴を挟んでも、
        その前の本物の行との比較が続くようにするため。

        Args:
            text: セグメントの本文。

        Returns:
            ログに出す判定理由。落とす必要がなければ ``None``。
        """
        normalized = normalize(text)
        if not normalized:
            return None

        if normalized in self.phrases:
            return "定型フレーズ"
        repeat = self._repeated_unit(normalized)
        if repeat is not None:
            return f"同一文字列の反復（{repeat!r}）"
        if normalized == self._previous:
            return "直前の行と同一"

        self._previous = normalized
        return None

    @staticmethod
    def _repeated_unit(normalized: str) -> Optional[str]:
        """短い単位の繰り返しだけで構成されていれば、その単位を返す。

        Args:
            normalized: :func:`normalize` 済みのテキスト。

        Returns:
            繰り返しの単位。該当しなければ ``None``。
        """
        for length in range(1, MAX_REPEAT_UNIT_LEN + 1):
            if len(normalized) < length * MIN_REPEAT_COUNT:
                break
            if len(normalized) % length != 0:
                continue
            unit = normalized[:length]
            if unit * (len(normalized) // length) == normalized:
                return unit
        return None
