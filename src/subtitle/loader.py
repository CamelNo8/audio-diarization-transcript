"""台本 CSV と音声認識 SRT の読み込み。

読み込みに失敗しても呼び出し元を止めないよう、エラーはログに出して空リストを返す。
"""

from __future__ import annotations

import re
from typing import Any

from src.common.csv_io import read_dict_rows
from src.common.logging import get_logger
from src.common.timecode import time_str_to_seconds
from src.subtitle.ngram import normalize_text

logger = get_logger(__name__)

#: 仮名・漢字・英数字。これが1文字も無いテキストは意味を持たないとみなす。
_MEANINGFUL_CHAR_RE = re.compile(r"[぀-ヿ一-鿿a-zA-Z0-9]")

#: SRT の本文先頭に埋め込まれた ``[話者名]``。
_LEADING_SPEAKER_RE = re.compile(r"^\s*\[([^\]]*)\]\s*(.*)$")

#: 台本の本文から取り除く注記。
_SCRIPT_ANNOTATION = "心の声"

#: 意味のあるテキストとみなす正規化後の最小文字数。
_MIN_MEANINGFUL_LEN = 2


def load_scripts_from_csv(filename: str) -> list[dict[str, Any]]:
    """台本 CSV から台詞行だけを読み込む。

    Args:
        filename: 台本 CSV のパス。``type`` / ``speaker`` / ``contents`` 列を持つ。

    Returns:
        ``id`` / ``speaker`` / ``dialogue`` を持つ辞書のリスト。``id`` は 0 起点の連番。
        読み込みに失敗した場合は空リスト。
    """
    scripts: list[dict[str, Any]] = []
    try:
        for row in read_dict_rows(filename):
            if row.get("type") != "dialogue":
                continue
            scripts.append(
                {
                    "id": len(scripts),
                    "speaker": row.get("speaker", ""),
                    "dialogue": row.get("contents", "").replace(_SCRIPT_ANNOTATION, ""),
                }
            )
        logger.info(
            f"'{filename}' から {len(scripts)} 件の台詞データを読み込みました。"
        )
    except FileNotFoundError:
        logger.error(f"エラー: ファイル '{filename}' が見つかりません。")
    except Exception as e:
        logger.error(f"エラー: {filename} の読み込み中にエラーが発生しました: {e}")
    return scripts


def load_stt_from_srt(filename: str) -> list[dict[str, Any]]:
    """音声認識 SRT を読み込む。

    Whisper の幻覚に由来する不正なセグメント（``end <= start`` や意味のない
    反復テキスト）は破棄する。

    Args:
        filename: SRT のパス。

    Returns:
        ``id`` / ``start`` / ``end`` / ``text`` / ``stt_speaker`` を持つ辞書のリスト。
        ``id`` は 0 起点の連番。読み込みに失敗した場合は空リスト。
    """
    stt: list[dict[str, Any]] = []
    skipped_time = 0
    skipped_noise = 0
    try:
        with open(filename, encoding="utf-8") as srtfile:
            for block in srtfile.read().strip().split("\n\n"):
                parsed = _parse_srt_block(block)
                if parsed is None:
                    continue
                start_str, end_str, speaker, text = parsed
                if time_str_to_seconds(end_str) <= time_str_to_seconds(start_str):
                    skipped_time += 1
                    continue
                if not _is_meaningful_stt_text(text):
                    skipped_noise += 1
                    continue
                stt.append(
                    {
                        "id": len(stt),
                        "start": start_str,
                        "end": end_str,
                        "text": text,
                        "stt_speaker": speaker,
                    }
                )
        logger.info(
            f"'{filename}' から {len(stt)} 件の字幕データを読み込みました。"
            f"（時刻不正で除外: {skipped_time}, 幻覚テキストで除外: {skipped_noise}）"
        )
    except FileNotFoundError:
        logger.error(f"エラー: ファイル '{filename}' が見つかりません。")
    except Exception as e:
        logger.error(f"エラー: {filename} の読み込み中にエラーが発生しました: {e}")
    return stt


def _parse_srt_block(block: str) -> tuple[str, str, str, str] | None:
    """SRT の1ブロックを (開始, 終了, 話者, 本文) に分解する。

    Returns:
        分解結果。時刻行を持たないブロックは ``None``。
    """
    lines = block.strip().split("\n")
    if len(lines) < 3 or "-->" not in lines[1]:
        return None
    start_str, end_str = (t.strip() for t in lines[1].split("-->"))
    text = " ".join(lines[2:])

    # SRT 生成時に "[話者名] 本文" の形で埋め込まれているため分離する
    speaker = ""
    matched = _LEADING_SPEAKER_RE.match(text)
    if matched:
        speaker = matched.group(1).strip()
        text = matched.group(2)
    return start_str, end_str, speaker, text


def _is_meaningful_stt_text(text: str, min_len: int = _MIN_MEANINGFUL_LEN) -> bool:
    """Whisper の幻覚（``!``、``caus caus caus`` 等）でないかを判定する。

    Args:
        text: 判定するテキスト。
        min_len: 正規化後に必要な最小文字数。

    Returns:
        意味のあるテキストなら True。
    """
    norm = normalize_text(text)
    if len(norm) < min_len:
        return False
    if not _MEANINGFUL_CHAR_RE.search(norm):
        return False
    tokens = norm.split()
    # 全文がひとつのトークンの反復（例: 'caus caus caus'）は幻覚とみなす
    return not (len(tokens) >= 2 and len(set(tokens)) == 1)
