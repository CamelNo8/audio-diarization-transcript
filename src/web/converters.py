"""Web UI が受け取ったファイルの形式変換。

Step 1 の文字起こし CSV を利用者向けの SRT にする処理と、Step 2 で
プレーンテキストの台本を台本 CSV にする処理を持つ。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from src.common.csv_io import read_dict_rows
from src.common.timecode import colon_ms_to_comma_ms

#: ``話者:本文`` 形式の行。話者名は行頭の記号を除いた 31 文字未満に限る。
_SPEAKER_LINE_RE = re.compile(r"^\s*([^\s:：（(][^:：]{0,30}?)\s*[:：]\s*(.+)$")

#: 丸括弧だけで囲まれた行（ト書き・場面説明）。
_SCENE_PAREN_RE = re.compile(r"^\s*[（(](.+)[)）]\s*$")


def csv_to_srt_with_speaker(csv_path: Path, srt_path: Path) -> int:
    """Step 1 の文字起こし CSV から speaker prefix 付き SRT を生成する。

    本文は ``[speaker] text`` 形式。speaker が空 / ``Unknown`` / ``Unknown_*``
    の場合もそのまま prefix として書き出す（利用者が視認できるように）。

    Args:
        csv_path: 文字起こし CSV。存在しなければ何もしない。
        srt_path: 生成先の SRT。

    Returns:
        書き出した字幕ブロック数。
    """
    if not csv_path.exists():
        return 0
    blocks = []
    idx = 1
    for row in read_dict_rows(csv_path):
        start = colon_ms_to_comma_ms((row.get("start") or "").strip())
        end = colon_ms_to_comma_ms((row.get("end") or "").strip())
        body = (row.get("text") or "").strip()
        speaker = (row.get("speaker") or "").strip()
        if not start or not end or not body:
            continue
        line = f"[{speaker}] {body}" if speaker else body
        blocks.append(f"{idx}\n{start} --> {end}\n{line}\n")
        idx += 1
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return idx - 1


def txt_to_script_csv_bytes(txt_bytes: bytes) -> bytes:
    """プレーンテキストを台本 CSV（id,scene_id,type,speaker,contents）に変換する。

    各非空行を 1 行に変換する:

    - ``# <内容>`` または ``（...）`` ``(...)`` → type=scene, speaker=空
    - ``<話者>:<本文>`` / ``<話者>：<本文>`` → type=dialogue, speaker=話者
    - それ以外 → type=dialogue, speaker=空, contents=行

    id は 1 から連番、scene_id は空欄。

    Args:
        txt_bytes: UTF-8（BOM 付き可）のテキスト。

    Returns:
        BOM 付き UTF-8 の CSV バイト列。
    """
    text = txt_bytes.decode("utf-8-sig")
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id", "scene_id", "type", "speaker", "contents"])
    idx = 1
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        writer.writerow([idx, *_classify_script_line(line)])
        idx += 1
    return buf.getvalue().encode("utf-8-sig")


def _classify_script_line(line: str) -> tuple[str, str, str, str]:
    """台本 1 行を ``(scene_id, type, speaker, contents)`` に振り分ける。"""
    if line.startswith("#"):
        return "", "scene", "", line.lstrip("#").strip()
    m_scene = _SCENE_PAREN_RE.match(line)
    if m_scene:
        return "", "scene", "", m_scene.group(1).strip()
    m_speaker = _SPEAKER_LINE_RE.match(line)
    if m_speaker:
        return "", "dialogue", m_speaker.group(1).strip(), m_speaker.group(2).strip()
    return "", "dialogue", "", line
