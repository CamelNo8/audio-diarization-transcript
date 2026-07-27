"""対応表 CSV の出力と、マッチング結果のサマリ表示。

対応表は「マッチした行」「使われなかった音声認識」「使われなかった台本」を
1つの時系列にまとめたもので、人が編集したうえで字幕生成
（:mod:`src.subtitle.exporter`）に渡す中間成果物にあたる。
"""

from __future__ import annotations

import csv
from typing import Any

from src.common.csv_io import CSV_ENCODING
from src.common.logging import get_logger
from src.common.timecode import seconds_to_time_str, time_str_to_seconds

logger = get_logger(__name__)

#: 対応表 CSV の列。順序も含めて後段（字幕生成・Web UI）が依存する。
CSV_HEADER = [
    "type",
    "script_start_id",
    "script_end_id",
    "script_speaker",
    "script_dialogue",
    "stt_start_id",
    "stt_end_id",
    "stt_speaker",
    "stt_text",
    "start_time",
    "end_time",
    "speaker",
    "subtitle_text",
]

#: 時系列の並べ替えにだけ使う一時キー。CSV には書き出さない。
_SORT_KEY = "sort_key"


def export_results_to_csv(
    scripts: list[dict],
    stt: list[dict],
    matched_pairs: list[dict],
    matched_script_indices: set[int],
    matched_stt_indices: set[int],
    filename: str,
) -> None:
    """マッチング結果を時系列に並べ、未使用データと合わせて CSV に出力する。

    Args:
        scripts: 台本データ。
        stt: 音声認識データ。
        matched_pairs: マッチしたペアのリスト。
        matched_script_indices: 使用済みの台本 id。
        matched_stt_indices: 使用済みの音声認識 id。
        filename: 出力先の CSV パス。
    """
    logger.info(f"最終結果をCSVファイル '{filename}' に生成しています...")

    timed_rows = [_build_matched_row(match, stt) for match in matched_pairs]
    timed_rows += [
        _build_unmatched_stt_row(s) for s in stt if s["id"] not in matched_stt_indices
    ]
    timed_rows.sort(key=lambda row: row[_SORT_KEY])

    unmatched_script_rows = [
        _build_unmatched_script_row(s)
        for s in scripts
        if s["id"] not in matched_script_indices
    ]
    unmatched_script_rows.sort(key=lambda row: row["script_start_id"])

    merged_rows = _merge_rows(timed_rows, unmatched_script_rows)
    logger.info("Unmatched_Script の時系列を補完しています...")
    _fill_unmatched_script_times(merged_rows)
    _write_csv(filename, merged_rows)


def _build_matched_row(match: dict, stt: list[dict]) -> dict[str, Any]:
    """マッチしたペアを対応表の1行にする。"""
    script_ng, stt_ng = match["script_ng"], match["stt_ng"]
    first_stt = stt[stt_ng["original_ids"][0]]
    last_stt = stt[stt_ng["original_ids"][-1]]
    script_speaker = script_ng.get("speaker", "")
    stt_speaker = first_stt.get("stt_speaker", "")

    return {
        "type": "Matched",
        "script_start_id": script_ng["original_ids"][0],
        "script_end_id": script_ng["original_ids"][-1],
        "script_speaker": script_speaker,
        "script_dialogue": script_ng["text"],
        "stt_start_id": stt_ng["original_ids"][0],
        "stt_end_id": stt_ng["original_ids"][-1],
        "stt_text": stt_ng["text"],
        "stt_speaker": stt_speaker,
        "start_time": first_stt["start"],
        "end_time": last_stt["end"],
        # 台本に話者がない場合のみ音声認識の話者で補う
        "speaker": script_speaker if script_speaker else stt_speaker,
        "subtitle_text": script_ng["text"],
        _SORT_KEY: time_str_to_seconds(first_stt["start"]),
    }


def _build_unmatched_stt_row(s: dict) -> dict[str, Any]:
    """どの台詞にも対応しなかった音声認識を対応表の1行にする。"""
    stt_speaker = s.get("stt_speaker", "")
    return {
        "type": "Unmatched_STT",
        "script_start_id": "",
        "script_end_id": "",
        "script_speaker": "",
        "script_dialogue": "",
        "stt_start_id": s["id"],
        "stt_end_id": s["id"],
        "stt_text": s["text"],
        "stt_speaker": stt_speaker,
        "start_time": s["start"],
        "end_time": s["end"],
        # 台本がないため字幕は音声認識を使う → 話者も音声認識のものを使う
        "speaker": stt_speaker,
        "subtitle_text": s["text"],
        _SORT_KEY: time_str_to_seconds(s["start"]),
    }


def _build_unmatched_script_row(s: dict) -> dict[str, Any]:
    """どの音声にも対応しなかった台詞を対応表の1行にする（時刻は後で補完）。"""
    script_speaker = s.get("speaker", "")
    script_dialogue = s.get("dialogue", "")
    return {
        "type": "Unmatched_Script",
        "script_start_id": s["id"],
        "script_end_id": s["id"],
        "script_speaker": script_speaker,
        "script_dialogue": script_dialogue,
        "stt_start_id": "",
        "stt_end_id": "",
        "stt_text": "",
        "stt_speaker": "",
        "start_time": "",
        "end_time": "",
        "speaker": script_speaker,
        "subtitle_text": script_dialogue,
    }


def _merge_rows(
    timed_rows: list[dict], unmatched_script_rows: list[dict]
) -> list[dict[str, Any]]:
    """時刻を持つ行の列に、未使用の台詞を台本の順序どおり差し込む。

    未使用の台詞は時刻を持たないため、次に現れるマッチ行の台本 id より小さい
    ものを、その直前に挿入していく。残りは末尾にまとめる。
    """
    merged: list[dict[str, Any]] = []
    unmatched_idx = 0

    for row in timed_rows:
        if row["type"] == "Matched":
            while (
                unmatched_idx < len(unmatched_script_rows)
                and unmatched_script_rows[unmatched_idx]["script_start_id"]
                < row["script_start_id"]
            ):
                merged.append(unmatched_script_rows[unmatched_idx])
                unmatched_idx += 1
        merged.append(row)

    merged.extend(unmatched_script_rows[unmatched_idx:])
    return merged


def _fill_unmatched_script_times(rows: list[dict]) -> None:
    """未使用の台詞に、前後の行の空き時間を等分した時刻を割り当てる。"""
    i = 0
    while i < len(rows):
        if rows[i]["type"] != "Unmatched_Script":
            i += 1
            continue
        block_end = _find_block_end(rows, i)
        prev_sec, next_sec = _surrounding_seconds(rows, i, block_end)
        _assign_block_times(rows, i, block_end, prev_sec, next_sec)
        i = block_end + 1


def _find_block_end(rows: list[dict], block_start: int) -> int:
    """未使用の台詞が連続する範囲の末尾インデックスを返す。"""
    block_end = block_start
    while (
        block_end + 1 < len(rows) and rows[block_end + 1]["type"] == "Unmatched_Script"
    ):
        block_end += 1
    return block_end


def _surrounding_seconds(
    rows: list[dict], block_start: int, block_end: int
) -> tuple[float, float]:
    """連続ブロックの直前の終了時刻と直後の開始時刻を秒で返す。

    直前が無ければ 0 秒、直後が無ければ直前と同じ（期間ゼロ）とみなす。
    """
    prev_sec = 0.0
    if block_start > 0:
        prev_row = rows[block_start - 1]
        if prev_row.get("end_time"):
            prev_sec = time_str_to_seconds(prev_row["end_time"])
        elif prev_row.get("start_time"):
            prev_sec = time_str_to_seconds(prev_row["start_time"])

    next_sec = prev_sec
    if block_end + 1 < len(rows) and rows[block_end + 1].get("start_time"):
        next_sec = time_str_to_seconds(rows[block_end + 1]["start_time"])
    return prev_sec, next_sec


def _assign_block_times(
    rows: list[dict],
    block_start: int,
    block_end: int,
    prev_sec: float,
    next_sec: float,
) -> None:
    """ブロック内の各行に、空き時間を等分した開始・終了時刻を書き込む。"""
    block_size = block_end - block_start + 1
    duration_sec = max(0.0, next_sec - prev_sec)

    if duration_sec <= 0:
        # 空きがない（直後がすぐ来る、または末尾）ので期間ゼロで埋める
        prev_str = seconds_to_time_str(prev_sec)
        next_str = seconds_to_time_str(next_sec)
        for j in range(block_start, block_end + 1):
            rows[j]["start_time"] = prev_str
            rows[j]["end_time"] = next_str
        return

    chunk_duration_sec = duration_sec / block_size
    current_sec = prev_sec
    for j in range(block_start, block_end + 1):
        rows[j]["start_time"] = seconds_to_time_str(current_sec)
        current_sec += chunk_duration_sec
        # 最後の要素はきっちり next_sec に合わせる（浮動小数点誤差の吸収）
        rows[j]["end_time"] = seconds_to_time_str(
            next_sec if j == block_end else current_sec
        )


def _write_csv(filename: str, rows: list[dict]) -> None:
    """対応表を CSV として書き出す。書き込みに失敗してもログのみで続行する。"""
    try:
        with open(filename, "w", newline="", encoding=CSV_ENCODING) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in rows:
                row.pop(_SORT_KEY, None)
                writer.writerow(row)
        logger.info(f"CSVファイル '{filename}' の生成が完了しました。")
    except IOError as e:
        logger.error(f"エラー: CSVファイルの書き込みに失敗しました: {e}")


def display_summary(
    scripts: list[dict],
    stt: list[dict],
    matched_pairs: list[dict],
    matched_script_indices: set[int],
    matched_stt_indices: set[int],
) -> None:
    """マッチングの詳細と統計情報をログに出力する。

    Args:
        scripts: 台本データ。
        stt: 音声認識データ。
        matched_pairs: マッチしたペアのリスト。
        matched_script_indices: 使用済みの台本 id。
        matched_stt_indices: 使用済みの音声認識 id。
    """
    _log_match_details(matched_pairs, stt)
    _log_unmatched_scripts(scripts, matched_script_indices)
    _log_unmatched_stt(stt, matched_stt_indices)
    _log_statistics(
        scripts, stt, matched_pairs, matched_script_indices, matched_stt_indices
    )


def _format_id_range(original_ids: list[int]) -> str:
    """元 id のリストを ``No.3`` / ``No.3-5`` の形に整える。"""
    if len(original_ids) == 1:
        return f"No.{original_ids[0]}"
    return f"No.{original_ids[0]}-{original_ids[-1]}"


def _log_match_details(matched_pairs: list[dict], stt: list[dict]) -> None:
    """マッチした組を1件ずつログに出す。"""
    logger.info("--- マッチング詳細 ---")
    for i, match in enumerate(matched_pairs, 1):
        script_ng, stt_ng = match["script_ng"], match["stt_ng"]
        logger.info(
            f"◆ マッチ {i}: 台本 {_format_id_range(script_ng['original_ids'])} "
            f"({script_ng['n']}-gram) ⇔ 音声認識 "
            f"{_format_id_range(stt_ng['original_ids'])} ({stt_ng['n']}-gram)"
        )
        logger.info(
            f"  [時刻]     : {stt[stt_ng['original_ids'][0]]['start']} "
            f"--> {stt[stt_ng['original_ids'][-1]]['end']}"
        )
        logger.info(f"  [台本]     : {script_ng['text']}")
        logger.info(f"  [音声認識] : {stt_ng['text']}")
        logger.info(f"  [類似度]   : {match['similarity']:.4f}\n" + "-" * 80)


def _log_unmatched_scripts(
    scripts: list[dict], matched_script_indices: set[int]
) -> None:
    """使われなかった台詞をログに出す。"""
    logger.info("--- 未使用の台本 ---")
    unmatched = [s for s in scripts if s["id"] not in matched_script_indices]
    for s in unmatched:
        logger.info(f"◆ 台本 No.{s['id']}: {s['speaker']}「{s['dialogue']}」")
    if not unmatched:
        logger.info("全ての台本要素が使用されました。")


def _log_unmatched_stt(stt: list[dict], matched_stt_indices: set[int]) -> None:
    """使われなかった音声認識をログに出す。"""
    logger.info("--- 未使用の音声認識 ---")
    unmatched = [s for s in stt if s["id"] not in matched_stt_indices]
    for s in unmatched:
        logger.info(f"◆ 音声認識 No.{s['id']} ({s['start']}): {s['text']}")
    if not unmatched:
        logger.info("全ての音声認識要素が使用されました。")


def _log_statistics(
    scripts: list[dict],
    stt: list[dict],
    matched_pairs: list[dict],
    matched_script_indices: set[int],
    matched_stt_indices: set[int],
) -> None:
    """使用率の統計をログに出す。"""
    logger.info("--- 統計情報 ---")
    logger.info(f"マッチング数: {len(matched_pairs)}組")
    logger.info(
        f"使用された台本要素: {len(matched_script_indices)}/{len(scripts)} "
        f"({len(matched_script_indices) / len(scripts) * 100:.1f}%)"
    )
    logger.info(
        f"使用された音声認識要素: {len(matched_stt_indices)}/{len(stt)} "
        f"({len(matched_stt_indices) / len(stt) * 100:.1f}%)"
    )
