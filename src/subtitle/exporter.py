"""編集済みの対応表 CSV から字幕 SRT を生成する。

対応表（:mod:`src.subtitle.report` が出力し、人が編集したもの）の
``start_time`` / ``end_time`` / ``speaker`` / ``subtitle_text`` の4列だけを使う。
"""

from __future__ import annotations

import argparse
import sys

from src.common.csv_io import read_dict_rows
from src.common.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: 字幕生成に必要な列。
REQUIRED_COLUMNS = ["start_time", "end_time", "speaker", "subtitle_text"]

#: このうち1つでも空なら、その行は字幕にできない（``speaker`` は空でもよい）。
_ESSENTIAL_COLUMNS = ["start_time", "end_time", "subtitle_text"]


def load_subtitle_data(input_csv_path: str) -> list[dict[str, str]]:
    """対応表 CSV から字幕生成に必要なデータを抽出する。

    Args:
        input_csv_path: 入力する編集済み CSV のパス。

    Returns:
        :data:`REQUIRED_COLUMNS` の4キーを持つ辞書のリスト。
        読み込みに失敗した場合は空リスト。
    """
    subtitle_data = []
    try:
        for i, row in enumerate(read_dict_rows(input_csv_path)):
            if not _is_usable_row(row, i + 1):
                continue
            subtitle_data.append({col: row[col] for col in REQUIRED_COLUMNS})
    except FileNotFoundError:
        logger.error(f"エラー: 入力ファイル '{input_csv_path}' が見つかりません。")
        return []
    except Exception as e:
        logger.error(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        return []

    logger.info(
        f"'{input_csv_path}' から {len(subtitle_data)} 件の字幕データを読み込みました。"
    )
    return subtitle_data


def _is_usable_row(row: dict[str, str], line_no: int) -> bool:
    """字幕にできる行かを判定し、できない場合は理由を警告に出す。

    ``speaker`` の空欄は仕様どおりの正常系なので警告しない
    （対応表では普通に発生するため、警告すると本当の警告が埋もれる）。
    """
    if not all(col in row for col in REQUIRED_COLUMNS):
        logger.warning(
            f"警告: {line_no}行目に必要な列が不足しています。スキップします。"
        )
        return False
    if not all(row[col] for col in _ESSENTIAL_COLUMNS):
        logger.warning(
            f"警告: {line_no}行目に必須データ（時間またはテキスト）がありません。"
            "スキップします。"
        )
        return False
    return True


def format_subtitle_text(
    current_speaker: str, subtitle_text: str, previous_speaker: str | None
) -> str:
    """話者名の表示ルールに基づき、SRT に表示するテキストを組み立てる。

    直前と話者が変わったとき（および最初の字幕）だけ ``(話者名)`` を前置する。

    Args:
        current_speaker: 現在の字幕の話者。空欄なら話者名は付けない。
        subtitle_text: 現在の字幕テキスト。
        previous_speaker: 直前の字幕の話者。無い場合は ``None``。

    Returns:
        SRT に表示するテキスト。
    """
    if not current_speaker:
        return subtitle_text
    if previous_speaker and current_speaker == previous_speaker:
        return subtitle_text
    return f"({current_speaker}){subtitle_text}"


def generate_srt_content(subtitle_data: list[dict[str, str]]) -> str:
    """字幕データのリストから SRT ファイル全体の内容を生成する。

    Args:
        subtitle_data: :func:`load_subtitle_data` が返した形のリスト。

    Returns:
        SRT 形式の文字列。データが空なら空文字。
    """
    srt_blocks = []
    previous_speaker: str | None = None

    for subtitle_index, data in enumerate(subtitle_data, 1):
        speaker = data["speaker"].strip()
        timestamp_line = f"{data['start_time']} --> {data['end_time']}"
        text_to_display = format_subtitle_text(
            speaker, data["subtitle_text"], previous_speaker
        )
        # 話者名が空欄の行の次は必ず話者名を表示したいので、比較対象を捨てる
        previous_speaker = speaker if speaker else None
        srt_blocks.append(f"{subtitle_index}\n{timestamp_line}\n{text_to_display}\n")

    return "\n".join(srt_blocks)


def write_srt_file(output_srt_path: str, srt_content: str) -> None:
    """SRT の内容をファイルに書き込む。失敗してもログのみで続行する。

    Args:
        output_srt_path: 出力する SRT のパス。
        srt_content: 書き込む内容。
    """
    try:
        with open(output_srt_path, mode="w", encoding="utf-8") as f:
            f.write(srt_content)
        logger.info(f"SRTファイルを '{output_srt_path}' に正常に書き込みました。")
    except IOError as e:
        logger.error(f"エラー: ファイルの書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        logger.error(f"エラー: 予期せぬエラーが発生しました: {e}")


def main() -> int:
    """CLI エントリポイント。対応表 CSV から SRT を生成する。

    Returns:
        終了コード。データが読み込めなかった場合は 1。
    """
    parser = argparse.ArgumentParser(
        description="編集済みの対応表CSVファイルからSRT字幕ファイルを生成します。"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="入力する編集済みCSVファイルのパス (例: 対応表_edited.csv)",
    )
    parser.add_argument(
        "output_srt",
        type=str,
        help="出力するSRTファイルのパス (例: subtitles_exported.srt)",
    )

    configure_logging()
    args = parser.parse_args()

    subtitle_data = load_subtitle_data(args.input_csv)
    if not subtitle_data:
        logger.error("データが読み込めなかったため、処理を終了します。")
        return 1

    write_srt_file(args.output_srt, generate_srt_content(subtitle_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
