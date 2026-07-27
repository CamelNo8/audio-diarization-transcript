"""CSV の読み書きの共通処理。

本プロジェクトの CSV は Excel で直接開けるよう、入出力ともに BOM 付き
UTF-8（``utf-8-sig``）で統一する。
"""

from __future__ import annotations

import csv
from pathlib import Path

#: Excel が文字化けせず開けるようにするためのエンコーディング
CSV_ENCODING = "utf-8-sig"


def read_dict_rows(csv_path: Path) -> list[dict[str, str]]:
    """CSV を1行1辞書として読み込む（1行目をヘッダとして扱う）。

    Args:
        csv_path: 読み込む CSV のパス。

    Returns:
        ヘッダ名をキーとした辞書のリスト。データ行が無ければ空リスト。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        OSError: 読み込みに失敗した場合。
    """
    with open(csv_path, encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def read_rows(csv_path: Path) -> list[list[str]]:
    """CSV をヘッダ行込みの二次元リストとして読み込む。

    Args:
        csv_path: 読み込む CSV のパス。

    Returns:
        1行目にヘッダを含む行のリスト。空ファイルなら空リスト。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        OSError: 読み込みに失敗した場合。
    """
    with open(csv_path, encoding=CSV_ENCODING, newline="") as f:
        return list(csv.reader(f))


def write_rows(
    csv_path: Path, rows: list[list[str]], *, quoting: int = csv.QUOTE_ALL
) -> None:
    """二次元リストを CSV として書き出す（既存ファイルは上書き）。

    Args:
        csv_path: 出力先。
        rows: 1行目にヘッダを含む行のリスト。
        quoting: csv モジュールのクォート方針。既定は全フィールドを引用符で囲む。

    Raises:
        OSError: 書き込みに失敗した場合。
    """
    with open(csv_path, "w", encoding=CSV_ENCODING, newline="") as f:
        csv.writer(f, quoting=quoting).writerows(rows)
