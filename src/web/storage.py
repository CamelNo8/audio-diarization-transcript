"""作業ディレクトリ（``temp/``）へのファイル保存。

アップロード音声・変換した台本・生成した字幕はすべてここに置く。
保存先を広げない（規約7.4）ため、パスの組み立ては :func:`temp_path` に
一本化し、フォームから来た名前は先に
:func:`src.common.filenames.safe_output_name` で正規化してから渡す。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

from src.config import TEMP_DIR as _CONFIGURED_TEMP_DIR

#: 作業ディレクトリ。テストではこのモジュール属性を差し替える。
TEMP_DIR = _CONFIGURED_TEMP_DIR


def temp_path(name: str) -> Path:
    """作業ディレクトリ直下のパスを返す。

    Args:
        name: 検証済みのファイル名。パス区切りを含めてはならない。
    """
    return TEMP_DIR / name


def save_upload(upload: UploadFile, name: str) -> Path:
    """アップロードされたファイルを作業ディレクトリへ保存する。

    Args:
        upload: FastAPI が受け取ったアップロード。
        name: 保存名。呼び出し側で ``Path(...).name`` などに通しておく。

    Returns:
        保存先のパス。
    """
    dest = temp_path(name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest
