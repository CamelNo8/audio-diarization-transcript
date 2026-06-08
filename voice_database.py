"""声紋データベース管理モジュール。

voice_databases/<DB名>/<話者名>.<ext> の構造で永続管理する。
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mov", ".ogg", ".opus", ".aac", ".wma",
}

_INVALID_NAME_CHARS = set('/\\:*?"<>|')


def get_root() -> Path:
    """声紋DB のルートディレクトリを返す（環境変数で上書き可）。"""
    env = os.getenv("VOICE_DB_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "voice_databases"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_name(raw: str) -> Optional[str]:
    """DB名 / 話者名として使える文字列に整える。NG なら None。"""
    name = (raw or "").strip()
    if not name:
        return None
    if name in (".", ".."):
        return None
    if any(c in _INVALID_NAME_CHARS for c in name):
        return None
    return name


def list_databases() -> List[Dict]:
    """DB 一覧をメタ情報付きで返す。"""
    root = get_root()
    result = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        speakers = list_speakers(entry.name)
        result.append({
            "name": entry.name,
            "speaker_count": len(speakers),
            "path": str(entry),
        })
    return result


def database_dir(name: str) -> Path:
    """DB ディレクトリパスを返す（存在しなければ FileNotFoundError）。"""
    safe = sanitize_name(name)
    if safe is None:
        raise ValueError(f"無効なデータベース名: {name!r}")
    path = get_root() / safe
    if not path.is_dir():
        raise FileNotFoundError(f"データベースが存在しません: {safe}")
    return path


def create_database(name: str) -> Path:
    """新規DB（ディレクトリ）を作成して返す。既存なら ValueError。"""
    safe = sanitize_name(name)
    if safe is None:
        raise ValueError(f"無効なデータベース名: {name!r}")
    path = get_root() / safe
    if path.exists():
        raise ValueError(f"データベースは既に存在します: {safe}")
    path.mkdir(parents=True)
    return path


def list_speakers(db_name: str) -> List[Dict]:
    """DB 内の話者ファイル一覧を返す。"""
    path = database_dir(db_name)
    speakers = []
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        speakers.append({
            "filename": f.name,
            "speaker_name": f.stem,
            "size_bytes": f.stat().st_size,
        })
    return speakers


def speaker_path(db_name: str, filename: str) -> Path:
    """DB 内の話者ファイルのパスを返す（存在チェック付き）。"""
    safe_filename = Path(filename).name  # path traversal 防止
    if safe_filename != filename:
        raise ValueError(f"無効なファイル名: {filename!r}")
    path = database_dir(db_name) / safe_filename
    if not path.is_file():
        raise FileNotFoundError(f"話者ファイルが存在しません: {db_name}/{filename}")
    return path


def delete_speaker(db_name: str, filename: str) -> None:
    """DB 内の話者ファイルを削除する。"""
    path = speaker_path(db_name, filename)
    path.unlink()


def add_speaker_file(db_name: str, src_path: Path, dest_filename: Optional[str] = None) -> Path:
    """src_path を DB にコピーして登録する。

    dest_filename を省略すると src_path.name を使用。
    既存ファイルがある場合は上書きする。
    """
    dst_dir = database_dir(db_name)
    if dest_filename is None:
        dest_filename = src_path.name
    safe_filename = Path(dest_filename).name
    if safe_filename != dest_filename:
        raise ValueError(f"無効なファイル名: {dest_filename!r}")
    if Path(safe_filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"対応していない拡張子: {safe_filename} "
            f"(対応: {sorted(SUPPORTED_AUDIO_EXTENSIONS)})"
        )
    dst = dst_dir / safe_filename
    shutil.copyfile(src_path, dst)
    return dst
