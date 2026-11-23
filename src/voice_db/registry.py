"""声紋データベース管理モジュール。

voice_databases/<DB名>/<話者名>.<ext> の構造で永続管理する。

短いクリップを複数まとめて1人として登録する場合は、繋ぐ前の素材を
``voice_databases/<DB名>/_clips/<話者名>/`` に残し、それらを連結したものを
``<話者名>.wav`` として置く。素材フォルダはディレクトリなので、話者ファイルの
一覧も照合対象の収集もファイルだけを見ており、素材が話者として読まれることはない。

削除は**ゴミ箱への退避**として実装している。確認ダイアログを押し間違えても
``voice_databases/.trash/`` から手で戻せるようにするため。溜まったら手で消す
運用で、自動削除はしない。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from src.common.audio import concat_audio, probe_duration_sec
from src.common.logging import get_logger
from src.config import (
    CLIPS_DIR_NAME,
    DEFAULT_VOICE_DB_ROOT,
    INVALID_NAME_CHARS,
)

logger = get_logger(__name__)

#: 削除したものの退避先（ルート直下）。ドット始まりなので DB 一覧には出ない。
TRASH_DIR_NAME = ".trash"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".mp4",
    ".mov",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
}


def get_root() -> Path:
    """声紋DB のルートディレクトリを返す（環境変数で上書き可）。"""
    env = os.getenv("VOICE_DB_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = DEFAULT_VOICE_DB_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_name(raw: str) -> Optional[str]:
    """DB名 / 話者名として使える文字列に整える。NG なら None。

    ドットで始まる名前は弾く。``.`` / ``..`` によるパストラバーサルに加えて、
    ゴミ箱（``.trash``）を DB として作成・参照・削除できてしまう経路も塞ぐため。
    """
    name = (raw or "").strip()
    if not name:
        return None
    if name.startswith("."):
        return None
    if any(c in INVALID_NAME_CHARS for c in name):
        return None
    return name


def list_databases() -> List[Dict]:
    """DB 一覧をメタ情報付きで返す。ドットで始まる名前は対象外。"""
    root = get_root()
    result = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        speakers = list_speakers(entry.name)
        result.append(
            {
                "name": entry.name,
                "speaker_count": len(speakers),
                "path": str(entry),
            }
        )
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


def trash_dir() -> Path:
    """削除したものの退避先を返す（無ければ作る）。"""
    path = get_root() / TRASH_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_to_trash(path: Path, label: str) -> Path:
    """``path`` をゴミ箱へ退避してその場所を返す。

    Args:
        path: 退避するファイルまたはディレクトリ。
        label: 退避先に付ける名前。日時を前置きするので、同じものを何度
            消しても区別できる。同秒に同名を消した場合は連番を足す。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = trash_dir() / f"{stamp}_{label}"
    suffix = 2
    while dest.exists():
        dest = trash_dir() / f"{stamp}_{label}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(dest))
    logger.info(f"ゴミ箱へ退避しました: {path} → {dest}")
    return dest


def delete_database(name: str) -> Path:
    """DB を中の話者ファイルごとゴミ箱へ退避する。

    Returns:
        退避先のパス。戻したいときはこれをディレクトリごと移動すればよい。
    """
    path = database_dir(name)
    return move_to_trash(path, path.name)


def list_speakers(db_name: str) -> List[Dict]:
    """DB 内の話者ファイル一覧を返す。

    登録音声が短すぎないかを画面で確かめられるよう、音声長と素材クリップの
    本数も一緒に返す。音声長は ffprobe で取るため、取得できなければ None。
    """
    path = database_dir(db_name)
    speakers = []
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        st = f.stat()
        speakers.append(
            {
                "filename": f.name,
                "speaker_name": f.stem,
                "size_bytes": st.st_size,
                "mtime": int(st.st_mtime),
                "duration_sec": probe_duration_sec(f),
                "clip_count": len(list_clips(db_name, f.stem)),
            }
        )
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


def delete_speaker(db_name: str, filename: str) -> Path:
    """DB 内の話者ファイルをゴミ箱へ退避する。素材フォルダも一緒に退避する。

    素材を残すと、同じ話者名で登録し直したときに消したはずの古いクリップが
    混ざってしまうため、話者ファイルと必ずセットで動かす。

    Returns:
        退避先のパス（話者ファイルのほう）。
    """
    path = speaker_path(db_name, filename)
    clips = clips_dir(db_name, Path(filename).stem, create=False)
    if clips.is_dir():
        move_to_trash(clips, f"{path.parent.name}_clips_{clips.name}")
    return move_to_trash(path, f"{path.parent.name}_{path.name}")


def rename_speaker(db_name: str, filename: str, new_speaker_name: str) -> Path:
    """話者ファイルをリネームして話者名（=ファイル名の拡張子なし部分）を変更する。

    元の拡張子は維持する。リネーム先が既存の場合は ValueError（上書き防止）。
    """
    src = speaker_path(db_name, filename)
    safe_name = sanitize_name(new_speaker_name)
    if safe_name is None:
        raise ValueError(f"無効な話者名: {new_speaker_name!r}")
    dst = src.with_name(safe_name + src.suffix)
    if dst == src:
        return src
    if dst.exists():
        raise ValueError(f"同名の話者が既に存在します: {dst.name}")

    # 素材フォルダは話者名で引くため、追従させないと繋ぎ直しが効かなくなる
    src_clips = clips_dir(db_name, Path(filename).stem, create=False)
    dst_clips = clips_dir(db_name, safe_name, create=False)
    if src_clips.is_dir() and dst_clips.exists():
        raise ValueError(f"同名の素材フォルダが既に存在します: {dst_clips.name}")

    src.rename(dst)
    if src_clips.is_dir():
        src_clips.rename(dst_clips)
    return dst


def add_speaker_file(
    db_name: str, src_path: Path, dest_filename: Optional[str] = None
) -> Path:
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


def clips_dir(db_name: str, speaker_name: str, create: bool = True) -> Path:
    """繋ぐ前の素材クリップを置くフォルダのパスを返す。

    Args:
        db_name: 対象の DB 名。
        speaker_name: 話者名。
        create: True なら存在しなければ作る。

    Returns:
        ``<DB>/_clips/<話者名>/`` のパス。

    Raises:
        ValueError: 話者名が無効な場合。
        FileNotFoundError: DB が存在しない場合。
    """
    safe_speaker = sanitize_name(speaker_name)
    if safe_speaker is None:
        raise ValueError(f"無効な話者名: {speaker_name!r}")

    path = database_dir(db_name) / CLIPS_DIR_NAME / safe_speaker
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def list_clips(db_name: str, speaker_name: str) -> List[Path]:
    """話者の素材クリップを名前順で返す。無ければ空リスト。

    ファイル名の先頭に連番が付いているため、名前順＝投入順になる。

    Raises:
        ValueError: 話者名が無効な場合。
        FileNotFoundError: DB が存在しない場合。
    """
    path = clips_dir(db_name, speaker_name, create=False)
    if not path.is_dir():
        return []
    return [
        f
        for f in sorted(path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]


def add_speaker_clips(
    db_name: str,
    speaker_name: str,
    clip_paths: List[Path],
    clip_names: Optional[List[str]] = None,
) -> Path:
    """複数のクリップを1人の話者としてまとめて登録する。

    素材を ``_clips/<話者名>/`` へ連番付きで保存し、そこにある**全素材**を
    連結して ``<話者名>.wav`` を作る。既に同じ話者名があれば、既存の素材も
    含めて繋ぎ直す（あとからクリップを足せるようにするため）。

    Args:
        db_name: 登録先の DB 名。
        speaker_name: 話者名。
        clip_paths: 今回追加するクリップのパス。
        clip_names: 素材として残すときの名前。省略すると ``clip_paths`` の
            ファイル名を使う。Web のアップロードは一時ファイルに衝突回避の
            接頭辞が付くため、元のファイル名をここで渡せるようにしている。

    Returns:
        作成した ``<話者名>.wav`` のパス。

    Raises:
        ValueError: クリップが空、話者名が無効、対応外の拡張子が混ざっている場合、
            ``clip_names`` の数が ``clip_paths`` と合わない場合。
        FileNotFoundError: DB またはクリップが存在しない場合。
        RuntimeError: ffmpeg による連結が失敗した場合。
    """
    if not clip_paths:
        raise ValueError("クリップが指定されていません。1本以上を選んでください。")

    if clip_names is None:
        clip_names = [Path(c).name for c in clip_paths]
    if len(clip_names) != len(clip_paths):
        raise ValueError(
            f"クリップの数と名前の数が合いません: "
            f"{len(clip_paths)}本 / {len(clip_names)}個の名前"
        )

    for clip_name in clip_names:
        suffix = Path(clip_name).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"対応していない拡張子: {clip_name} "
                f"(対応: {sorted(SUPPORTED_AUDIO_EXTENSIONS)})"
            )

    dest_dir = clips_dir(db_name, speaker_name)
    next_number = len(list_clips(db_name, speaker_name)) + 1
    for clip, clip_name in zip(clip_paths, clip_names):
        safe_name = Path(clip_name).name
        shutil.copyfile(clip, dest_dir / f"{next_number:03d}_{safe_name}")
        next_number += 1

    all_clips = list_clips(db_name, speaker_name)
    merged = database_dir(db_name) / f"{dest_dir.name}.wav"
    concat_audio(all_clips, merged)
    logger.info(f"Merged {len(all_clips)} clips into {merged}")
    return merged
