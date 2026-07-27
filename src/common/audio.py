"""ffmpeg / ffprobe による音声ファイル操作の共通処理。

外部コマンドは必ずリスト形式で渡す（規約7.3。``shell=True`` と文字列連結を使わない）。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from src.config import TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ


def build_extract_command(
    src: Path,
    dst: Path,
    *,
    start: Optional[float] = None,
    end: Optional[float] = None,
    to_wav16k: bool = True,
    quiet: bool = True,
) -> list[str]:
    """音声切り出し用の ffmpeg コマンドを組み立てる。

    Args:
        src: 入力ファイル。
        dst: 出力ファイル。既存ファイルは上書きする。
        start: 切り出し開始秒。None または 0 なら先頭から。
        end: 切り出し終了秒。None なら末尾まで。
        to_wav16k: True なら 16kHz モノラルの PCM WAV へ再エンコードする。
            False なら出力拡張子に応じたコーデックを ffmpeg に任せる。
        quiet: True なら標準入力を奪わず、ffmpeg のログをエラーのみに絞る。

    Returns:
        subprocess へ渡すコマンドの引数リスト。
    """
    cmd = ["ffmpeg"]
    if quiet:
        cmd += ["-nostdin", "-loglevel", "error"]
    # -ss / -to は入力より前に置く（入力側シークの方が高速なため）
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vn"]
    if to_wav16k:
        cmd += [
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(TARGET_SAMPLE_RATE_HZ),
            "-ac",
            str(TARGET_CHANNELS),
        ]
    cmd += ["-y", str(dst)]
    return cmd


def extract_audio(
    src: Path,
    dst: Path,
    *,
    start: Optional[float] = None,
    end: Optional[float] = None,
    to_wav16k: bool = True,
    quiet: bool = True,
) -> None:
    """ffmpeg で音声を切り出し・変換する。

    引数は :func:`build_extract_command` と同じ。

    Raises:
        subprocess.CalledProcessError: ffmpeg が異常終了した場合。
            ``stderr`` に ffmpeg のエラー出力が入る。
        FileNotFoundError: ffmpeg が PATH に無い場合。
    """
    subprocess.run(
        build_extract_command(
            src, dst, start=start, end=end, to_wav16k=to_wav16k, quiet=quiet
        ),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def probe_duration_sec(audio_path: Path) -> Optional[float]:
    """ffprobe で音声長（秒）を取得する。

    Args:
        audio_path: 対象の音声ファイル。

    Returns:
        音声長（秒）。取得できなかった場合は None。
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logging.warning(f"ffprobe failed for {audio_path}: {e}")
        return None
