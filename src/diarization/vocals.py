"""audio-separator による背景音（BGM）除去。

モデルは Hugging Face 経由でダウンロードされるため、FB CDN が不通の環境でも動く。
除去に失敗した場合は元音声で処理を続けられるよう ``None`` を返し、致命的エラーには
しない。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.common.audio import extract_audio
from src.common.logging import get_logger
from src.config import DEFAULT_SEPARATOR_MODEL

logger = get_logger(__name__)


def extract_vocals(
    wav_path: Path, separator_model: Optional[str] = None
) -> Optional[Path]:
    """WAV からボーカルを抽出し、16kHz mono に変換した新しい一時 WAV を返す。

    Args:
        wav_path: 入力 WAV（16kHz mono を想定）。
        separator_model: audio-separator のモデルファイル名。省略時は既定値。

    Returns:
        抽出後の一時 WAV のパス。抽出できなかった場合は ``None``
        （呼び出し元は元音声のまま処理を続ける）。
    """
    model_name = separator_model or DEFAULT_SEPARATOR_MODEL
    logger.info(f"audio-separator で背景音を除去中（model={model_name}）...")

    outdir = wav_path.parent / f"sep_{wav_path.stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    output_files = _separate(wav_path, outdir, model_name)
    if output_files is None:
        shutil.rmtree(outdir, ignore_errors=True)
        return None

    vocals_path = _find_vocals_file(output_files, outdir)
    if vocals_path is None:
        shutil.rmtree(outdir, ignore_errors=True)
        return None

    converted = _to_16k_mono(vocals_path)
    shutil.rmtree(outdir, ignore_errors=True)
    return converted


def _separate(wav_path: Path, outdir: Path, model_name: str) -> Optional[list]:
    """audio-separator を実行し、出力ファイル名のリストを返す。失敗時は ``None``。"""
    try:
        from audio_separator.separator import Separator  # type: ignore
    except ImportError:
        logger.warning(
            "audio-separator が見つかりません。"
            "`uv sync` で依存をインストールしてください。"
            "元音声で処理を続行します。"
        )
        return None

    try:
        separator = Separator(output_dir=str(outdir), log_level=logging.WARNING)
        separator.load_model(model_filename=model_name)
        return separator.separate(str(wav_path))
    except Exception as e:
        logger.warning(f"audio-separator 失敗。元音声で続行します。\n{e}")
        return None


def _find_vocals_file(output_files: list, outdir: Path) -> Optional[Path]:
    """出力ファイル群からボーカル側の WAV を選ぶ。

    ``output_files`` は ``["...(Vocals)....wav", "...(Instrumental)....wav"]`` の形。
    パスが相対だったり実在しなかったりする実装差があるため、出力先も走査する。
    """
    vocals_file = next(
        (f for f in (output_files or []) if "vocal" in Path(f).name.lower()), None
    )
    if not vocals_file:
        logger.warning(f"ボーカル抽出ファイルが見つかりません: {output_files}")
        return None

    vocals_path = Path(vocals_file)
    if not vocals_path.is_absolute():
        vocals_path = outdir / vocals_path.name
    if vocals_path.exists():
        return vocals_path

    candidates = list(outdir.rglob("*[Vv]ocals*.wav"))
    if not candidates:
        logger.warning(f"ボーカル WAV が見当たりません: {outdir}")
        return None
    return candidates[0]


def _to_16k_mono(vocals_path: Path) -> Optional[Path]:
    """抽出したボーカルを 16kHz mono の一時 WAV へ変換する。失敗時は ``None``。"""
    fd, replaced_str = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    replaced_path = Path(replaced_str)
    try:
        # quiet=False: 変換失敗時の原因を stderr にそのまま残すため
        extract_audio(vocals_path, replaced_path, quiet=False)
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"ボーカル出力の 16kHz 変換に失敗。元音声で続行します。\n{e.stderr}"
        )
        try:
            replaced_path.unlink()
        except OSError:
            pass
        return None
    return replaced_path
