"""声紋登録ディレクトリの読み取りと、話者識別器のプロセス内キャッシュ。

識別器（pyannote の埋め込みモデル）の初期化は重いため、モデル名ごとに1つだけ
作って使い回す。使い回す際は登録済みの話者をいったん空にして、呼び出しごとの
登録内容が混ざらないようにする。
"""

from __future__ import annotations

import os
from pathlib import Path

from src.common.logging import get_logger
from src.diarization.speaker_identifier import SpeakerIdentifier

logger = get_logger(__name__)

#: 声紋登録ディレクトリで対象とする拡張子。
SUPPORTED_REGISTRY_EXTENSIONS = {
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

#: モデル名 → 初期化済みの識別器
_SPEAKER_IDENTIFIER_CACHE: dict[str, SpeakerIdentifier] = {}

#: Hugging Face のオフライン取得を制御する環境変数。
_HF_HUB_OFFLINE = "HF_HUB_OFFLINE"


def get_cached_speaker_identifier(
    model_name: str, hf_token: str, threshold: float
) -> SpeakerIdentifier:
    """識別器を取得する（同じモデル名なら初期化済みのものを再利用する）。

    Args:
        model_name: 埋め込みモデル名。
        hf_token: Hugging Face アクセストークン。
        threshold: 話者一致判定のコサイン距離しきい値。

    Returns:
        登録話者が空の状態にリセットされた識別器。
    """
    cached = _SPEAKER_IDENTIFIER_CACHE.get(model_name)
    if cached is not None:
        cached.threshold = threshold
        cached.registry_embeddings = {}
        cached.unknown_counter = 1
        return cached

    _SPEAKER_IDENTIFIER_CACHE[model_name] = _create_identifier(
        model_name, hf_token, threshold
    )
    return _SPEAKER_IDENTIFIER_CACHE[model_name]


def _create_identifier(
    model_name: str, hf_token: str, threshold: float
) -> SpeakerIdentifier:
    """識別器を作る。オフラインで作れない場合だけ一度オンラインで再試行する。"""
    try:
        return SpeakerIdentifier(
            model_name=model_name, hf_token=hf_token, threshold=threshold
        )
    except Exception:
        # モデルがローカルに無い初回のみ、オフライン指定を一時的に外して取得する
        previous = os.environ.get(_HF_HUB_OFFLINE)
        os.environ[_HF_HUB_OFFLINE] = "0"
        try:
            return SpeakerIdentifier(
                model_name=model_name, hf_token=hf_token, threshold=threshold
            )
        finally:
            _restore_env(_HF_HUB_OFFLINE, previous)


def _restore_env(key: str, previous: str | None) -> None:
    """環境変数を元の状態へ戻す（元が未設定ならキーごと削除する）。"""
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def collect_registry_files(registry_dir: Path) -> dict[str, Path]:
    """声紋登録ディレクトリ内の音声ファイルを収集する。

    Args:
        registry_dir: 登録用音声を置いたディレクトリ。

    Returns:
        ``{話者名: パス}``。話者名はファイル名の stem。

    Raises:
        NotADirectoryError: ディレクトリが存在しない場合。
        ValueError: ファイル名の stem が空、重複、または対象ファイルが無い場合。
    """
    if not registry_dir.is_dir():
        raise NotADirectoryError(f"声紋登録ディレクトリが存在しません: {registry_dir}")

    parsed: dict[str, Path] = {}
    for path in sorted(registry_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_REGISTRY_EXTENSIONS:
            continue
        name = path.stem
        if not name:
            raise ValueError(f"ファイル名が空のため登録できません: {path}")
        if name in parsed:
            raise ValueError(
                f"登録ファイル名（stem）が重複しています: {name} "
                f"({parsed[name]} と {path})"
            )
        parsed[name] = path

    if not parsed:
        raise ValueError(
            f"声紋登録ディレクトリ内に対象音声ファイルが見つかりません: {registry_dir} "
            f"(対象拡張子: {sorted(SUPPORTED_REGISTRY_EXTENSIONS)})"
        )

    return parsed
