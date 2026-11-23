"""窓口1: 映像/音声ファイルを文字起こしして字幕 SRT を作る。

入口は2つ。

- :func:`transcribe_to_srt` … 映像/音声 → 字幕SRT（＋文字起こしCSV・RTTM）
- :func:`rebuild_srt_from_csv` … 話者名を直した CSV から SRT を作り直す

工程は「前処理（WAV変換・背景音除去）→ 話者分離 → 話者照合 → 文字起こし →
話者の割り当て → CSV/SRT 出力」。重い工程の中身は
:mod:`src.diarization` と :mod:`src.whisper_backends` にある。

**話者分離はここで1回だけ走る。** 照合（:mod:`src.speaker_verification`）には
その結果を渡す。照合側でもう一度分離すると一番重い工程が二重になるため。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.csv_io import read_dict_rows
from src.common.logging import get_logger
from src.common.timecode import colon_ms_to_comma_ms
from src.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    DENOISE_MODELS,
)
from src.diarization.processor import AudioProcessor
from src.diarization.speaker_identifier import SpeakerIdentifier

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptionSettings:
    """文字起こしのモデル設定。"""

    #: Whisper のモデルID（``large-v3`` などの短縮表記も可）
    model_id: str = DEFAULT_WHISPER_MODEL
    #: Whisper の実行系。``auto`` は環境に応じて選ぶ
    backend: str = "auto"
    #: 話者分離（pyannote）のモデルID
    diarization_model_id: str = DEFAULT_DIARIZATION_MODEL
    #: Hugging Face トークン
    hf_token: str = ""
    #: 既知の総話者数。``None`` ならモデルが推定する
    num_speakers: Optional[int] = None
    #: 背景音除去の強度（``off`` / ``fast`` / ``high``）
    denoise_mode: str = "off"


@dataclass(frozen=True)
class TranscriptionOutputs:
    """文字起こしの生成物。"""

    #: 1区間1行の文字起こし CSV（start,end,speaker,text,cosine_distance）
    csv_path: Path
    #: 字幕 SRT。本文は ``[話者] 本文``
    srt_path: Path
    #: 未登録だった話者クラスタのメタ情報（ラベル付け画面が使う）
    unknown_clusters: List[Dict[str, Any]]


def transcribe_to_srt(
    media_path: Path,
    srt_path: Path,
    csv_path: Path,
    *,
    settings: TranscriptionSettings,
    identifier: Optional[SpeakerIdentifier] = None,
    unknown_clusters_dir: Optional[Path] = None,
) -> Optional[TranscriptionOutputs]:
    """映像/音声を文字起こしし、話者名つきの字幕 SRT を書き出す。

    Args:
        media_path: 対象の映像/音声ファイル。
        srt_path: 出力する字幕 SRT のパス。
        csv_path: 出力する文字起こし CSV のパス（RTTM も同じ場所に残す）。
        settings: モデルと前処理の設定。
        identifier: :func:`src.speaker_verification.load_verifier` が返した照合器。
            ``None`` なら話者名は付けず ``Unknown_NN`` のままにする。
        unknown_clusters_dir: 未登録話者の代表音声の保存先。``None`` なら保存しない。

    Returns:
        生成物。途中で失敗した場合は ``None``。
    """
    separator_model = DENOISE_MODELS.get(settings.denoise_mode)
    with AudioProcessor(
        audio_file=media_path,
        output_csv_path=csv_path,
        mlx_model_id=settings.model_id,
        pyannote_model_id=settings.diarization_model_id,
        hf_token=settings.hf_token,
        identifier=identifier,
        denoise=separator_model is not None,
        separator_model=separator_model,
        whisper_backend=settings.backend,
    ) as processor:
        is_success = processor.process_and_save_to_csv(
            known_num_speakers=settings.num_speakers
        )
        # 一時 WAV が消える前に、未登録話者の代表音声を取り出しておく
        unknown_clusters = (
            processor.persist_unknown_clusters(unknown_clusters_dir)
            if is_success and unknown_clusters_dir is not None
            else []
        )

    if not is_success or not csv_path.exists():
        return None

    rebuild_srt_from_csv(csv_path, srt_path)
    return TranscriptionOutputs(csv_path, srt_path, unknown_clusters)


def rebuild_srt_from_csv(csv_path: Path, srt_path: Path) -> int:
    """文字起こし CSV から ``[話者] 本文`` 形式の SRT を作る。

    話者名は空でも ``Unknown_NN`` でもそのまま前に付ける（利用者が
    視認できるように）。この形式は窓口3（:mod:`src.matching`）の
    SRT 読み込みが前提にしているので変えない。

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
