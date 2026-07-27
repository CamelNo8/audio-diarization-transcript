"""音声の前処理から話者分離・照合・文字起こし・CSV 出力までの統括。

工程は「前処理（WAV変換・背景音除去）→ 話者分離 → 話者照合 → 文字起こし →
セグメントへの話者割り当て → 出力」の順で、各工程は同パッケージ内の
専用モジュール（:mod:`~src.diarization.pipeline` 等）に委ねる。
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyannote.core import Segment

from src.common.audio import extract_audio
from src.common.csv_io import write_rows
from src.common.logging import get_logger
from src.diarization import clusters as clusters_mod
from src.diarization.clusters import ClusterAssignments
from src.diarization.interactive import resolve_unknown_speakers
from src.diarization.pipeline import (
    get_cached_pipeline,
    select_device,
    unwrap_diarization,
)
from src.diarization import transcript
from src.diarization.speaker_identifier import SpeakerIdentifier
from src.diarization.vocals import extract_vocals
from src.transcription.backend import transcribe_full

logger = get_logger(__name__)

#: 文字起こしの言語。
TRANSCRIPTION_LANGUAGE = "ja"


def create_transcript_csv_path(audio_file_path: Path) -> Path:
    """音声ファイルパスから、出力 CSV のパスを生成する。

    Args:
        audio_file_path: 対象の音声ファイル。

    Returns:
        カレントディレクトリ直下の ``<音声名>-transcription-<日時>.csv``。
    """
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return Path.cwd() / f"{audio_file_path.stem}-transcription-{timestamp_str}.csv"


class AudioProcessor:
    """音声の前処理、話者分離・特定、文字起こしを行い、結果を CSV に出力する。"""

    def __init__(
        self,
        audio_file: Path,
        output_csv_path: Path,
        mlx_model_id: str,
        pyannote_model_id: str,
        hf_token: str,
        identifier: Optional[SpeakerIdentifier] = None,
        registry_dir: Optional[Path] = None,
        interactive_unknown_resolve: bool = True,
        denoise: bool = False,
        separator_model: Optional[str] = None,
        whisper_backend: str = "auto",
    ):
        self.audio_file = audio_file
        self.output_csv_path = output_csv_path
        self.mlx_model_id = mlx_model_id
        self.whisper_backend = whisper_backend
        self.pyannote_model_id = pyannote_model_id
        self.hf_token = hf_token
        self.identifier = identifier
        self.registry_dir = registry_dir
        self.interactive_unknown_resolve = interactive_unknown_resolve
        self.denoise = denoise
        self.separator_model = separator_model

        self.temp_wav_path: Optional[Path] = None
        self.assignments = ClusterAssignments()

    def __enter__(self) -> "AudioProcessor":
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """前処理で作った一時 WAV を削除する。"""
        if self.temp_wav_path and self.temp_wav_path.exists():
            try:
                logger.info(f"Cleaning up temporary file: {self.temp_wav_path}")
                self.temp_wav_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete temporary file: {e}")

    # ------------------------------------------------------------------
    # 前処理
    # ------------------------------------------------------------------

    def prepare_audio(self) -> None:
        """任意の音声/動画ファイルを 16kHz mono の WAV 一時ファイルへ変換する。

        ``denoise=True`` の場合は変換後にボーカルを抽出し、背景音/BGM を除去する。

        Raises:
            RuntimeError: ffmpeg による変換に失敗した場合。
        """
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.temp_wav_path = Path(temp_path_str)

        logger.info(f"Converting audio to temporary WAV format: {self.temp_wav_path}")
        try:
            # quiet=False: 変換失敗時の原因を stderr にそのまま残すため
            extract_audio(self.audio_file, self.temp_wav_path, quiet=False)
            logger.info("Audio conversion successful.")
        except subprocess.CalledProcessError as e:
            logger.critical(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

        if self.denoise:
            self._replace_with_vocals()

    def _replace_with_vocals(self) -> None:
        """背景音を除去した WAV へ ``temp_wav_path`` を差し替える。

        除去できなかった場合は元の WAV のまま処理を続行する。
        """
        assert self.temp_wav_path is not None
        replaced = extract_vocals(self.temp_wav_path, self.separator_model)
        if replaced is None:
            return
        try:
            self.temp_wav_path.unlink()
        except OSError:
            pass
        self.temp_wav_path = replaced
        logger.info(f"ボーカル抽出済み: {self.temp_wav_path}")

    # ------------------------------------------------------------------
    # CLI / Web 向け: 話者名つき CSV まで出力する
    # ------------------------------------------------------------------

    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        """前処理・話者分離／特定・文字起こし・CSV 保存までを実行する。

        Args:
            known_num_speakers: 既知の話者数。``None`` ならモデルが推定する。

        Returns:
            すべて成功したら True。途中で失敗したら False。

        Note:
            発話が 1 件も検出されなくても失敗とはせず、ヘッダ行だけの CSV を
            書き出す。「完走したら CSV が存在する」という後段（CLI・Web UI）の
            前提を満たすため。
        """
        self.prepare_audio()

        diarization = self._run_diarization(known_num_speakers)
        if diarization is None:
            return False
        self._assign_speakers(diarization)

        try:
            segments = self._transcribe()
        except Exception as e:
            logger.error(f"Error during whisper transcription: {e}", exc_info=True)
            return False

        if not segments:
            logger.warning("No speech segments detected by Whisper.")

        return self._write_transcript_csv(diarization, segments)

    def _run_diarization(self, known_num_speakers: Optional[int]):
        """話者分離を実行する。失敗した場合はログを残して ``None`` を返す。"""
        try:
            pipeline = get_cached_pipeline(
                self.pyannote_model_id, self.hf_token, select_device()
            )
        except Exception as e:
            logger.critical(f"Error loading Pyannote pipeline: {e}")
            return None

        logger.info("Running speaker diarization...")
        try:
            result = pipeline(
                str(self.temp_wav_path), **_diarization_params(known_num_speakers)
            )
        except Exception as e:
            logger.error(f"Error during speaker diarization: {e}")
            return None
        return unwrap_diarization(result)

    def _assign_speakers(self, diarization) -> None:
        """各クラスタに話者名を割り当てる（識別器が無ければ全て Unknown）。"""
        if self.identifier is None:
            clusters_mod.label_all_as_unknown(diarization, self.assignments)
            return

        logger.info("Identifying speakers for each cluster...")
        clusters_mod.identify_clusters(
            diarization, self.identifier, self.temp_wav_path, self.assignments
        )
        if self.registry_dir is not None and self.interactive_unknown_resolve:
            resolve_unknown_speakers(
                self.identifier,
                self.assignments,
                self.temp_wav_path,
                self.registry_dir,
            )

    def _transcribe(self) -> List[Dict[str, Any]]:
        """音声全体を文字起こしし、セグメントのリストを返す。"""
        result = transcribe_full(
            self.temp_wav_path,
            model_id=self.mlx_model_id,
            language=TRANSCRIPTION_LANGUAGE,
            prefer_device=select_device(),
            backend=self.whisper_backend,
        )
        return result.get("segments", [])

    def _write_transcript_csv(self, diarization, segments: List[Dict]) -> bool:
        """文字起こしセグメントに話者を割り当て、CSV へ書き出す。"""
        logger.info(f"Merging results and writing to {self.output_csv_path}...")
        rows = transcript.build_rows(diarization, segments, self.assignments)

        try:
            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_rows(self.output_csv_path, rows)
        except OSError as e:
            logger.error(f"Error saving to CSV: {e}")
            return False

        logger.info(f"Successfully finished writing results to {self.output_csv_path}")
        return True

    # ------------------------------------------------------------------
    # API 向け: 話者照合と CSV 出力は行わず、結果を dict で返す
    # ------------------------------------------------------------------

    def process_for_api(
        self,
        known_num_speakers: Optional[int] = None,
        vocals_out: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Spark などのリモート推論サーバー用に、前処理〜マージまでを実行する。

        話者名は付与しない（照合は呼び出し側が vocals WAV を使って行うため、
        クラスタIDのみ返す）。

        Args:
            known_num_speakers: 既知の話者数。
            vocals_out: 処理済み WAV のコピー先。

        Returns:
            ``segments`` / ``clusters`` / ``num_speakers`` / ``vocals_path``
            を持つ dict。
        """
        self.prepare_audio()

        pipeline = get_cached_pipeline(
            self.pyannote_model_id, self.hf_token, select_device()
        )
        logger.info("Running speaker diarization (API)...")
        diarization = unwrap_diarization(
            pipeline(str(self.temp_wav_path), **_diarization_params(known_num_speakers))
        )

        clusters = {
            cluster_id: _representative_bounds(diarization, cluster_id)
            for cluster_id in diarization.labels()
        }
        segments = self._merge_segments_with_clusters(diarization, self._transcribe())

        return {
            "segments": segments,
            "clusters": clusters,
            "num_speakers": len(clusters),
            "vocals_path": self._copy_vocals(vocals_out),
        }

    def _merge_segments_with_clusters(
        self, diarization, segments: List[Dict]
    ) -> List[Dict[str, Any]]:
        """文字起こしセグメントに、最も重なるクラスタIDを付ける。"""
        merged = []
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            start, end = float(seg["start"]), float(seg["end"])
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "cluster_id": clusters_mod.dominant_cluster(
                        diarization, Segment(start, end)
                    ),
                }
            )
        return merged

    def _copy_vocals(self, vocals_out: Optional[Path]) -> Optional[str]:
        """処理済み（背景音除去済み）WAV を照合用に確定パスへコピーする。"""
        if vocals_out is None or self.temp_wav_path is None:
            return None
        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.temp_wav_path, vocals_out)
        return str(vocals_out)

    # ------------------------------------------------------------------
    # Web UI 向け: Unknown クラスタの永続化
    # ------------------------------------------------------------------

    def persist_unknown_clusters(self, output_dir: Path) -> list:
        """Unknown と判定されたクラスタの代表音声を保存し、メタ情報を返す。

        Args:
            output_dir: 代表音声の出力先。

        Returns:
            :func:`src.diarization.clusters.persist_unknown_clusters` の戻り値。
        """
        return clusters_mod.persist_unknown_clusters(
            self.assignments, self.temp_wav_path, output_dir
        )


def _diarization_params(known_num_speakers: Optional[int]) -> Dict[str, int]:
    """話者分離パイプラインへ渡す追加パラメータを組み立てる。"""
    if known_num_speakers is None:
        return {}
    return {"num_speakers": known_num_speakers}


def _representative_bounds(diarization, cluster_id: str) -> Dict[str, float]:
    """クラスタの代表区間の開始・終了秒を返す（照合用）。"""
    rep = clusters_mod.pick_representative_segment(
        diarization.label_timeline(cluster_id)
    )
    return {"rep_start": float(rep.start), "rep_end": float(rep.end)}
