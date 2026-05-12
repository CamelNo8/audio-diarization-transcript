import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio.core.io import Audio
from pathlib import Path
import warnings
import sys
import logging
from typing import Optional, Dict
import csv
import datetime
import subprocess
import tempfile
import os

from speaker_identification import SpeakerIdentifier

# mlx_whisper のインポート（Mac最適化）
try:
    import mlx_whisper
except ImportError:
    print(
        "Critical Error: mlx_whisper is required but not installed. Please install it using: uv pip install mlx-whisper",
        file=sys.stderr,
    )
    sys.exit(1)

# Hugging Face トークンに関するFutureWarningを抑制
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


def format_time(seconds: float) -> str:
    """秒数を HH:MM:SS:ms 形式（ミリ秒は3桁）に丸めて変換します。"""
    if seconds < 0:
        return "00:00:00:000"
    total_ms = int(round(seconds * 1000))
    hours, remainder_ms = divmod(total_ms, 3600 * 1000)
    minutes, remainder_ms = divmod(remainder_ms, 60 * 1000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}:{millis:03}"

def create_transcript_csv_path(audio_file_path: Path) -> Path:
    """指定された音声ファイルパスから、出力CSVファイルのPathを生成します。"""
    base_name = audio_file_path.stem
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d%H%M%S")
    output_filename = f"{base_name}-transcription-{timestamp_str}.csv"
    return Path.cwd() / output_filename


class AudioProcessor:
    """音声ファイルの前処理、話者分離・特定、文字起こしを行い、結果をCSVに出力するクラス。"""

    _PIPELINE_CACHE: Dict[tuple[str, str], Pipeline] = {}  # [OPTIMIZED]

    def __init__(
        self,
        audio_file: Path,
        output_csv_path: Path,
        mlx_model_id: str,
        pyannote_model_id: str,
        hf_token: str,
        identifier: Optional[SpeakerIdentifier] = None,
    ):
        self.audio_file = audio_file
        self.output_csv_path = output_csv_path
        self.mlx_model_id = mlx_model_id
        self.pyannote_model_id = pyannote_model_id
        self.hf_token = hf_token
        self.identifier = identifier
        
        self.temp_wav_path: Optional[Path] = None
        self.speaker_mapping: Dict[str, str] = {}  # クラスターID -> 話者名
        self.speaker_distance_mapping: Dict[str, Optional[float]] = {}  # クラスターID -> コサイン距離
        self.speaker_candidate_distance_mapping: Dict[str, Optional[Dict[str, float]]] = {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.cleanup()

    def cleanup(self):
        if self.temp_wav_path and self.temp_wav_path.exists():
            try:
                logging.info(f"Cleaning up temporary file: {self.temp_wav_path}")
                self.temp_wav_path.unlink()
            except Exception as e:
                logging.error(f"Failed to delete temporary file: {e}")

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # [OPTIMIZED]
    @classmethod
    def _get_cached_pipeline(
        cls, model_id: str, hf_token: str, device: torch.device
    ) -> Pipeline:
        cache_key = (model_id, hf_token)
        pipeline = cls._PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            logging.info(f"Loading Pyannote pipeline ({model_id})...")  # [OPTIMIZED]
            try:
                pipeline = Pipeline.from_pretrained(
                    model_id, use_auth_token=hf_token, local_files_only=True
                )  # [OPTIMIZED]
            except Exception:
                pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)  # [OPTIMIZED]
            cls._PIPELINE_CACHE[cache_key] = pipeline  # [OPTIMIZED]
        pipeline.to(device)
        return pipeline

    def _set_speaker_metadata(
        self,
        cluster_id: str,
        speaker_name: str,
        distance: Optional[float],
        candidate_distances: Optional[Dict[str, float]],
    ) -> None:
        self.speaker_mapping[cluster_id] = speaker_name
        self.speaker_distance_mapping[cluster_id] = distance
        self.speaker_candidate_distance_mapping[cluster_id] = candidate_distances

    def prepare_audio(self):
        """ffmpegを使用して任意の音声/動画ファイルをWAV形式の一時ファイルに変換します。"""
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.temp_wav_path = Path(temp_path_str)

        logging.info(f"Converting audio to temporary WAV format: {self.temp_wav_path}")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(self.audio_file),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-y",
                    str(self.temp_wav_path)
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            logging.info("Audio conversion successful.")
        except subprocess.CalledProcessError as e:
            logging.critical(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        """全体のプロセス（前処理、話者分離・特定、文字起こし、マージ、CSV保存）を実行します。"""
        # 1. 音声の前処理 (WAV変換)
        self.prepare_audio()

        # 2. 話者分離 (Pyannote Diarization)
        try:
            # use_auth_tokenパラメータを使用して初期化
            pipeline = self._get_cached_pipeline(  # [OPTIMIZED]
                self.pyannote_model_id,
                self.hf_token,
                self._select_device(),
            )
        except Exception as e:
            logging.critical(f"Error loading Pyannote pipeline: {e}")  # [OPTIMIZED]
            return False

        logging.info("Running speaker diarization...")
        diarization_params = {}
        if known_num_speakers is not None:
            diarization_params["num_speakers"] = known_num_speakers

        try:
            diarization = pipeline(str(self.temp_wav_path), **diarization_params)
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}")
            return False

        # 3. 各クラスターごとの話者照合・特定
        if self.identifier:
            logging.info("Identifying speakers for each cluster...")
            audio_io = Audio()
            cluster_segments = []  # [OPTIMIZED]
            for cluster_id in diarization.labels():
                segments = diarization.label_timeline(cluster_id)
                # 特徴量抽出の安定性を高めるため、1秒以上のセグメントを優先
                valid_segments = [s for s in segments if s.duration >= 1.0]
                longest_segment = max(valid_segments, key=lambda s: s.duration) if valid_segments else max(segments, key=lambda s: s.duration)
                cluster_segments.append((cluster_id, longest_segment))  # [OPTIMIZED]

            waveforms = []  # [OPTIMIZED]
            for cluster_id, longest_segment in cluster_segments:
                try:
                    # クラスターの代表音声区間から波形を取得
                    waveform, sr = audio_io.crop(str(self.temp_wav_path), longest_segment)
                    waveforms.append((cluster_id, waveform, sr))  # [OPTIMIZED]
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        f"Unknown_{cluster_id}",
                        None,
                        None,
                    )

            embeddings = []  # [OPTIMIZED]
            for cluster_id, waveform, sr in waveforms:
                try:
                    embedding = self.identifier.get_embedding_from_waveform(waveform, sr)
                    embeddings.append((cluster_id, embedding))  # [OPTIMIZED]
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        f"Unknown_{cluster_id}",
                        None,
                        None,
                    )

            for cluster_id, embedding in embeddings:
                try:
                    identified_name, cosine_distance, candidate_distances = (
                        self.identifier.identify_speaker_with_distances(embedding)
                    )

                    self._set_speaker_metadata(
                        cluster_id,
                        identified_name,
                        cosine_distance,
                        candidate_distances,
                    )
                    logging.info(
                        f"Cluster '{cluster_id}' identified as -> {identified_name} "
                        f"(cosine_distance={cosine_distance:.6f})"
                        if cosine_distance is not None
                        else f"Cluster '{cluster_id}' identified as -> {identified_name}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        f"Unknown_{cluster_id}",
                        None,
                        None,
                    )
        else:
            # 識別器が設定されていない場合はクラスターIDをそのまま使用
            for cluster_id in diarization.labels():
                self._set_speaker_metadata(cluster_id, cluster_id, None, None)

        # 4. 全文一括文字起こし (mlx-whisper)
        logging.info(f"Running mlx-whisper transcription ({self.mlx_model_id})...")
        try:
            whisper_result = mlx_whisper.transcribe(
                str(self.temp_wav_path),
                path_or_hf_repo=self.mlx_model_id,
                verbose=False,
                language="ja"
            )
        except Exception as e:
            logging.error(f"Error during mlx-whisper transcription: {e}", exc_info=True)
            return False

        segments = whisper_result.get("segments", [])
        if not segments:
            logging.warning("No speech segments detected by Whisper.")
            return True

        # 5. マージ処理とCSV出力
        logging.info(f"Merging results and writing to {self.output_csv_path}...")
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(["start", "end", "speaker", "text", "cosine_distance"])

                for seg in segments:
                    w_start, w_end, w_text = seg["start"], seg["end"], seg["text"].strip()
                    if not w_text:
                        continue

                    w_segment = Segment(w_start, w_end)
                    speaker_durations = {}

                    # pyannoteのDiarization結果と被る長さを計算
                    for p_seg, _, cluster_id in diarization.itertracks(yield_label=True):
                        overlap = w_segment & p_seg
                        if overlap:
                            speaker_durations[cluster_id] = speaker_durations.get(cluster_id, 0.0) + overlap.duration

                    # 最も重複時間が長いクラスターを選び、話者名に変換
                    if speaker_durations:
                        best_cluster = max(speaker_durations, key=speaker_durations.get)
                        best_speaker = self.speaker_mapping.get(best_cluster, best_cluster)
                        best_distance = self.speaker_distance_mapping.get(best_cluster)
                        candidate_distances = self.speaker_candidate_distance_mapping.get(best_cluster)
                    else:
                        best_speaker = "Unknown"
                        best_distance = None
                        candidate_distances = None

                    start_str = format_time(w_start)
                    end_str = format_time(w_end)
                    cosine_distance_str = f"{best_distance:.6f}" if best_distance is not None else ""

                    csv_writer.writerow([start_str, end_str, best_speaker, w_text, cosine_distance_str])
                    logging.info(
                        f"  [{start_str} - {end_str}] {best_speaker}: {w_text} "
                        f"(cosine_distance={cosine_distance_str or 'N/A'})"
                    )
                    if candidate_distances:
                        sorted_candidates = sorted(
                            candidate_distances.items(), key=lambda item: item[1]
                        )
                        candidates_str = ", ".join(
                            f"{name}={dist:.6f}" for name, dist in sorted_candidates
                        )
                    else:
                        candidates_str = "N/A"

                    print(
                        f"  [{start_str} - {end_str}] cluster={best_cluster} "
                        f"speaker={best_speaker} candidates: {candidates_str}"
                    )

            logging.info(f"Successfully finished writing results to {self.output_csv_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return False
