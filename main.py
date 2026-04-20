import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pathlib import Path
import warnings
import sys
import logging
from typing import Optional
import csv
import datetime
import argparse
import subprocess
import tempfile
import os
import shutil

# mlx_whisper のインポート（Mac最適化）
try:
    import mlx_whisper
except ImportError:
    print(
        "Critical Error: mlx_whisper is required but not installed. Please install it using: uv pip install mlx-whisper",
        file=sys.stderr,
    )
    sys.exit(1)

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Hugging Face トークンに関するFutureWarningを抑制
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


def format_time(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列に変換します。"""
    delta = datetime.timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if total_seconds < 0:
        return "00:00:00"
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class AudioProcessor:
    """音声ファイルの前処理、話者分離、文字起こしを行い、結果をCSVに出力するクラス。"""

    def __init__(
        self,
        audio_file: Path,
        output_csv_path: Path,
        mlx_model_id: str,
        pyannote_model_id: str,
    ):
        """AudioProcessorを初期化します。"""
        logging.info(f"Initializing AudioProcessor for file: {audio_file}")
        if not audio_file.is_file():
            logging.critical(f"Audio file not found at initialization: {audio_file}")
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        self.audio_file = audio_file
        self.output_csv_path = output_csv_path
        self.mlx_model_id = mlx_model_id
        self.pyannote_model_id = pyannote_model_id
        
        self.temp_wav_path: Optional[Path] = None

    def __enter__(self):
        """with文の開始時に呼ばれます。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with文の終了時、またはエラー発生時に必ず呼ばれ、一時ファイルを削除します。"""
        self.cleanup()

    def cleanup(self):
        """一時ファイルのクリーンアップを行います。"""
        if self.temp_wav_path and self.temp_wav_path.exists():
            try:
                logging.info(f"Cleaning up temporary file: {self.temp_wav_path}")
                self.temp_wav_path.unlink()
            except Exception as e:
                logging.error(f"Failed to delete temporary file: {e}")

    def prepare_audio(self):
        """ffmpegを使用して任意の音声/動画ファイルをWAV形式の一時ファイルに変換します。"""
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # ファイルディスクリプタを閉じておく
        self.temp_wav_path = Path(temp_path_str)

        logging.info(f"Converting audio to temporary WAV format: {self.temp_wav_path}")
        try:
            # -vn: ビデオ除外, -acodec pcm_s16le: PCM 16bit リトルエンディアン, -ar 16000: 16kHz, -ac 1: モノラル
            subprocess.run(
                [
                    "ffmpeg", "-i", str(self.audio_file),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    str(self.temp_wav_path)
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            logging.info("Audio conversion successful.")
        except subprocess.CalledProcessError as e:
            logging.critical(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        """全体のプロセス（前処理、話者分離、文字起こし、マージ、CSV保存）を実行します。"""
        # 1. 音声の前処理 (WAV変換)
        self.prepare_audio()

        # 2. 話者分離 (Pyannote)
        logging.info(f"Loading Pyannote pipeline ({self.pyannote_model_id})...")
        try:
            pipeline = Pipeline.from_pretrained(self.pyannote_model_id)
            # デバイス設定
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            elif torch.backends.mps.is_available():
                pipeline.to(torch.device("mps"))
            else:
                pipeline.to(torch.device("cpu"))
        except Exception as e:
            logging.critical(f"Error loading Pyannote pipeline: {e}")
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

        # 3. 全文一括文字起こし (mlx-whisper)
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

        # 4. マージ処理とCSV出力
        logging.info(f"Merging results and writing to {self.output_csv_path}...")
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(["start", "end", "speaker", "text"])

                for seg in segments:
                    w_start = seg["start"]
                    w_end = seg["end"]
                    w_text = seg["text"].strip()

                    if not w_text:
                        continue

                    # 文字起こしのセグメントに対して、最もオーバーラップが長い話者を特定する
                    w_segment = Segment(w_start, w_end)
                    speaker_durations = {}

                    for p_seg, _, speaker in diarization.itertracks(yield_label=True):
                        overlap = w_segment & p_seg
                        if overlap:
                            # 辞書に話者ごとの重複時間を加算
                            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + overlap.duration

                    # 話者ラベルは個別の仮割り当てを出力せず、常に 'speaker' とする
                    best_speaker = "speaker"

                    start_str = format_time(w_start)
                    end_str = format_time(w_end)

                    csv_writer.writerow([start_str, end_str, best_speaker, w_text])
                    logging.info(f"  [{start_str} - {end_str}] {best_speaker}: {w_text}")

            logging.info(f"Successfully finished writing results to {self.output_csv_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return False


def create_transcript_csv_path(audio_file_path: Path) -> Path:
    """指定された音声ファイルパスから、出力CSVファイルのPathを生成します。"""
    base_name = audio_file_path.stem
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d%H%M%S")
    output_filename = f"{base_name}-transcription-{timestamp_str}.csv"
    return Path.cwd() / output_filename


# ---- スクリプト実行部分 ----
if __name__ == "__main__":
    # ffmpegの存在チェック
    if not shutil.which("ffmpeg"):
        logging.critical("Critical Error: ffmpeg is required but not found in PATH. Please install FFmpeg.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Audio processing script with speaker diarization and mlx-whisper transcription."
    )
    parser.add_argument(
        "audio_file_path", type=Path, help="Path to the audio or video file to process."
    )
    parser.add_argument(
        "--output_csv_path",
        type=Path,
        default=None,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--mlx_model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="Hugging Face ID or path of the mlx-whisper model (e.g., 'mlx-community/whisper-large-v3-mlx').",
    )
    parser.add_argument(
        "--pyannote_model_id",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face ID of the Pyannote diarization model.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Known number of speakers. If not specified, the model estimates automatically.",
    )

    args = parser.parse_args()

    audio_file_path = args.audio_file_path
    output_csv_path = args.output_csv_path
    mlx_model = args.mlx_model
    pyannote_model_id = args.pyannote_model_id
    known_num_speakers = args.num_speakers

    if not audio_file_path.is_file():
        logging.critical(f"Critical Error: Audio file not found at {audio_file_path}")
        sys.exit(1)
    else:
        logging.info(f"Audio file found at {audio_file_path}.")

    if output_csv_path is None:
        try:
            output_csv_path = create_transcript_csv_path(audio_file_path)
            logging.info(f"Output CSV path defaulting to: {output_csv_path}")
        except Exception as e:
            logging.critical(f"Critical Error: Could not generate default output CSV path: {e}")
            sys.exit(1)

    logging.info("Script execution started.")
    logging.info(f"mlx-whisper model: {mlx_model}")
    logging.info(f"Pyannote Diarization model ID: {pyannote_model_id}")

    try:
        # with文を使って、処理終了時に必ず cleanup() が呼ばれるようにする
        with AudioProcessor(
            audio_file=audio_file_path,
            output_csv_path=output_csv_path,
            mlx_model_id=mlx_model,
            pyannote_model_id=pyannote_model_id,
        ) as processor:
            
            success = processor.process_and_save_to_csv(known_num_speakers=known_num_speakers)

            if success:
                logging.info(f"Processing complete. Results saved to {output_csv_path}")
            else:
                logging.error("Processing failed. Please check the logs above for details.")
                sys.exit(1)

    except Exception as e:
        logging.critical(f"A critical error occurred during the main execution: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Script execution finished successfully.")
    sys.exit(0)