import numpy as np
import torch
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cdist
from pathlib import Path
import logging
import os
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

# 登録音声の前後から削るノイズ排除用マージン（秒）
REGISTRATION_TRIM_SEC = 0.5


class SpeakerIdentifier:
    """登録話者の管理、特徴量抽出、照合を行うクラス"""

    def __init__(self, model_name: str, hf_token: str, threshold: float = 0.5):
        self.threshold = threshold
        self.inference = self._load_model(model_name, hf_token)
        self.registry_embeddings = {}
        self.unknown_counter = 1

    def _load_model(self, model_name: str, hf_token: str) -> Inference:
        logging.info(f"Loading embedding model ({model_name})...")
        if not hf_token:
            raise ValueError(
                "Hugging Face トークンが未設定です。"
                "環境変数 HF_TOKEN か --hf_token で指定してください。"
            )
        try:
            model = Model.from_pretrained(model_name, use_auth_token=hf_token)
        except TypeError:
            model = Model.from_pretrained(model_name, token=hf_token)
        
        return Inference(model, window="whole")

    def register_speaker(self, name: str, audio_path: Path):
        """指定した音声ファイルから特徴量を抽出し、話者を登録する。

        登録音声の前後をノイズ排除のためにトリミングしてから埋め込みを抽出する。
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"登録用音声ファイルが見つかりません: {audio_path}")

        trimmed_path: Optional[Path] = None
        try:
            trimmed_path = self._preprocess_registration_audio(audio_path)
            target_path = trimmed_path if trimmed_path is not None else audio_path
            embedding = self.inference(str(target_path))
            self.registry_embeddings[name] = self._normalize_embedding(embedding)
            logging.info(f"Speaker registered: {name} ({audio_path})")
        finally:
            if trimmed_path is not None and trimmed_path.exists():
                try:
                    trimmed_path.unlink()
                except OSError as e:
                    logging.warning(f"Failed to delete temporary trimmed file {trimmed_path}: {e}")

    def _preprocess_registration_audio(self, audio_path: Path) -> Optional[Path]:
        """登録音声の前後を REGISTRATION_TRIM_SEC 秒トリミングした一時 WAV を返す。

        音声が短すぎてトリミングできない場合は None を返し、呼び出し元は元音声をそのまま使う。
        """
        duration = self._probe_duration_sec(audio_path)
        trim = REGISTRATION_TRIM_SEC
        min_remaining = 0.5  # トリム後に残したい最低長

        if duration is None:
            logging.warning(
                f"音声長を取得できなかったためトリミングをスキップします: {audio_path}"
            )
            return None
        if duration < trim * 2 + min_remaining:
            logging.warning(
                f"音声が短すぎるためトリミングをスキップします "
                f"(duration={duration:.3f}s, required>{trim * 2 + min_remaining:.3f}s): {audio_path}"
            )
            return None

        end_time = duration - trim
        fd, tmp_path_str = tempfile.mkstemp(suffix=".wav", prefix="register_trimmed_")
        os.close(fd)
        tmp_path = Path(tmp_path_str)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error",
                    "-ss", f"{trim:.3f}",
                    "-to", f"{end_time:.3f}",
                    "-i", str(audio_path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-y",
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(
                f"登録音声のトリミングに失敗したため元音声を使用します ({audio_path}): {e.stderr}"
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None

        return tmp_path

    @staticmethod
    def _probe_duration_sec(audio_path: Path) -> Optional[float]:
        """ffprobe で音声長（秒）を取得する。失敗時は None。"""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
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

    def get_embedding_from_waveform(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Pyannoteから直接切り出した波形データから特徴量を抽出する"""
        embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        return self._normalize_embedding(embedding)

    def _normalize_embedding(self, embedding) -> np.ndarray:
        """埋め込みベクトルを正規化（L2ノルムで除算）する"""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("埋め込みベクトルのノルムが 0 です。音声が短すぎるか無音の可能性があります。")

        return embedding_array / norm

    def _next_unknown_name(self) -> str:
        self.unknown_counter += 1
        return "Unknown"

    def identify_speaker(self, embedding: np.ndarray) -> Tuple[str, Optional[float]]:
        """抽出した特徴量と登録話者を比較し、推定話者名とコサイン距離を返す"""
        name, best_dist, _ = self.identify_speaker_with_distances(embedding)
        return name, best_dist

    def identify_speaker_with_distances(
        self, embedding: np.ndarray
    ) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
        """推定話者名、最短距離、全候補の距離を返す"""
        if not self.registry_embeddings:
            # 登録話者がいない場合はすぐにUnknownを返す
            return self._next_unknown_name(), None, None

        distances = {}
        for name, reg_embedding in self.registry_embeddings.items():
            dist = float(cdist(reg_embedding, embedding, metric="cosine")[0, 0])
            distances[name] = dist
            
        # 最も距離が近い（最近傍の）登録話者を見つける
        best_name = min(distances, key=distances.get)
        best_dist = distances[best_name]

        # 最短距離が閾値以下であれば特定、超えていればUnknownとする
        if best_dist <= self.threshold:
            return best_name, best_dist, distances

        return self._next_unknown_name(), best_dist, distances