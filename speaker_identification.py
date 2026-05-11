import numpy as np
import torch
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cdist
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

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
        """指定した音声ファイルから特徴量を抽出し、話者を登録する"""
        if not audio_path.exists():
            raise FileNotFoundError(f"登録用音声ファイルが見つかりません: {audio_path}")
        
        embedding = self.inference(str(audio_path))
        self.registry_embeddings[name] = self._normalize_embedding(embedding)
        logging.info(f"Speaker registered: {name} ({audio_path})")

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
        name = f"Unknown_{self.unknown_counter}"
        self.unknown_counter += 1
        return name

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