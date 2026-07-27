"""登録話者の管理と、埋め込みのコサイン距離による話者照合。

登録音声の埋め込みは（同じファイル・同じモデル・同じ前処理なら）決定的なので、
:mod:`src.diarization.embedding_cache` 経由で永続キャッシュする。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.diarization.torch_compat import patch_torch_load

patch_torch_load()

from pyannote.audio import Inference, Model  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from src.common.audio import extract_audio, probe_duration_sec  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_SPEAKER_THRESHOLD,
    REGISTRATION_MIN_REMAINING_SEC,
    REGISTRATION_TRIM_SEC,
)
from src.diarization.embedding_cache import (  # noqa: E402
    EmbeddingCache,
    get_default_cache,
)

logger = get_logger(__name__)

#: 埋め込みキャッシュのスキーマ版。register_speaker の前処理（トリミング量・
#: 正規化方法など）を変えたらここを上げると旧キャッシュと分離される。
_EMBEDDING_CACHE_SCHEMA = "v1"


class SpeakerIdentifier:
    """登録話者の管理、特徴量抽出、照合を行うクラス。"""

    def __init__(
        self,
        model_name: str,
        hf_token: str,
        threshold: float = DEFAULT_SPEAKER_THRESHOLD,
        cache: Optional[EmbeddingCache] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.inference = self._load_model(model_name, hf_token)
        self.registry_embeddings: Dict[str, np.ndarray] = {}
        self.unknown_counter = 1
        self._cache = cache if cache is not None else get_default_cache()

    @property
    def _cache_namespace(self) -> str:
        """埋め込みキャッシュの名前空間（モデル・前処理パラメータ・版を畳み込む）。"""
        return (
            f"{self.model_name}__trim{REGISTRATION_TRIM_SEC}__{_EMBEDDING_CACHE_SCHEMA}"
        )

    def _load_model(self, model_name: str, hf_token: str) -> Inference:
        """埋め込みモデルを読み込み、音声全体を1ベクトルにする推論器を返す。

        Raises:
            ValueError: Hugging Face トークンが未設定の場合。
        """
        logger.info(f"Loading embedding model ({model_name})...")
        if not hf_token:
            raise ValueError(
                "Hugging Face トークンが未設定です。"
                "環境変数 HF_TOKEN か --hf_token で指定してください。"
            )
        try:
            model = Model.from_pretrained(model_name, token=hf_token)
        except TypeError:
            # 古い huggingface_hub / pyannote は use_auth_token を取る
            model = Model.from_pretrained(model_name, use_auth_token=hf_token)

        return Inference(model, window="whole")

    def register_speaker(self, name: str, audio_path: Path) -> None:
        """音声ファイルから特徴量を抽出し、話者を登録する。

        登録音声の前後をノイズ排除のためにトリミングしてから埋め込みを抽出する。
        埋め込みは永続キャッシュ経由で取得し、同一ファイルの再計算（GPU 推論）を避ける。

        Args:
            name: 登録する話者名。
            audio_path: 登録用音声のパス。

        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合。
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"登録用音声ファイルが見つかりません: {audio_path}")

        self.registry_embeddings[name] = self._cache.get_or_compute(
            self._cache_namespace,
            audio_path,
            self._compute_registration_embedding,
        )
        logger.info(f"Speaker registered: {name} ({audio_path})")

    def _compute_registration_embedding(self, audio_path: Path) -> np.ndarray:
        """登録音声を前処理してから埋め込みを抽出・正規化する（キャッシュミス時のみ）。"""
        trimmed_path: Optional[Path] = None
        try:
            trimmed_path = self._preprocess_registration_audio(audio_path)
            target_path = trimmed_path if trimmed_path is not None else audio_path
            embedding = self.inference(str(target_path))
            return self._normalize_embedding(embedding)
        finally:
            if trimmed_path is not None and trimmed_path.exists():
                try:
                    trimmed_path.unlink()
                except OSError as e:
                    logger.warning(
                        f"Failed to delete temporary trimmed file {trimmed_path}: {e}"
                    )

    def _preprocess_registration_audio(self, audio_path: Path) -> Optional[Path]:
        """登録音声の前後を切り落とした一時 WAV を返す。

        Returns:
            トリミング後の一時ファイル。音声が短すぎる／長さを取得できない場合は
            ``None``（呼び出し元は元音声をそのまま使う）。
        """
        duration = probe_duration_sec(audio_path)
        trim = REGISTRATION_TRIM_SEC
        required = trim * 2 + REGISTRATION_MIN_REMAINING_SEC

        if duration is None:
            logger.warning(
                f"音声長を取得できなかったためトリミングをスキップします: {audio_path}"
            )
            return None
        if duration < required:
            logger.warning(
                f"音声が短すぎるためトリミングをスキップします "
                f"(duration={duration:.3f}s, required>{required:.3f}s): {audio_path}"
            )
            return None

        fd, tmp_path_str = tempfile.mkstemp(suffix=".wav", prefix="register_trimmed_")
        os.close(fd)
        tmp_path = Path(tmp_path_str)
        try:
            extract_audio(audio_path, tmp_path, start=trim, end=duration - trim)
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"登録音声のトリミングに失敗したため元音声を使用します "
                f"({audio_path}): {e.stderr}"
            )
            _unlink_quietly(tmp_path)
            return None

        return tmp_path

    def get_embedding_from_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> np.ndarray:
        """pyannote から直接切り出した波形データから特徴量を抽出する。"""
        embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        return self._normalize_embedding(embedding)

    def identify_from_audio_path(
        self, audio_path: Path
    ) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
        """音声ファイルのパスから直接話者を照合する（Web の事後ラベル付け向け）。

        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合。
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
        embedding = self.inference(str(audio_path))
        normalized = self._normalize_embedding(embedding)
        return self.identify_speaker_with_distances(normalized)

    def _normalize_embedding(self, embedding) -> np.ndarray:
        """埋め込みベクトルを L2 ノルムで正規化し ``(1, D)`` 形に整える。

        Raises:
            ValueError: ノルムが 0 の場合（無音・極端に短い音声）。
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        embedding_array = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError(
                "埋め込みベクトルのノルムが 0 です。"
                "音声が短すぎるか無音の可能性があります。"
            )

        return embedding_array / norm

    def _next_unknown_name(self) -> str:
        """未登録話者へ払い出す ``Unknown_NN`` を返す（呼ぶたびに採番が進む）。"""
        name = f"Unknown_{self.unknown_counter:02d}"
        self.unknown_counter += 1
        return name

    def identify_speaker(self, embedding: np.ndarray) -> Tuple[str, Optional[float]]:
        """推定話者名とコサイン距離を返す。"""
        name, best_dist, _ = self.identify_speaker_with_distances(embedding)
        return name, best_dist

    def identify_speaker_with_distances(
        self, embedding: np.ndarray
    ) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
        """推定話者名・最短距離・全登録話者との距離を返す。

        最短距離が閾値以下なら登録話者、超えていれば ``Unknown_NN`` とする。
        登録話者がいない場合は ``(Unknown_NN, None, None)``。
        """
        if not self.registry_embeddings:
            return self._next_unknown_name(), None, None

        distances = {
            name: float(cdist(reg_embedding, embedding, metric="cosine")[0, 0])
            for name, reg_embedding in self.registry_embeddings.items()
        }
        best_name = min(distances, key=distances.get)
        best_dist = distances[best_name]

        if best_dist <= self.threshold:
            return best_name, best_dist, distances
        return self._next_unknown_name(), best_dist, distances


def _unlink_quietly(path: Path) -> None:
    """一時ファイルを削除する。消せなくても処理は続行する。"""
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        pass
