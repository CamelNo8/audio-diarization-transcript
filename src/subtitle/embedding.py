"""文埋め込みの生成と FAISS による近傍探索。

このモジュールは PyTorch と FAISS を読み込むため、OpenMP の衝突回避の
環境変数をここで設定する。**本パッケージ内で最初に import されること**を
前提にしている（:mod:`src.subtitle.matcher` を参照）。
"""

from __future__ import annotations

import os

# PyTorch (MPS) と FAISS-CPU が OpenMP を取り合って segfault する macOS の挙動を抑止。
# faiss / sentence_transformers を読み込む前に設定する必要がある。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time  # noqa: E402
from typing import Any  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.common.logging import get_logger  # noqa: E402
from src.config import SENTENCE_EMBEDDING_MODEL  # noqa: E402

logger = get_logger(__name__)

try:
    faiss.omp_set_num_threads(1)
except Exception:
    # 古い faiss には omp_set_num_threads が無い。環境変数だけで十分なため無視する。
    pass


def encode_texts(
    texts: list[str], model_name: str = SENTENCE_EMBEDDING_MODEL
) -> tuple[Any, np.ndarray]:
    """テキストのリストを L2 正規化済みのベクトルに変換する。

    Args:
        texts: ベクトル化する文字列のリスト。
        model_name: SentenceTransformer のモデル名。

    Returns:
        (読み込んだモデル, ベクトルの2次元配列) の組。
    """
    logger.info(f"\nモデル '{model_name}' を読み込んでいます... ({len(texts)} texts)")
    model = SentenceTransformer(model_name)
    logger.info("モデルの読み込み完了。文章をベクトルに変換しています...")
    t0 = time.time()
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    logger.info(
        f"[encode_texts] model.encode done in {time.time() - t0:.2f}s, normalizing..."
    )
    faiss.normalize_L2(embeddings)
    logger.info(f"ベクトル変換が完了しました。 ({embeddings.shape})")
    return model, embeddings


def find_similar_vectors(
    query_embeddings: np.ndarray, index_embeddings: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """FAISS の内積インデックスを構築し、各クエリの近傍を探索する。

    Args:
        query_embeddings: 検索するベクトル（台本側）。
        index_embeddings: 検索対象のベクトル（音声認識側）。
        k: 1件あたり取得する候補数。対象件数を超える場合は対象件数に丸める。

    Returns:
        (類似度, インデックス) の組。どちらも ``(クエリ数, k)`` の形。
    """
    logger.info(
        f"[find_similar_vectors] start: query={query_embeddings.shape}, "
        f"index={index_embeddings.shape}, k={k}"
    )
    t0 = time.time()
    index = faiss.IndexFlatIP(index_embeddings.shape[1])
    index.add(index_embeddings)
    logger.info(
        f"[find_similar_vectors] index built in {time.time() - t0:.2f}s; searching..."
    )
    t1 = time.time()
    k_eff = min(k, index_embeddings.shape[0])
    distances, indices = index.search(query_embeddings, k_eff)
    logger.info(f"[find_similar_vectors] search done in {time.time() - t1:.2f}s")
    return distances, indices


def release_model_memory() -> None:
    """埋め込みモデルが確保した GPU / MPS メモリを解放する。

    モデルへの参照を捨てたあとに呼ぶ。解放できない環境では何もしない。
    """
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        # torch が無い / MPS 非対応でも処理は続行できる
        pass
