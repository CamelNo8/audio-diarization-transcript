"""候補ペアの生成と、重み付き最長増加部分列（WLIS）による対応付けの決定。

台本と音声認識はどちらも時系列に進むという前提を使い、「台本の位置」と
「音声認識の時刻」がともに単調増加する組合せのうち、類似度の合計が最大に
なるものを選ぶ。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

#: N-gram が1つ長くなるごとに類似度へ掛ける係数の増分。
#: 長い一致ほど偶然ではないため、短い一致より優先する。
_LENGTH_BONUS_PER_N = 0.1

#: WLIS のループで進捗ログを出す間隔（行数）。
_PROGRESS_LOG_INTERVAL = 500


def create_candidate_pairs(
    script_ngrams: list[dict],
    stt_ngrams: list[dict],
    distances: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    """近傍探索の結果を、重み付き類似度を持つ候補ペアの表に展開する。

    Args:
        script_ngrams: 台本側の N-gram。
        stt_ngrams: 音声認識側の N-gram。
        distances: 近傍探索が返した類似度。``(台本N-gram数, 近傍数)``。
        indices: 近傍探索が返した ``stt_ngrams`` のインデックス。同じ形。

    Returns:
        台本位置・音声認識時刻の昇順、同着なら類似度の降順に並べた DataFrame。
    """
    logger.info("\nWLISアルゴリズムのためのデータ準備中...")
    all_pairs = []
    for i in range(len(script_ngrams)):
        script_ng = script_ngrams[i]
        weight_bonus = 1.0 + _LENGTH_BONUS_PER_N * (script_ng["n"] - 1)
        for j in range(distances.shape[1]):
            stt_ng = stt_ngrams[indices[i][j]]
            similarity = distances[i][j]
            all_pairs.append(
                {
                    "script_ngram_id": script_ng["id"],
                    "stt_ngram_id": stt_ng["id"],
                    "similarity": similarity,
                    "weighted_similarity": float(similarity) * weight_bonus,
                    "script_start_index": script_ng["start_index"],
                    "script_end_index": script_ng["end_index"],
                    "stt_start_time": stt_ng["start_time"],
                    "stt_end_time": stt_ng["end_time"],
                }
            )

    sim_df = pd.DataFrame(all_pairs)
    sim_df = sim_df.sort_values(
        ["script_start_index", "stt_start_time", "similarity"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    logger.info(f"{len(sim_df)}個の候補ペアを作成し、ソートしました。")
    return sim_df


def apply_wlis(sim_df: pd.DataFrame) -> list[int]:
    """重み付き最長増加部分列を求め、採用する候補ペアの行番号を返す。

    Args:
        sim_df: :func:`create_candidate_pairs` が返した候補ペアの表。

    Returns:
        採用した行番号のリスト（台本順）。候補が無ければ空リスト。

    Raises:
        KeyError: 必要な列が欠けている場合。
    """
    logger.info("\n最適なマッチングを探索中...")
    if len(sim_df) == 0:
        logger.info("有効な候補ペアが見つかりませんでした。")
        return []

    dp, prev = _solve_wlis(sim_df)
    return _trace_back(dp, prev)


def _solve_wlis(sim_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """各行を終点としたときの最大スコア（dp）と、直前の行（prev）を求める。"""
    # 列を NumPy 配列に取り出して .loc[] の遅さを避ける（n^2 のループ向け）
    weights = sim_df["weighted_similarity"].to_numpy()
    s_start = sim_df["script_start_index"].to_numpy()
    s_end = sim_df["script_end_index"].to_numpy()
    t_start_raw = sim_df["stt_start_time"].to_numpy()
    t_end_raw = sim_df["stt_end_time"].to_numpy()
    # 不正な時刻（end < start）でも有効に動くよう、区間を正規化する。
    # 区間の前後関係は max(start,end) <= min(next_start,next_end) で判定する。
    t_low = np.minimum(t_start_raw, t_end_raw)
    t_high = np.maximum(t_start_raw, t_end_raw)

    n = len(sim_df)
    dp = weights.copy().astype(np.float64)
    prev = np.full(n, -1, dtype=int)

    t0 = time.time()
    for i in range(n):
        # 条件を一括判定して dp[j] + weights[i] が最大の j を選ぶ
        mask = (s_end[:i] < s_start[i]) & (t_high[:i] <= t_low[i])
        if mask.any():
            candidate_scores = np.where(mask, dp[:i] + weights[i], -np.inf)
            j_best = int(np.argmax(candidate_scores))
            if candidate_scores[j_best] > dp[i]:
                dp[i] = candidate_scores[j_best]
                prev[i] = j_best
        if i % _PROGRESS_LOG_INTERVAL == 0 and i > 0:
            logger.info(f"  apply_wlis: {i}/{n} ({time.time() - t0:.1f}s)")
    logger.info(f"  apply_wlis: loop done in {time.time() - t0:.2f}s")
    return dp, prev


def _trace_back(dp: np.ndarray, prev: np.ndarray) -> list[int]:
    """最大スコアの行から prev をたどって経路を復元する。"""
    best_path_indices = []
    current_index = int(np.argmax(dp))
    while current_index != -1:
        best_path_indices.append(current_index)
        current_index = prev[current_index]
    best_path_indices.reverse()
    return best_path_indices


def process_final_results(
    best_path_indices: list[int],
    sim_df: pd.DataFrame,
    script_ngrams: list[dict],
    stt_ngrams: list[dict],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    """採用した行を、元の台本・音声認識の N-gram と使用済み id 集合に戻す。

    Args:
        best_path_indices: :func:`apply_wlis` が返した行番号。
        sim_df: 候補ペアの表。
        script_ngrams: 台本側の N-gram。
        stt_ngrams: 音声認識側の N-gram。

    Returns:
        (マッチしたペアのリスト, 使用済み台本 id, 使用済み音声認識 id) の組。
    """
    matched_pairs = []
    matched_script_indices: set[int] = set()
    matched_stt_indices: set[int] = set()

    for idx in best_path_indices:
        pair = sim_df.loc[idx]
        script_ng = script_ngrams[int(pair["script_ngram_id"])]
        stt_ng = stt_ngrams[int(pair["stt_ngram_id"])]

        matched_pairs.append(
            {"script_ng": script_ng, "stt_ng": stt_ng, "similarity": pair["similarity"]}
        )
        matched_script_indices.update(script_ng["original_ids"])
        matched_stt_indices.update(stt_ng["original_ids"])

    return matched_pairs, matched_script_indices, matched_stt_indices
