"""候補ペア生成・WLIS・結果整形の特性テスト。

リファクタリング前の ``subtitle_matcher.py`` の振る舞いを固定する。
分割後の ``src/subtitle/alignment.py`` に対して同じ期待値を保つ。
埋め込みモデルと FAISS は呼ばず、検索結果（距離・インデックス）を直接与える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.subtitle.alignment import (
    apply_wlis,
    create_candidate_pairs,
    process_final_results,
)


def _台本ngram(ngram_id: int, n: int, start_index: int, **extra) -> dict:
    return {
        "id": ngram_id,
        "n": n,
        "start_index": start_index,
        "end_index": start_index + n - 1,
        **extra,
    }


def _字幕ngram(ngram_id: int, start_time: float, end_time: float, **extra) -> dict:
    return {
        "id": ngram_id,
        "start_time": start_time,
        "end_time": end_time,
        **extra,
    }


def _ペア表(rows: list[tuple[float, int, int, float, float]]) -> pd.DataFrame:
    """(重み, 台本開始, 台本終了, 字幕開始秒, 字幕終了秒) から WLIS 入力を作る。"""
    return pd.DataFrame(
        [
            {
                "weighted_similarity": weight,
                "script_start_index": s_start,
                "script_end_index": s_end,
                "stt_start_time": t_start,
                "stt_end_time": t_end,
            }
            for weight, s_start, s_end, t_start, t_end in rows
        ]
    )


class Test候補ペアの生成:
    """create_candidate_pairs — 近傍検索の結果を DataFrame に展開する。"""

    @pytest.fixture
    def 検索結果(self):
        script_ngrams = [
            _台本ngram(0, n=1, start_index=0),
            _台本ngram(1, n=2, start_index=0),
        ]
        stt_ngrams = [
            _字幕ngram(0, 0.0, 1.0),
            _字幕ngram(1, 2.0, 3.0),
        ]
        distances = np.array([[0.9, 0.5], [0.8, 0.4]], dtype=np.float32)
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        return script_ngrams, stt_ngrams, distances, indices

    def test_台本ngram数と近傍数の積だけ行が作られる(self, 検索結果):
        df = create_candidate_pairs(*検索結果)

        assert len(df) == 4

    def test_必要な列がすべて揃う(self, 検索結果):
        df = create_candidate_pairs(*検索結果)

        assert set(df.columns) == {
            "script_ngram_id",
            "stt_ngram_id",
            "similarity",
            "weighted_similarity",
            "script_start_index",
            "script_end_index",
            "stt_start_time",
            "stt_end_time",
        }

    def test_ngramが長いほど類似度に重みが加算される(self, 検索結果):
        df = create_candidate_pairs(*検索結果)

        # n=1 は係数 1.0、n=2 は係数 1.1
        def 重み(script_ngram_id: int, stt_ngram_id: int) -> float:
            行 = df[
                (df["script_ngram_id"] == script_ngram_id)
                & (df["stt_ngram_id"] == stt_ngram_id)
            ]
            return float(行["weighted_similarity"].iloc[0])

        assert 重み(0, 0) == pytest.approx(0.9, abs=1e-6)
        assert 重み(1, 1) == pytest.approx(0.8 * 1.1, abs=1e-6)

    def test_台本位置_字幕時刻_類似度降順の順に並ぶ(self, 検索結果):
        df = create_candidate_pairs(*検索結果)

        assert df["stt_start_time"].tolist() == [0.0, 0.0, 2.0, 2.0]
        assert df["similarity"].tolist() == pytest.approx(
            [0.9, 0.4, 0.8, 0.5], abs=1e-6
        )

    def test_インデックスは0から振り直される(self, 検索結果):
        df = create_candidate_pairs(*検索結果)

        assert df.index.tolist() == [0, 1, 2, 3]


class TestWLISによる最適経路の探索:
    """apply_wlis — 台本順・時刻順の両方が単調増加する組合せを選ぶ。"""

    def test_台本順と時刻順がそろう組合せが選ばれる(self):
        # 行2 は行0・行1 と台本位置が重なるため連結できない
        df = _ペア表(
            [
                (1.0, 0, 0, 0.0, 1.0),
                (1.0, 1, 1, 2.0, 3.0),
                (1.5, 0, 1, 0.0, 3.0),
            ]
        )

        assert apply_wlis(df) == [0, 1]

    def test_台本位置が重なるペアは連結されない(self):
        df = _ペア表(
            [
                (1.0, 0, 1, 0.0, 1.0),
                (1.0, 1, 2, 2.0, 3.0),
            ]
        )

        # script_end_index(1) < script_start_index(1) が偽なので単独選択になる
        assert apply_wlis(df) == [0]

    def test_時刻が逆行するペアは連結されない(self):
        df = _ペア表(
            [
                (1.0, 0, 0, 5.0, 6.0),
                (1.0, 1, 1, 0.0, 1.0),
            ]
        )

        assert apply_wlis(df) == [0]

    def test_終了が開始より前の不正な時刻でも区間として扱われる(self):
        # (end < start) の行でも min/max に正規化して前後関係を判定する
        df = _ペア表(
            [
                (1.0, 0, 0, 1.0, 0.0),
                (1.0, 1, 1, 2.0, 3.0),
            ]
        )

        assert apply_wlis(df) == [0, 1]

    def test_境界として終了時刻と開始時刻が等しい場合は連結できる(self):
        df = _ペア表(
            [
                (1.0, 0, 0, 0.0, 2.0),
                (1.0, 1, 1, 2.0, 3.0),
            ]
        )

        assert apply_wlis(df) == [0, 1]

    def test_1件だけならその1件が選ばれる(self):
        df = _ペア表([(0.5, 0, 0, 0.0, 1.0)])

        assert apply_wlis(df) == [0]

    def test_空の候補表では空リストが返る(self):
        assert apply_wlis(pd.DataFrame()) == []

    def test_同スコアが並ぶ場合は先に現れた方が選ばれる(self):
        df = _ペア表(
            [
                (1.0, 0, 0, 0.0, 1.0),
                (1.0, 0, 0, 0.0, 1.0),
            ]
        )

        assert apply_wlis(df) == [0]

    def test_必要な列が欠けている場合はKeyErrorになる(self):
        df = _ペア表([(1.0, 0, 0, 0.0, 1.0)]).drop(columns=["stt_end_time"])

        with pytest.raises(KeyError):
            apply_wlis(df)


class Test最終結果の整形:
    """process_final_results — 選ばれた行を元データの id 集合へ戻す。"""

    @pytest.fixture
    def 整形材料(self):
        script_ngrams = [
            _台本ngram(0, n=1, start_index=0, original_ids=[0]),
            _台本ngram(1, n=2, start_index=1, original_ids=[1, 2]),
        ]
        stt_ngrams = [
            _字幕ngram(0, 0.0, 1.0, original_ids=[10]),
            _字幕ngram(1, 2.0, 3.0, original_ids=[11, 12]),
        ]
        sim_df = pd.DataFrame(
            [
                {"script_ngram_id": 0, "stt_ngram_id": 0, "similarity": 0.9},
                {"script_ngram_id": 1, "stt_ngram_id": 1, "similarity": 0.8},
            ]
        )
        return script_ngrams, stt_ngrams, sim_df

    def test_選ばれた行が台本と字幕のngramに解決される(self, 整形材料):
        script_ngrams, stt_ngrams, sim_df = 整形材料

        matched, _, _ = process_final_results([0, 1], sim_df, script_ngrams, stt_ngrams)

        assert len(matched) == 2
        assert matched[0]["script_ng"] is script_ngrams[0]
        assert matched[0]["stt_ng"] is stt_ngrams[0]
        assert matched[0]["similarity"] == pytest.approx(0.9)

    def test_使用済みの元idが集合として集まる(self, 整形材料):
        script_ngrams, stt_ngrams, sim_df = 整形材料

        _, script_ids, stt_ids = process_final_results(
            [0, 1], sim_df, script_ngrams, stt_ngrams
        )

        assert script_ids == {0, 1, 2}
        assert stt_ids == {10, 11, 12}

    def test_選択が空なら空の結果が返る(self, 整形材料):
        script_ngrams, stt_ngrams, sim_df = 整形材料

        matched, script_ids, stt_ids = process_final_results(
            [], sim_df, script_ngrams, stt_ngrams
        )

        assert matched == []
        assert script_ids == set()
        assert stt_ids == set()
