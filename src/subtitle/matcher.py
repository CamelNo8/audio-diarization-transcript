"""台本と音声認識の対応付け処理のオーケストレーション。

Web UI からは別プロセス（``python -m src.subtitle.matcher``）として起動され、
標準出力が画面に転載される。そのため CLI ではログを標準出力へ流す。
"""

from __future__ import annotations

import sys
import time

# embedding が OpenMP の環境変数を設定するため、numpy / faiss / torch を
# 読み込む他のモジュール（alignment 等）より先に import する。
from src.subtitle.embedding import (
    encode_texts,
    find_similar_vectors,
    release_model_memory,
)
from src.common.logging import configure_logging, get_logger
from src.config import NGRAM_MAX_N, SIMILAR_VECTOR_TOP_K
from src.subtitle.alignment import (
    apply_wlis,
    create_candidate_pairs,
    process_final_results,
)
from src.subtitle.loader import load_scripts_from_csv, load_stt_from_srt
from src.subtitle.ngram import create_ngrams
from src.subtitle.report import display_summary, export_results_to_csv

logger = get_logger(__name__)

#: CLI の引数の数（台本CSV・音声認識SRT・出力CSV）。
_EXPECTED_ARG_COUNT = 3

_USAGE = (
    "使い方: python -m src.subtitle.matcher 台本CSVファイルパス "
    "音声認識SRTファイルパス 出力CSVファイル名（対応表）"
)


def run_matching_process(script_file: str, stt_file: str, output_filename: str) -> None:
    """台本と音声認識のマッチング処理全体を統括する。

    Args:
        script_file: 台本 CSV のパス。
        stt_file: 音声認識 SRT のパス。
        output_filename: 出力する対応表 CSV のパス。
    """
    t_start = time.time()
    logger.info(
        f"[run_matching_process] start: script={script_file}, "
        f"stt={stt_file}, out={output_filename}"
    )

    scripts = load_scripts_from_csv(script_file)
    stt = load_stt_from_srt(stt_file)
    if not scripts or not stt:
        logger.info("データ読み込みに失敗したため、処理を中断します。")
        return

    script_ngrams, stt_ngrams = _build_ngrams(scripts, stt)
    distances, indices = _search_similar(script_ngrams, stt_ngrams)
    matched_pairs, matched_script_ids, matched_stt_ids = _align(
        script_ngrams, stt_ngrams, distances, indices
    )
    export_results_to_csv(
        scripts,
        stt,
        matched_pairs,
        matched_script_ids,
        matched_stt_ids,
        filename=output_filename,
    )
    display_summary(scripts, stt, matched_pairs, matched_script_ids, matched_stt_ids)
    logger.info(f"[run_matching_process] DONE in {time.time() - t_start:.2f}s")


def _build_ngrams(
    scripts: list[dict], stt: list[dict]
) -> tuple[list[dict], list[dict]]:
    """台本・音声認識それぞれの N-gram を生成する。

    Returns:
        (台本の N-gram, 音声認識の N-gram) の組。
    """
    logger.info("\nn-gramチャンクを生成しています...")
    script_ngrams = create_ngrams(scripts, text_key="dialogue", max_n=NGRAM_MAX_N)
    stt_ngrams = create_ngrams(stt, text_key="text", max_n=NGRAM_MAX_N, has_time=True)
    logger.info(
        f"台本n-gram: {len(script_ngrams)}個、"
        f"音声認識n-gram: {len(stt_ngrams)}個を生成しました。"
    )
    return script_ngrams, stt_ngrams


def _search_similar(script_ngrams: list[dict], stt_ngrams: list[dict]):
    """N-gram をベクトル化し、台本の各 N-gram に近い音声認識 N-gram を探す。

    Returns:
        (類似度, インデックス) の組。
    """
    script_embeddings, stt_embeddings = _encode_ngrams(script_ngrams, stt_ngrams)
    logger.info("[run_matching_process] calling find_similar_vectors...")
    distances, indices = find_similar_vectors(
        script_embeddings, stt_embeddings, k=SIMILAR_VECTOR_TOP_K
    )
    logger.info(
        f"[run_matching_process] find_similar_vectors returned: "
        f"distances={distances.shape}, indices={indices.shape}"
    )
    return distances, indices


def _encode_ngrams(script_ngrams: list[dict], stt_ngrams: list[dict]):
    """台本・音声認識の N-gram をベクトル化する。

    モデルへの参照を残さず、変換後すぐに MPS / GPU のメモリを解放する。

    Returns:
        (台本のベクトル, 音声認識のベクトル) の組。
    """
    _, script_embeddings = encode_texts([ng["normalized_text"] for ng in script_ngrams])
    _, stt_embeddings = encode_texts([ng["normalized_text"] for ng in stt_ngrams])
    release_model_memory()
    logger.info("[run_matching_process] embedding cleanup done")
    return script_embeddings, stt_embeddings


def _align(script_ngrams: list[dict], stt_ngrams: list[dict], distances, indices):
    """候補ペアの生成から最適経路の決定・整形までをまとめて行う。

    Returns:
        (マッチしたペア, 使用済み台本 id, 使用済み音声認識 id) の組。
    """
    candidate_pairs_df = create_candidate_pairs(
        script_ngrams, stt_ngrams, distances, indices
    )
    logger.info(
        f"[run_matching_process] candidate_pairs_df: {len(candidate_pairs_df)} rows"
    )
    best_path_indices = apply_wlis(candidate_pairs_df)
    logger.info(
        f"[run_matching_process] apply_wlis returned {len(best_path_indices)} paths"
    )
    return process_final_results(
        best_path_indices, candidate_pairs_df, script_ngrams, stt_ngrams
    )


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。

    Args:
        argv: コマンドライン引数（プログラム名を除く）。省略時は ``sys.argv`` を使う。

    Returns:
        終了コード。引数の数が合わない場合は 1。
    """
    configure_logging()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != _EXPECTED_ARG_COUNT:
        logger.error("エラー: 引数の数が正しくありません。")
        logger.error(_USAGE)
        return 1
    run_matching_process(args[0], args[1], args[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
