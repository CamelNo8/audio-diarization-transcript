"""窓口3: 台本と文字起こしを対応付けて対応表を作る。

入口は2つ。

- :func:`run_matching` … 台本CSV＋文字起こしSRT → 対応表CSV
- :func:`script_csv_from_text` … 手動書き起こし .txt → 台本CSV

Web UI からは別プロセス（``python -m src.matching``）として起動され、
標準出力が画面に転載される。そのため CLI ではログを標準出力へ流す。
"""

from __future__ import annotations

import csv
import io
import re
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
    "使い方: python -m src.matching 台本CSVファイルパス "
    "音声認識SRTファイルパス 出力CSVファイル名（対応表）"
)

#: ``話者:本文`` 形式の行。話者名は行頭の記号を除いた 31 文字未満に限る。
_SPEAKER_LINE_RE = re.compile(r"^\s*([^\s:：（(][^:：]{0,30}?)\s*[:：]\s*(.+)$")

#: 丸括弧だけで囲まれた行（ト書き・場面説明）。
_SCENE_PAREN_RE = re.compile(r"^\s*[（(](.+)[)）]\s*$")


def run_matching(
    script_csv_path: str, transcript_srt_path: str, mapping_csv_path: str
) -> None:
    """台本と文字起こしを対応付け、対応表 CSV を書き出す。

    処理の流れは「読み込み → N-gram 生成 → 文埋め込み → 近傍探索 →
    候補ペア → WLIS で最適経路 → 対応表の書き出し」。

    Args:
        script_csv_path: 台本 CSV のパス。
        transcript_srt_path: 文字起こし SRT のパス。
        mapping_csv_path: 出力する対応表 CSV のパス。
    """
    t_start = time.time()
    logger.info(
        f"[run_matching] start: script={script_csv_path}, "
        f"stt={transcript_srt_path}, out={mapping_csv_path}"
    )

    scripts = load_scripts_from_csv(script_csv_path)
    stt = load_stt_from_srt(transcript_srt_path)
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
        filename=mapping_csv_path,
    )
    display_summary(scripts, stt, matched_pairs, matched_script_ids, matched_stt_ids)
    logger.info(f"[run_matching] DONE in {time.time() - t_start:.2f}s")


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
    logger.info("[run_matching] calling find_similar_vectors...")
    distances, indices = find_similar_vectors(
        script_embeddings, stt_embeddings, k=SIMILAR_VECTOR_TOP_K
    )
    logger.info(
        f"[run_matching] find_similar_vectors returned: "
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
    logger.info("[run_matching] embedding cleanup done")
    return script_embeddings, stt_embeddings


def _align(script_ngrams: list[dict], stt_ngrams: list[dict], distances, indices):
    """候補ペアの生成から最適経路の決定・整形までをまとめて行う。

    Returns:
        (マッチしたペア, 使用済み台本 id, 使用済み音声認識 id) の組。
    """
    candidate_pairs_df = create_candidate_pairs(
        script_ngrams, stt_ngrams, distances, indices
    )
    logger.info(f"[run_matching] candidate_pairs_df: {len(candidate_pairs_df)} rows")
    best_path_indices = apply_wlis(candidate_pairs_df)
    logger.info(f"[run_matching] apply_wlis returned {len(best_path_indices)} paths")
    return process_final_results(
        best_path_indices, candidate_pairs_df, script_ngrams, stt_ngrams
    )


def script_csv_from_text(txt_bytes: bytes) -> bytes:
    """手動書き起こしのテキストを台本 CSV（id,scene_id,type,speaker,contents）にする。

    各非空行を 1 行に変換する:

    - ``# <内容>`` または ``（...）`` ``(...)`` → type=scene, speaker=空
    - ``<話者>:<本文>`` / ``<話者>：<本文>`` → type=dialogue, speaker=話者
    - それ以外 → type=dialogue, speaker=空, contents=行

    id は 1 から連番、scene_id は空欄。

    Args:
        txt_bytes: UTF-8（BOM 付き可）のテキスト。

    Returns:
        BOM 付き UTF-8 の CSV バイト列。
    """
    text = txt_bytes.decode("utf-8-sig")
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id", "scene_id", "type", "speaker", "contents"])
    idx = 1
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        writer.writerow([idx, *_classify_script_line(line)])
        idx += 1
    return buf.getvalue().encode("utf-8-sig")


def _classify_script_line(line: str) -> tuple[str, str, str, str]:
    """台本 1 行を ``(scene_id, type, speaker, contents)`` に振り分ける。"""
    if line.startswith("#"):
        return "", "scene", "", line.lstrip("#").strip()
    m_scene = _SCENE_PAREN_RE.match(line)
    if m_scene:
        return "", "scene", "", m_scene.group(1).strip()
    m_speaker = _SPEAKER_LINE_RE.match(line)
    if m_speaker:
        return "", "dialogue", m_speaker.group(1).strip(), m_speaker.group(2).strip()
    return "", "dialogue", "", line


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
    run_matching(args[0], args[1], args[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
