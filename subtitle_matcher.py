import os

# PyTorch (MPS) と FAISS-CPU が OpenMP を取り合って segfault する macOS の挙動を抑止
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import csv
import time
import logging
from typing import List, Dict, Any, Tuple, Set
import sys  # コマンドライン引数のために追加

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """uvicorn ログにも CLI にも出るように logging + print(flush) を併用。"""
    logger.info(msg)
    print(msg, flush=True)

# ==============================================================================
# 1. データ読み込みヘルパー
# ==============================================================================

def load_scripts_from_csv(filename: str) -> List[Dict[str, Any]]:
    """
    CSVファイルから台本データを読み込み、整形する。
    """
    scripts = []
    script_id = 0
    try:
        with open(filename, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('type') == 'dialogue':
                    dialogue_text = row.get('contents', '').replace('心の声', '')
                    scripts.append({
                        "id": script_id,
                        "speaker": row.get('speaker', ''),
                        "dialogue": dialogue_text
                    })
                    script_id += 1
        print(f"'{filename}' から {len(scripts)} 件の台詞データを読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
    except Exception as e:
        print(f"エラー: {filename} の読み込み中にエラーが発生しました: {e}")
    return scripts

_MEANINGFUL_CHAR_RE = re.compile(r"[぀-ヿ一-鿿a-zA-Z0-9]")


def _is_meaningful_stt_text(text: str, min_len: int = 2) -> bool:
    """Whisper の幻覚（'!', 'caus', 'me' の反復等）を捨てるためのフィルタ。

    - 正規化後 (`normalize_text`) の長さが min_len 未満なら捨てる
    - 意味のある文字（仮名/漢字/英数）が含まれていなければ捨てる
    - 全文がひとつのトークンの反復（例: 'caus caus caus'）なら捨てる
    """
    norm = normalize_text(text)
    if len(norm) < min_len:
        return False
    if not _MEANINGFUL_CHAR_RE.search(norm):
        return False
    tokens = norm.split()
    if len(tokens) >= 2 and len(set(tokens)) == 1:
        return False
    return True


def load_stt_from_srt(filename: str) -> List[Dict[str, Any]]:
    """
    SRTファイルから字幕データを読み込む。

    不正なセグメント（end <= start や Whisper 幻覚で意味のないテキスト）は破棄する。
    """
    stt = []
    stt_id = 0
    skipped_time = 0
    skipped_noise = 0
    try:
        with open(filename, mode='r', encoding='utf-8') as srtfile:
            content = srtfile.read()
            blocks = content.strip().split('\n\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3 and '-->' in lines[1]:
                    start_str, end_str = [t.strip() for t in lines[1].split('-->')]
                    text = " ".join(lines[2:])

                    # 時刻の妥当性チェック（end <= start は Whisper の幻覚由来でよくある）
                    start_sec = time_str_to_seconds(start_str)
                    end_sec = time_str_to_seconds(end_str)
                    if end_sec <= start_sec:
                        skipped_time += 1
                        continue

                    # 意味のないテキスト（'!', 'caus caus caus' 等）は破棄
                    if not _is_meaningful_stt_text(text):
                        skipped_noise += 1
                        continue

                    stt.append({
                        "id": stt_id,
                        "start": start_str,
                        "end": end_str,
                        "text": text
                    })
                    stt_id += 1
        print(
            f"'{filename}' から {len(stt)} 件の字幕データを読み込みました。"
            f"（時刻不正で除外: {skipped_time}, 幻覚テキストで除外: {skipped_noise}）"
        )
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
    except Exception as e:
        print(f"エラー: {filename} の読み込み中にエラーが発生しました: {e}")
    return stt

# ==============================================================================
# 2. テキスト前処理 & n-gram生成
# ==============================================================================

def time_str_to_seconds(time_str: str) -> float:
    """
    時間文字列 (HH:MM:SS,ms) を秒に変換する。
    """
    try:
        parts = time_str.replace(',', '.').split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        # 不正な形式や空文字の場合は 0.0 を返す
        return 0.0

def seconds_to_time_str(seconds: float) -> str:
    """
    秒を時間文字列 (HH:MM:SS,ms) に変換する。
    """
    if seconds < 0:
        seconds = 0
    try:
        millis = int((seconds * 1000) % 1000)
        total_seconds = int(seconds)
        sec = total_seconds % 60
        total_minutes = total_seconds // 60
        mins = total_minutes % 60
        hours = total_minutes // 60
        return f"{hours:02d}:{mins:02d}:{sec:02d},{millis:03d}"
    except Exception:
        return "00:00:00,000"

def normalize_text(s: str) -> str:
    """
    テキストを正規化して比較しやすくする。
    """
    s = s.replace('（', ' ').replace('）', ' ')
    s = re.sub(r"\[.*?\]", " ", s)
    s = re.sub(r"[、,。．.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def create_ngrams(data_list: List[Dict], text_key: str, max_n: int, has_time: bool = False) -> List[Dict]:
    """
    データリストからn-gramを生成する。
    """
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(data_list) - n + 1):
            chunk = data_list[i:i+n]

            if 'speaker' in chunk[0] and not all(item['speaker'] == chunk[0]['speaker'] for item in chunk):
                continue

            combined_text = " ".join(item[text_key] for item in chunk)
            ngram_dict = {
                "id": len(ngrams),
                "text": combined_text,
                "normalized_text": normalize_text(combined_text),
                "start_index": i,
                "end_index": i + n - 1,
                "original_ids": [item["id"] for item in chunk],
                "n": n
            }
            if 'speaker' in chunk[0]:
                ngram_dict['speaker'] = chunk[0]['speaker']
            if has_time:
                ngram_dict["start_time"] = time_str_to_seconds(chunk[0]["start"])
                ngram_dict["end_time"] = time_str_to_seconds(chunk[-1]["end"])
            ngrams.append(ngram_dict)
    return ngrams

# ==============================================================================
# 3. ベクトル化 & 類似度検索
# ==============================================================================

def encode_texts(texts: List[str], model_name: str = 'stsb-xlm-r-multilingual') -> Tuple[Any, np.ndarray]:
    """
    SentenceTransformerモデルを読み込み、テキストリストをベクトルに変換する。
    """
    _log(f"\nモデル '{model_name}' を読み込んでいます... ({len(texts)} texts)")
    model = SentenceTransformer(model_name)
    _log("モデルの読み込み完了。文章をベクトルに変換しています...")
    t0 = time.time()
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    _log(f"[encode_texts] model.encode done in {time.time()-t0:.2f}s, normalizing...")
    faiss.normalize_L2(embeddings)
    _log(f"ベクトル変換が完了しました。 ({embeddings.shape})")
    return model, embeddings

def find_similar_vectors(query_embeddings: np.ndarray, index_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faissインデックスを構築し、類似ベクトルを検索する。
    """
    _log(
        f"[find_similar_vectors] start: query={query_embeddings.shape}, "
        f"index={index_embeddings.shape}, k={k}"
    )
    t0 = time.time()
    dimension = index_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(index_embeddings)
    _log(f"[find_similar_vectors] index built in {time.time()-t0:.2f}s; searching...")
    t1 = time.time()
    k_eff = min(k, index_embeddings.shape[0])
    distances, indices = index.search(query_embeddings, k_eff)
    _log(f"[find_similar_vectors] search done in {time.time()-t1:.2f}s")
    return distances, indices

# ==============================================================================
# 4. マッチング候補の生成と最適化 (WLIS)
# ==============================================================================

def create_candidate_pairs(script_ngrams: List[Dict], stt_ngrams: List[Dict], distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    """
    検索結果から、重み付けされた類似度を持つ候補ペアのDataFrameを作成する。
    """
    _log("\nWLISアルゴリズムのためのデータ準備中...")
    all_pairs = []
    for i in range(len(script_ngrams)):
        for j in range(distances.shape[1]):
            stt_ngram_index = indices[i][j]
            similarity = distances[i][j]
            script_ng = script_ngrams[i]
            stt_ng = stt_ngrams[stt_ngram_index]

            weight_bonus = 1.0 + 0.1 * (script_ng["n"] - 1)
            weighted_similarity = float(similarity) * weight_bonus

            all_pairs.append({
                "script_ngram_id": script_ng["id"],
                "stt_ngram_id": stt_ng["id"],
                "similarity": similarity,
                "weighted_similarity": weighted_similarity,
                "script_start_index": script_ng["start_index"],
                "script_end_index": script_ng["end_index"],
                "stt_start_time": stt_ng["start_time"],
                "stt_end_time": stt_ng["end_time"],
            })

    sim_df = pd.DataFrame(all_pairs)
    sim_df = sim_df.sort_values(
        ["script_start_index", "stt_start_time", "similarity"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    _log(f"{len(sim_df)}個の候補ペアを作成し、ソートしました。")
    return sim_df

def apply_wlis(sim_df: pd.DataFrame) -> List[int]:
    """
    重み付き最長増加部分シーケンス（WLIS）アルゴリズムを適用して、最適なペアのシーケンスを見つける。
    """
    _log("\n最適なマッチングを探索中...")
    n = len(sim_df)
    if n == 0:
        _log("有効な候補ペアが見つかりませんでした。")
        return []

    # 列をNumPy配列に取り出して .loc[] の遅さを避ける (n^2 のループ向け)
    weights = sim_df["weighted_similarity"].to_numpy()
    s_start = sim_df["script_start_index"].to_numpy()
    s_end = sim_df["script_end_index"].to_numpy()
    t_start_raw = sim_df["stt_start_time"].to_numpy()
    t_end_raw = sim_df["stt_end_time"].to_numpy()
    # 不正な時刻（end < start）でも有効に動くよう、区間を正規化する
    # 区間の前後関係は max(start,end) ≦ min(next_start,next_end) で判定
    t_low = np.minimum(t_start_raw, t_end_raw)
    t_high = np.maximum(t_start_raw, t_end_raw)

    dp = weights.copy().astype(np.float64)
    prev = np.full(n, -1, dtype=int)

    t0 = time.time()
    for i in range(n):
        # ベクトル化: 条件を一括判定して dp[j] + weights[i] が最大の j を選ぶ
        mask = (s_end[:i] < s_start[i]) & (t_high[:i] <= t_low[i])
        if mask.any():
            candidate_scores = dp[:i] + weights[i]
            # mask=False は無視
            candidate_scores = np.where(mask, candidate_scores, -np.inf)
            j_best = int(np.argmax(candidate_scores))
            if candidate_scores[j_best] > dp[i]:
                dp[i] = candidate_scores[j_best]
                prev[i] = j_best
        if i % 500 == 0 and i > 0:
            _log(f"  apply_wlis: {i}/{n} ({time.time()-t0:.1f}s)")
    _log(f"  apply_wlis: loop done in {time.time()-t0:.2f}s")

    max_score_index = int(np.argmax(dp))
    best_path_indices = []
    current_index = max_score_index
    while current_index != -1:
        best_path_indices.append(current_index)
        current_index = prev[current_index]
    best_path_indices.reverse()

    return best_path_indices

# ==============================================================================
# 5. 結果の整形と出力
# ==============================================================================

def process_final_results(best_path_indices: List[int], sim_df: pd.DataFrame, script_ngrams: List[Dict], stt_ngrams: List[Dict]) -> Tuple[List[Dict], Set[int], Set[int]]:
    """
    最適化された結果を最終的なフォーマットに整形する。
    """
    matched_pairs = []
    matched_script_indices = set()
    matched_stt_indices = set()

    for idx in best_path_indices:
        pair = sim_df.loc[idx]
        script_ng = script_ngrams[int(pair["script_ngram_id"])]
        stt_ng = stt_ngrams[int(pair["stt_ngram_id"])]

        matched_pairs.append({"script_ng": script_ng, "stt_ng": stt_ng, "similarity": pair["similarity"]})
        matched_script_indices.update(script_ng["original_ids"])
        matched_stt_indices.update(stt_ng["original_ids"])

    return matched_pairs, matched_script_indices, matched_stt_indices

def export_results_to_csv(scripts: List, stt: List, matched_pairs: List, matched_script_indices: Set, matched_stt_indices: Set, filename: str):
    """
    マッチング結果を時系列でソートし、未使用データと共にCSVに出力する。
    """
    print(f"\n最終結果をCSVファイル '{filename}' に生成しています...")
    timed_data = []

    # Matchedデータを作成
    for match in matched_pairs:
        script_ng, stt_ng = match['script_ng'], match['stt_ng']
        start_time = stt[stt_ng['original_ids'][0]]['start']

        # script_start_idとscript_end_idの設定
        script_start_id = script_ng['original_ids'][0]
        script_end_id = script_ng['original_ids'][-1]

        # ★ 変更: stt_id_range -> stt_start_id / stt_end_id
        stt_ids = stt_ng['original_ids']
        stt_start_id = stt_ids[0]
        stt_end_id = stt_ids[-1]

        # script_speakerとspeakerの設定
        script_speaker = script_ng.get('speaker', '')

        timed_data.append({
            "type": "Matched",
            "script_start_id": script_start_id,
            "script_end_id": script_end_id,
            "script_speaker": script_speaker,
            "script_dialogue": script_ng['text'],
            "stt_start_id": stt_start_id, # ★ 変更
            "stt_end_id": stt_end_id,     # ★ 変更
            "stt_text": stt_ng['text'],
            "start_time": start_time,
            "end_time": stt[stt_ng['original_ids'][-1]]['end'],
            "speaker": script_speaker,
            "subtitle_text": script_ng['text'],
            "sort_key": time_str_to_seconds(start_time)
        })

    # Unmatched_STTデータを作成
    for s in stt:
        if s["id"] not in matched_stt_indices:
            timed_data.append({
                "type": "Unmatched_STT",
                "script_start_id": "",
                "script_end_id": "",
                "script_speaker": "",
                "script_dialogue": "",
                "stt_start_id": s['id'], # ★ 変更
                "stt_end_id": s['id'],   # ★ 変更
                "stt_text": s['text'],
                "start_time": s['start'],
                "end_time": s['end'],
                "speaker": "",
                "subtitle_text": s['text'],
                "sort_key": time_str_to_seconds(s['start'])
            })

    # 時系列順にソート
    timed_data.sort(key=lambda x: x['sort_key'])

    # Unmatched_Scriptデータを作成し、script_start_idでソート
    unmatched_script_data = []
    for s in scripts:
        if s["id"] not in matched_script_indices:
            script_speaker = s.get('speaker', '')
            script_dialogue = s.get('dialogue', '')
            unmatched_script_data.append({
                "type": "Unmatched_Script",
                "script_start_id": s['id'],
                "script_end_id": s['id'],
                "script_speaker": script_speaker,
                "script_dialogue": script_dialogue,
                "stt_start_id": "", # ★ 変更
                "stt_end_id": "",   # ★ 変更
                "stt_text": "",
                "start_time": "", # この後補完する
                "end_time": "",   # この後補完する
                "speaker": script_speaker,
                "subtitle_text": script_dialogue
            })
    unmatched_script_data.sort(key=lambda x: x['script_start_id'])

    # timed_dataとunmatched_script_dataをマージ
    merged_data = []
    unmatched_idx = 0

    for row in timed_data:
        # Matched行の場合のみ、その前にUnmatched_Scriptを挿入
        if row['type'] == 'Matched':
            current_script_id = row['script_start_id']
            # 現在のMatched行のscript_start_idより小さいUnmatched_Scriptをすべて挿入
            while unmatched_idx < len(unmatched_script_data) and unmatched_script_data[unmatched_idx]['script_start_id'] < current_script_id:
                merged_data.append(unmatched_script_data[unmatched_idx])
                unmatched_idx += 1

        # 現在の行を追加
        merged_data.append(row)

    # 残りのUnmatched_Scriptをすべて末尾に追加
    while unmatched_idx < len(unmatched_script_data):
        merged_data.append(unmatched_script_data[unmatched_idx])
        unmatched_idx += 1

    # --- (新設) Unmatched_Script の時間補完処理 ---
    print("Unmatched_Script の時系列を補完しています...")
    i = 0
    while i < len(merged_data):
        row = merged_data[i]
        if row['type'] == 'Unmatched_Script':
            # 連続ブロックの開始
            block_start_index = i
            block_end_index = i

            # 連続ブロックの終了を探す
            while block_end_index + 1 < len(merged_data) and merged_data[block_end_index + 1]['type'] == 'Unmatched_Script':
                block_end_index += 1

            # --- 時間補完の実行 ---

            # 1. 直前の時刻 (prev_time_sec) を決定
            prev_time_sec = 0.0
            if block_start_index > 0:
                prev_row = merged_data[block_start_index - 1]
                # 直前の行 (Matched, Unmatched_STT) の end_time を使う
                if prev_row.get('end_time'):
                    prev_time_sec = time_str_to_seconds(prev_row['end_time'])
                # end_time がない場合、start_time を使う
                elif prev_row.get('start_time'):
                    prev_time_sec = time_str_to_seconds(prev_row['start_time'])

            # 2. 直後の時刻 (next_time_sec) を決定
            next_time_sec = -1.0 # 未設定フラグ
            if block_end_index + 1 < len(merged_data):
                next_row = merged_data[block_end_index + 1]
                if next_row.get('start_time'):
                    next_time_sec = time_str_to_seconds(next_row['start_time'])

            # 直後の時刻が不明(末尾ブロック or 次の行に時刻がない)の場合
            if next_time_sec < 0.0:
                next_time_sec = prev_time_sec # 期間ゼロとして扱う

            # 3. 期間と分割数を計算
            block_size = (block_end_index - block_start_index) + 1
            duration_sec = max(0, next_time_sec - prev_time_sec) # 負の期間は0にする

            if duration_sec > 0:
                chunk_duration_sec = duration_sec / block_size
                current_time_sec = prev_time_sec

                # 4. ブロック内の各行に時間を割り当て
                for j in range(block_start_index, block_end_index + 1):
                    start_sec = current_time_sec
                    end_sec = current_time_sec + chunk_duration_sec

                    merged_data[j]['start_time'] = seconds_to_time_str(start_sec)
                    # 最後の要素はきっちり next_time_sec に合わせる (浮動小数点誤差吸収)
                    if j == block_end_index:
                         merged_data[j]['end_time'] = seconds_to_time_str(next_time_sec)
                    else:
                         merged_data[j]['end_time'] = seconds_to_time_str(end_sec)

                    current_time_sec = end_sec
            else:
                # 期間がない場合 (直後がすぐ来る、または末尾)
                prev_time_str = seconds_to_time_str(prev_time_sec)
                next_time_str = seconds_to_time_str(next_time_sec)
                for j in range(block_start_index, block_end_index + 1):
                    merged_data[j]['start_time'] = prev_time_str
                    merged_data[j]['end_time'] = next_time_str # 期間ゼロ

            # --- 処理完了 ---
            i = block_end_index + 1 # 次のループはブロックの直後から
        else:
            i += 1
    # --- (時間補完 終了) ---

    # CSVに書き込み
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # ★ 変更: ヘッダーの定義
            header = [
                'type',
                'script_start_id', 'script_end_id', 'script_speaker', 'script_dialogue',
                'stt_start_id', 'stt_end_id', 'stt_text',
                'start_time', 'end_time', 'speaker', 'subtitle_text'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            for row in merged_data:
                # sort_keyフィールドがあれば削除
                if 'sort_key' in row:
                    del row['sort_key']
                writer.writerow(row)

        print(f"CSVファイル '{filename}' の生成が完了しました。")
    except IOError as e:
        print(f"エラー: CSVファイルの書き込みに失敗しました: {e}")

def display_summary(scripts: List, stt: List, matched_pairs: List, matched_script_indices: Set, matched_stt_indices: Set):
    """
    コンソールにマッチングの詳細と統計情報を表示する。
    """
    print("\n--- マッチング詳細 ---")
    for i, match in enumerate(matched_pairs, 1):
        script_ng, stt_ng = match["script_ng"], match["stt_ng"]

        # script_rangeの作成
        script_ids = script_ng['original_ids']
        script_range = f"No.{script_ids[0]}" if len(script_ids) == 1 else f"No.{script_ids[0]}-{script_ids[-1]}"

        # stt_rangeの作成
        stt_ids = stt_ng['original_ids']
        stt_range = f"No.{stt_ids[0]}" if len(stt_ids) == 1 else f"No.{stt_ids[0]}-{stt_ids[-1]}"

        print(f"◆ マッチ {i}: 台本 {script_range} ({script_ng['n']}-gram) ⇔ 音声認識 {stt_range} ({stt_ng['n']}-gram)")
        print(f"  [時刻]     : {stt[stt_ng['original_ids'][0]]['start']} --> {stt[stt_ng['original_ids'][-1]]['end']}")
        print(f"  [台本]     : {script_ng['text']}")
        print(f"  [音声認識] : {stt_ng['text']}")
        print(f"  [類似度]   : {match['similarity']:.4f}\n" + "-" * 80)

    print("\n--- 未使用の台本 ---")
    unmatched_scripts = [s for s in scripts if s["id"] not in matched_script_indices]
    for s in unmatched_scripts: print(f"◆ 台本 No.{s['id']}: {s['speaker']}「{s['dialogue']}」")
    if not unmatched_scripts: print("全ての台本要素が使用されました。")

    print("\n--- 未使用の音声認識 ---")
    unmatched_stt = [s for s in stt if s["id"] not in matched_stt_indices]
    for s in unmatched_stt: print(f"◆ 音声認識 No.{s['id']} ({s['start']}): {s['text']}")
    if not unmatched_stt: print("全ての音声認識要素が使用されました。")

    print("\n--- 統計情報 ---")
    print(f"マッチング数: {len(matched_pairs)}組")
    print(f"使用された台本要素: {len(matched_script_indices)}/{len(scripts)} ({len(matched_script_indices)/len(scripts)*100:.1f}%)")
    print(f"使用された音声認識要素: {len(matched_stt_indices)}/{len(stt)} ({len(matched_stt_indices)/len(stt)*100:.1f}%)")

# ==============================================================================
# 6. メイン処理
# ==============================================================================

def run_matching_process(script_file: str, stt_file: str, output_filename: str):
    """
    台本と音声認識のマッチング処理全体を統括する。
    """
    t_start = time.time()
    _log(f"[run_matching_process] start: script={script_file}, stt={stt_file}, out={output_filename}")

    # 1. データ読み込み
    scripts = load_scripts_from_csv(script_file)
    stt = load_stt_from_srt(stt_file)
    if not scripts or not stt:
        _log("データ読み込みに失敗したため、処理を中断します。")
        return

    # 2. n-gram生成
    _log("\nn-gramチャンクを生成しています...")
    script_ngrams = create_ngrams(scripts, text_key="dialogue", max_n=3)
    stt_ngrams = create_ngrams(stt, text_key="text", max_n=3, has_time=True)
    _log(f"台本n-gram: {len(script_ngrams)}個、音声認識n-gram: {len(stt_ngrams)}個を生成しました。")

    # 3. テキストのベクトル化（モデル参照を残さず MPS リソースを早めに解放）
    script_texts = [ng["normalized_text"] for ng in script_ngrams]
    stt_texts = [ng["normalized_text"] for ng in stt_ngrams]
    _, script_embeddings = encode_texts(script_texts)
    _, stt_embeddings = encode_texts(stt_texts)
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    _log("[run_matching_process] embedding cleanup done")

    # 4. 類似度検索
    _log("[run_matching_process] calling find_similar_vectors...")
    distances, indices = find_similar_vectors(script_embeddings, stt_embeddings, k=10)
    _log(f"[run_matching_process] find_similar_vectors returned: distances={distances.shape}, indices={indices.shape}")

    # 5. 候補ペア生成と最適化
    candidate_pairs_df = create_candidate_pairs(script_ngrams, stt_ngrams, distances, indices)
    _log(f"[run_matching_process] candidate_pairs_df: {len(candidate_pairs_df)} rows")
    best_path_indices = apply_wlis(candidate_pairs_df)
    _log(f"[run_matching_process] apply_wlis returned {len(best_path_indices)} paths")

    # 6. 結果の集計と出力
    matched_pairs, matched_script_indices, matched_stt_indices = process_final_results(
        best_path_indices, candidate_pairs_df, script_ngrams, stt_ngrams
    )
    # 引数で受け取った出力ファイル名を渡す
    export_results_to_csv(scripts, stt, matched_pairs, matched_script_indices, matched_stt_indices, filename=output_filename)
    display_summary(scripts, stt, matched_pairs, matched_script_indices, matched_stt_indices)
    _log(f"[run_matching_process] DONE in {time.time()-t_start:.2f}s")


if __name__ == '__main__':
    try:
        # 'faiss' が 'faiss-cpu' または 'faiss-gpu' のどちらかでインポートされることを期待
        import faiss
    except ImportError:
        print("エラー: Faiss ライブラリが見つかりません。")
        print("'pip install faiss-cpu' (CPU版) または 'pip install faiss-gpu' (GPU版) を実行してください。")
        exit()

    try:
        import sentence_transformers
    except ImportError:
        print("エラー: sentence-transformers ライブラリが見つかりません。")
        print("'pip install sentence-transformers' を実行してください。")
        exit()

    try:
        import pandas
    except ImportError:
        print("エラー: pandas ライブラリが見つかりません。")
        print("'pip install pandas' を実行してください。")
        exit()

    # --- コマンドライン引数の処理 ---
    if len(sys.argv) != 4:
        print("エラー: 引数の数が正しくありません。")
        print("使い方: python subtitle_matcher.py 台本CSVファイルパス 音声認識SRTファイルパス 出力CSVファイル名（対応表）")
        exit()

    # --- 設定 ---
    SCRIPT_CSV_FILE = sys.argv[1]          # 引数1
    SPEECH_TO_TEXT_SRT_FILE = sys.argv[2]  # 引数2
    OUTPUT_CSV_FILE = sys.argv[3]          # 引数3

    # --- 実行 ---
    run_matching_process(SCRIPT_CSV_FILE, SPEECH_TO_TEXT_SRT_FILE, OUTPUT_CSV_FILE)
