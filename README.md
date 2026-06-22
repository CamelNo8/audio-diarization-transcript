# 話者照合 ＋ 文字起こし ＋ 字幕生成 統合アプリ (Audio Diarization & Subtitle)

音声・動画ファイルから文字起こしを行い、**事前に登録した話者の声と照合して「誰が話したか」を特定**し、
さらに**台本と突き合わせて字幕（SRT）まで生成**する統合アプリです。

- **文字起こし** … 環境に応じた Whisper 実装で日本語を高速・高精度に書き起こし
- **話者分離 (Diarization)** … pyannote で発話区間を検出
- **話者照合 (Identification)** … pyannote/embedding ＋ コサイン距離の最近傍探索で、登録済み話者の実名を付与
- **字幕生成** … 台本／手動書き起こしテキストと文字起こし結果をマッチングし、対応表 CSV → 字幕 SRT を出力

CLI（`main.py`）と、3 ステップを通しで行える **FastAPI + htmx の Web UI**（`app.py`）の両方を提供します。

---

## 動作環境

### OS と文字起こしバックエンド

文字起こしバックエンドは実行環境に応じて自動で切り替わります（`transcription_backend.py`）。
ユーザーはモデルID（`large-v3` 等）をそのまま指定でき、必要に応じて各バックエンド用に変換されます。

| 環境 | Whisper 実装 | 備考 |
|---|---|---|
| macOS (Apple Silicon) | **mlx-whisper** | Metal 最適化 |
| Windows / Linux (x86_64) | **faster-whisper** | NVIDIA GPU があれば CUDA、無ければ CPU フォールバック |
| DGX Spark (GB10 / aarch64) | **transformers (CUDA)** | aarch64 では CTranslate2 が CUDA 非対応のため。後述の専用セットアップが必要 |

- **Python**: 3.10 以上
- **FFmpeg**: システムにインストールされていること（音声の前処理に使用）
  - macOS: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` もしくは <https://www.gyan.dev/ffmpeg/builds/> から取得し PATH を通す
  - Linux: `apt install ffmpeg` 等

---

## インストール

### 標準環境（macOS / Windows / Linux）

高速なパッケージマネージャ [uv](https://github.com/astral-sh/uv) の使用を推奨します。

```bash
uv pip install -r pyproject.toml
# 標準の pip を使う場合: pip install .
```

`pyproject.toml` がプラットフォーム条件で依存を自動的に切り替えます
（Mac: mlx-whisper + `audio-separator[cpu]` / Win・Linux: faster-whisper + `audio-separator[gpu]`）。

### Windows（GPU）での追加セットアップ

NVIDIA GPU で動かす場合、CUDA 対応版 PyTorch を別途入れてください（既定では CPU 版が入るため）。

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- `cu124` は CUDA 12.4 向け。環境の CUDA バージョンに合わせて選択してください。
- faster-whisper の CUDA 実行には cuDNN が必要です（多くの環境では PyTorch 同梱で動作）。
- GPU が無い／認識されない場合は自動で CPU 実行へフォールバックします。
- 話者分離（pyannote）・背景音除去（audio-separator）も CUDA を自動検出して使用します。

### DGX Spark (GB10 / Blackwell sm_121 / aarch64) — `requirements-spark.txt`

**この特定マシンだけは `pyproject.toml` を使えません。専用の [`requirements-spark.txt`](requirements-spark.txt) を使います。**

> **なぜ別ファイルなのか**
> GB10 (sm_121) は **CUDA cu129** でないと実行時に nvrtc JIT が落ちます。cu129 に上げると
> torchaudio が 2.8+ になり pyannote 3.x が壊れるため **pyannote-audio 4.x** が連鎖し、さらに
> huggingface-hub 1.x / transformers 最新化まで必要になります。これは `pyproject.toml`
> （torch<2.8 / pyannote<4.0 / hub<1.0 を想定した Mac・一般 Linux/Win 用）とは**真逆のスタック**で、
> 1 つの定義に同居できません。`requirements-spark.txt` は、この**動作確定済みの組み合わせを凍結し、
> ゼロから再現可能にする**ためのものです（GB10 環境を `uv sync` で壊さないための保険でもあります）。
>
> Spark では **`uv sync` / `uv pip install -e .` は使わない**でください。

```bash
ntut                                       # 大学プロキシ（新 SSH セッションごとに必須）
uv pip install -r requirements-spark.txt   # torch 3点は +cu129 でピン済み
```

詳細なインストール順序・検証コマンド・起動方法は [`requirements-spark.txt`](requirements-spark.txt) 冒頭のコメントに記載しています。
標準環境（x86_64 の Linux/Windows）では **このファイルは使わず** `pyproject.toml` を参照してください。

#### Mac から1コマンドで起動（Path 1）

「手元の Mac で操作 → 重い処理は Spark の GPU が実行」という運用を、Mac 側のスクリプト2本で行えます。
`app.py` 自体は **Spark 上で**動き、Mac はそれを起動してブラウザを開くだけです。

```bash
./spark-up.sh     # Spark で app.py を起動 → SSHトンネル確立 → ブラウザで http://127.0.0.1:8001/ を自動オープン
./spark-down.sh   # トンネルを閉じる（Spark の app.py も止めるなら ./spark-down.sh --stop-server）
```

- SSH パスワードは ControlMaster（接続多重化）で原則1回だけ。接続先は `SPARK_USER` / `SPARK_HOST` / `REMOTE_REPO` / `LOCAL_PORT` 等の環境変数で上書きできます。
- ブラウザに出るのは `http://127.0.0.1:8001` ですが、**文字起こし・話者分離・声紋照合・マッチングはすべて Spark の GPU で実行**されます。動画やテキストのアップロードはトンネル越しに Spark へ届きます。
- **声紋DB は Spark 上の GUI（[`/databases`](http://127.0.0.1:8001/databases)）で作成・管理**します（DB は Spark に常駐。Mac からの scp は不要）。
- 将来は Mac 側 `app.py` をプロキシ化して Spark を完全なヘッドレス API にする構成（大規模声紋DB向けの埋め込みキャッシュ含む）への拡張を予定しています。

---

## Hugging Face トークンの準備

pyannote のモデルを使うには Hugging Face アカウントとアクセストークンが必要です。
以下のページで利用規約に同意しておいてください。

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

プロジェクト直下に `.env` を作成し、トークンを記述します（`.env` は `.gitignore` 済み）。

```dotenv
HF_TOKEN=hf_ここにあなたのHuggingFaceトークン
```

### オフライン設定について

速度優先のため、モデル読み込み時のオンライン確認をスキップする設定（`HF_HUB_OFFLINE=1`）を既定で入れています。
その結果、モデルの自動更新は取得されません。更新を確認したいときや、**モデルキャッシュが空の環境（新規 Spark 等）で
初回ダウンロードを行うとき**は、`HF_HUB_OFFLINE=0` にして実行してください。

```bash
HF_HUB_OFFLINE=0 uv run python main.py 対象の会議録音.mp4
```

---

## 使い方（CLI / `main.py`）

声紋登録用の音声をひとつのディレクトリにまとめ、`--registry_dir` で指定します。
直下の対応音声（`.wav .mp3 .m4a .flac .mp4 .mov .ogg .opus .aac .wma`）を自動登録し、**話者名はファイル名（拡張子なし）**がそのまま使われます。

```bash
# 登録話者ありで実行（voiceprints/UserA.wav, UserB.wav → UserA, UserB として登録）
uv run python main.py 対象の会議録音.mp4 --registry_dir ./voiceprints

# 登録話者なし（分離＋文字起こしのみ。話者は SPEAKER_00 等の匿名クラスタID）
uv run python main.py 対象の会議録音.mp4
```

登録音声は前後 0.5 秒をノイズ排除のためトリミングしてから声紋抽出します（短すぎる場合はスキップ）。

### オプション引数

| 引数 | 説明 | デフォルト |
|---|---|---|
| `audio_file_path` | **[必須]** 処理対象の音声/動画ファイルのパス | (なし) |
| `--registry_dir` | 声紋登録用ディレクトリ（直下の対応音声を自動登録、話者名 = ファイル名 stem） | None |
| `--output_csv_path` | 出力 CSV のパス | 自動生成（現在時刻） |
| `--threshold` | 話者一致判定のしきい値。これより距離が遠いと Unknown 扱い | 0.5 |
| `--num_speakers` | 既知の総話者数（指定すると精度向上） | None（自動推定） |
| `--hf_token` | Hugging Face トークン（`.env` より優先） | `.env` の値 |
| `--embedding_model` | 話者照合に使うモデル | pyannote/embedding |
| `--mlx_model` | Whisper モデルID | mlx-community/whisper-large-v3-mlx |
| `--pyannote_model_id` | Diarization モデルID | pyannote/speaker-diarization-3.1 |

### 処理の仕組み（照合ロジック）

1. `--registry_dir` 内の各音声から声紋（埋め込みベクトル）を抽出・正規化してメモリに保持（各音声は前後 0.5 秒トリミング）。
2. 対象音声を pyannote で話者分離し、各クラスター（声のまとまり）で最も長い発話区間を抽出。
3. その区間から声紋を抽出し、登録済み声紋とのコサイン距離を計算、最近傍の登録話者を候補とする。
4. 候補との距離が `--threshold`（既定 0.5）以下ならその名前を割り当て、超えれば `Unknown_X` 扱い。

### 出力例（`result.csv`）

```csv
"start","end","speaker","text","cosine_distance"
"00:00:01:000","00:00:05:000","UserA","こんにちは、本日の会議を始めます。","0.123456"
"00:00:05:000","00:00:08:000","UserB","よろしくお願いします。","0.234567"
"00:00:09:000","00:00:12:000","Unknown_1","（※登録外の人の発言）音声テストです。",""
```

---

## Web UI（FastAPI + htmx）

CLI に加えて、音声 →（対応表）→ 字幕までを通しで行える Web UI を提供します。

```bash
uv run python app.py
# または: uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

起動後、ブラウザで <http://127.0.0.1:8000> にアクセスします。
（HTTPS への自動昇格で繋がらない場合は `http://` を手打ちしてください。）

UI は横並び 3 カード構成：

- **Step 1: 文字起こし & 話者分離** — 音声/動画 ＋ 声紋登録ファイル群（任意）から文字起こし SRT を生成（本文は `[speaker] text` 形式）。内部で `main.py` と同じ `AudioProcessor` を呼び出し、サイドカー CSV も `temp/` に保存（Unknown ラベル付け用）。詳細設定でしきい値・話者数・モデル名・HF_TOKEN・denoise を指定可能。
- **Step 2: マッチング（対応表作成）** — 台本（`.csv`）または手動書き起こしテキスト（`.txt`）＋ Step 1 の SRT から、SentenceTransformer + FAISS + WLIS で対応表 CSV を生成（`subtitle_matcher.py`）。`.txt` は `話者:本文` / `（ト書き）` / `# 場面` を解釈して台本 CSV に自動変換。
- **Step 3: 字幕生成** — 編集済み対応表 CSV から字幕 SRT を生成（`subtitle_exporter.py`）。

Step 1 → Step 2 はジョブ ID で連携し、生成物は `temp/` 配下に保存されます（再起動しても残ります）。
文字起こしは同期処理のため、長尺で uvicorn のタイムアウトに当たる場合は `--timeout-keep-alive` 等を調整してください。

### 声紋データベース（`voice_databases/`）

Web UI では声紋 DB を `voice_databases/<DB名>/<話者名>.wav` の構造で管理し、GUI から作成・話者追加・リネーム・トリミング・削除ができます（[`/databases`](http://127.0.0.1:8000/databases) ページ）。

- このディレクトリは **`.gitignore` 済み**（音声を Git/GitHub に上げないため）。`git clone` には含まれません。
- 中身はサーバーのディスクに永続します（`git pull`・再起動では消えません）。GUI の削除は即時・不可逆です。
- **別サーバーへ移植**したいときは、Git ではなく `scp` で直接コピーします（SSH 暗号化・外部を経由しない）:

  ```bash
  scp -r voice_databases user@new-server:~/path/to/audio-diarization-transcript/
  ```

### 背景音除去（denoise）

Step 1 の詳細設定で BGM/背景音の除去モードを選べます。

| モード | モデル | 実行系 | 用途 |
|---|---|---|---|
| `off` | （なし） | — | BGM が無い音声向け。Whisper はノイズ耐性が高く、まず `off` 推奨 |
| `fast` | Kim_Vocal_2.onnx | onnxruntime | 穏やかな除去。`off` で物足りないとき |
| `high` | mel_band_roformer | PyTorch (GPU) | 攻撃的。音楽のリード/バック分離向けで、トーク音声では用途違いになりがち |

> Spark (aarch64) では `onnxruntime-gpu` の wheel が無いため `fast`（ONNX）は CPU 実行になります。
> `high`（PyTorch）は cu129 で GPU 実行されます。

---

## CLI コマンド版（個別実行）

```bash
uv run python subtitle_matcher.py 台本.csv 音声認識.srt 対応表.csv
uv run python subtitle_exporter.py 対応表.csv 字幕.srt
```

台本 CSV の形式:

```csv
type,speaker,contents
dialogue,話者A,こんにちは、世界
dialogue,話者B,お疲れ様です
```
