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

文字起こしバックエンドは実行環境に応じて自動で切り替わります（`src/transcription/`）。
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

#### 推論 API としてだけ使う（Path 2 / 未使用）

Spark を GUI 無しの推論 API として動かす経路も用意してあります（現在の運用は上の Path 1）。

```bash
# Spark 側
uv run uvicorn src.spark.server:app --host 0.0.0.0 --port 8000

# PC 側（動作確認）
uv run python -m src.spark.client 音声.wav 3
```

話者照合（声紋DB）と GUI は PC 側に残す設計で、サーバーは cluster_id 付きセグメントと
照合用の処理済み WAV だけを返します。接続先は環境変数 `SPARK_URL` で指定します。

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
- **Step 2: マッチング（対応表作成）** — 台本（`.csv`）または手動書き起こしテキスト（`.txt`）＋ Step 1 の SRT から、SentenceTransformer + FAISS + WLIS で対応表 CSV を生成（`src/subtitle/`）。`.txt` は `話者:本文` / `（ト書き）` / `# 場面` を解釈して台本 CSV に自動変換。
- **Step 3: 字幕生成** — 編集済み対応表 CSV から字幕 SRT を生成（`src/subtitle/exporter.py`）。

Step 1 → Step 2 はジョブ ID で連携し、生成物は `temp/` 配下に保存されます（再起動しても残ります）。

#### Step 1 は非同期ジョブ

文字起こしはアップロード完了時点で受け付けだけ返し、処理は裏で走ります。画面は 2 秒ごとに
進捗を取りに行き、処理ログをその場に流しながら、完了すると結果パネルへ切り替わります。

- **長尺でも接続が切れません。** 処理中も他の画面（声紋DB管理など）は通常どおり開けます。
- ジョブの状態は `temp/clusters/<job_id>/job.json` に持ちます。**メモリには載せていない**ので、
  ページを再読み込みしても進捗を追えます。
- **ページを閉じても処理は続きます**が、結果パネルはその画面にしか出ません。
  生成物は `temp/` に残るので、閉じてしまった場合は `/download/<ファイル名>` から取得できます。
- サーバーを止めると処理も止まります。状態が `running` のまま残ったジョブは、
  実行し直してください（途中から再開する仕組みはありません）。

> **uvicorn は 1 ワーカーで起動してください。** ジョブの状態はファイルで共有しているので
> 複数ワーカーでも壊れませんが、重いモデルがワーカーの数だけメモリに載ります。

### 声紋データベース（`voice_databases/`）

Web UI では声紋 DB を `voice_databases/<DB名>/<話者名>.wav` の構造で管理し、GUI から作成・話者追加・リネーム・トリミング・削除ができます（[`/databases`](http://127.0.0.1:8000/databases) ページ）。

- このディレクトリは **`.gitignore` 済み**（音声を Git/GitHub に上げないため）。`git clone` には含まれません。
- 中身はサーバーのディスクに永続します（`git pull`・再起動では消えません）。
- **別サーバーへ移植**したいときは、Git ではなく `scp` で直接コピーします（SSH 暗号化・外部を経由しない）:

  ```bash
  scp -r voice_databases user@new-server:~/path/to/audio-diarization-transcript/
  ```

#### 削除とゴミ箱（`voice_databases/.trash/`）

GUI からの削除は**その場では消さず**、`voice_databases/.trash/` へ退避します。
確認ダイアログを押し間違えても戻せるようにするためです。

```
voice_databases/
├── 会議A/                              # 通常のDB
└── .trash/
    ├── 20260728-143000_会議B/          # DB ごと削除したもの
    └── 20260728-143512_会議A_太郎.wav  # 話者1件を削除したもの
```

- **戻すとき**は、`.trash/` の中身を元の名前に直して `voice_databases/` 直下へ移すだけです。

  ```bash
  mv voice_databases/.trash/20260728-143000_会議B voice_databases/会議B
  ```

- `.trash/` は DB 一覧には出ません。DB 名としても指定できません。
- **自動削除はしません。** 削除してもディスク使用量は減らないので、
  溜まってきたら手で空にしてください。

  ```bash
  rm -rf voice_databases/.trash/*
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
uv run python -m src.subtitle.matcher 台本.csv 音声認識.srt 対応表.csv
uv run python -m src.subtitle.exporter 対応表.csv 字幕.srt
```

台本 CSV の形式:

```csv
type,speaker,contents
dialogue,話者A,こんにちは、世界
dialogue,話者B,お疲れ様です
```

---

## プロジェクト構成

エントリポイントはルートに置き、実体は `src/` 配下に機能ごとのパッケージとして持ちます。

```
audio-diarization-transcript/
├── main.py                     # CLI のエントリポイント（実体は src/cli/）
├── app.py                      # Web のエントリポイント（実体は src/web/）
├── spark-up.sh / -down.sh / -logs.sh   # Spark 上の app.py を Mac から操作する
├── src/
│   ├── config.py               # 定数・既定値・パス（しきい値/モデル名/出力名）
│   ├── common/                 # 複数機能から使う共通処理
│   │   ├── audio.py            #   ffmpeg 呼び出し（切り出し・16kHz mono 変換）
│   │   ├── csv_io.py           #   BOM 付き UTF-8 CSV の読み書き
│   │   ├── json_io.py          #   JSON のアトミックな読み書き（ジョブ状態用）
│   │   ├── files.py            #   一時ファイルの後始末
│   │   ├── filenames.py        #   出力ファイル名の検証（パストラバーサル対策）
│   │   ├── logging.py          #   ロギング設定と get_logger
│   │   └── timecode.py         #   時刻文字列 ⇔ 秒の相互変換
│   ├── cli/runner.py           # CLI 本体（引数解析 → AudioProcessor 起動）
│   ├── diarization/            # 話者分離・話者照合
│   │   ├── processor.py        #   AudioProcessor（全体の司令塔）
│   │   ├── pipeline.py         #   pyannote パイプラインの読み込み・実行
│   │   ├── vocals.py           #   背景音除去（audio-separator）
│   │   ├── clusters.py         #   クラスタの代表区間・未知クラスタの永続化
│   │   ├── speaker_identifier.py   # 埋め込み抽出とコサイン距離による照合
│   │   ├── embedding_cache.py  #   話者エンベディングの永続キャッシュ
│   │   ├── registry.py         #   声紋登録ファイルの収集・照合器のキャッシュ
│   │   ├── transcript.py       #   文字起こし結果 → CSV
│   │   ├── interactive.py      #   CLI での対話的な未知話者解決
│   │   └── torch_compat.py     #   torch のバージョン差異の吸収
│   ├── transcription/          # Whisper バックエンドの切り替え
│   │   ├── backend.py          #   バックエンド選択
│   │   ├── model_ids.py        #   モデルID の表記変換
│   │   ├── mlx.py / faster_whisper.py / transformers_backend.py
│   ├── subtitle/               # 台本と音声認識の対応付け
│   │   ├── loader.py           #   台本CSV / SRT の読み込み
│   │   ├── ngram.py            #   テキスト正規化と N-gram 生成
│   │   ├── embedding.py        #   SentenceTransformer + FAISS 近傍探索
│   │   ├── alignment.py        #   候補ペア生成と WLIS
│   │   ├── report.py           #   マッチ結果の CSV 出力・集計表示
│   │   ├── matcher.py          #   マッチングの入口（別プロセスで実行される）
│   │   └── exporter.py         #   対応表CSV → 字幕SRT
│   ├── voice_db/registry.py    # 声紋データベース（作成・話者の追加/削除/改名）
│   ├── web/                    # FastAPI + htmx の Web アプリ層
│   │   ├── application.py      #   ルータ登録（create_app）
│   │   ├── templating.py       #   Jinja2 環境とエラー表示
│   │   ├── storage.py          #   temp/ への保存とパス解決
│   │   ├── jobs.py             #   ジョブの発行・永続化・進捗・CSV再ラベル
│   │   ├── converters.py       #   文字起こしCSV→SRT、テキスト→台本CSV
│   │   ├── identification.py   #   HFトークン解決・照合器の生成
│   │   ├── log_capture.py      #   処理ログの捕捉（進捗表示用）
│   │   ├── forms.py / errors.py
│   │   └── routes/             #   pages / transcription / matching /
│   │                           #   generation / databases / unknowns
│   └── spark/                  # DGX Spark へ推論をオフロードする API（Path 2）
│       ├── server.py           #   Spark 側で動かす FastAPI サーバー
│       └── client.py           #   PC 側から呼ぶクライアント
├── templates/                  # Jinja2 テンプレート（partials/ は htmx の差し替え片）
├── tests/                      # pytest（重いモデルはモック）
├── tools/                      # 調査用の使い捨てスクリプト
├── temp/                       # 作業ディレクトリ（.gitignore 済み）
└── voice_databases/            # 声紋DB（.gitignore 済み。.trash/ が削除済みの退避先）
```

---

## 開発

テストとリンタは開発用の依存（`pytest` / `httpx` / `ruff`）を入れてから実行します。

```bash
uv sync --group dev
# 標準の pip を使う場合: pip install pytest httpx ruff
```

```bash
uv run pytest -q                  # テスト（重いモデルとネットワークはモック）
uv run pytest tests/test_web_transcription.py -q   # ファイル単位
uv run ruff check .               # リンタ
uv run ruff format .              # フォーマッタ
```

- テストは**重いモデル（pyannote / Whisper / SentenceTransformer / audio-separator）を読み込みません**。
  ロジック層とルーティングだけを検証するため、GPU も HF トークンも不要です。
- テスト名は日本語で「何がどうなるか」を書いています。
- Web ルートのテストで差し替えるモジュールは [`tests/conftest.py`](tests/conftest.py) に集約しています。
  実装の置き場所を動かしたときは、まずここを直してください。
