話者照合 ＋ 文字起こし ＋ 字幕生成 統合アプリ (Audio Diarization & Subtitle)

このプロジェクトは、音声・動画ファイルから文字起こしを行い、さらに**「事前に登録した話者の声」と照合して「誰が話したか」を特定・CSV出力**するPythonスクリプトです。
加えて、台本CSVと文字起こし結果をマッチングして **対応表CSV → 字幕SRT** を生成する機能を Streamlit Web UI で統合しています。

Pyannoteによる話者分離（Diarization）と埋め込み抽出（Embedding）、および環境に応じた Whisper 実装（Mac: mlx-whisper / Windows・Linux: faster-whisper）による高速な文字起こしを組み合わせています。

主な機能

文字起こし: 高速・高精度な日本語の文字起こし。実行環境に応じて Mac では mlx-whisper、Windows/Linux では faster-whisper（CUDA 対応）を自動選択。

話者分離 (Diarization): pyannote/speaker-diarization-3.1 を使用し、音声内で人が話している区間を検出。

話者照合・特定: pyannote/embedding を使用。あらかじめ用意した声紋登録用ディレクトリ内の音声ファイルから声紋特徴量を抽出し（前後0.5秒をノイズ排除のためにトリミング）、文字起こしされた音声セグメントとコサイン距離で比較（閾値による判定付きの最近傍検索）。一致した話者の名前を付与します。

動作環境・必須要件

OS:
- macOS (Apple Silicon M1/M2/M3 等): 文字起こしに mlx-whisper を使用
- Windows / Linux: 文字起こしに faster-whisper を使用（NVIDIA GPU があれば CUDA で高速化、無ければ CPU で動作）

文字起こしバックエンドは実行環境に応じて自動で切り替わります（transcription_backend.py）。
ユーザー側でモデルID（mlx-community/whisper-large-v3-mlx 等）はそのまま指定でき、
Windows/Linux では faster-whisper 用のサイズ名（large-v3 等）へ自動変換されます。

Python: 3.10 以上

FFmpeg: システムにインストールされていること（音声の前処理に使用します）

Macの場合: brew install ffmpeg
Windowsの場合: winget install Gyan.FFmpeg もしくは https://www.gyan.dev/ffmpeg/builds/ から取得し PATH を通す

インストール

リポジトリの準備
このディレクトリに移動します。

依存パッケージのインストール
高速なパッケージマネージャ uv の使用を推奨します。

uv pip install -r pyproject.toml

(標準の pip を使用する場合は pip install . を実行してください)

Windows（GPU）での追加セットアップ
NVIDIA GPU で動作させる場合、CUDA 対応版の PyTorch を別途インストールしてください
（pyproject.toml の torch は既定で CPU 版が入るため）。

uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

- 上記の cu124 は CUDA 12.4 向け。環境の CUDA バージョンに合わせて選択してください。
- faster-whisper の CUDA 実行には cuDNN が必要です。多くの環境では上記 PyTorch 同梱の
  ライブラリで動作しますが、cuDNN 関連のエラーが出る場合は NVIDIA cuDNN を導入してください。
- GPU が無い / 認識されない場合は自動的に CPU 実行へフォールバックします（処理は遅くなります）。
- 話者分離（pyannote）も CUDA があれば自動的に GPU を使用します。
- 背景音除去（audio-separator）も GPU を自動検出して使用します。
  - Windows/Linux: onnxruntime-gpu の CUDAExecutionProvider（`audio-separator[gpu]`）。
  - Mac (Apple Silicon): CoreML + torch MPS（Metal）を自動使用します（追加設定不要）。

Hugging Face トークンの準備
Pyannote のモデル（speaker-diarization-3.1 など）を使用するには、Hugging Faceのアカウントとアクセストークンが必要です。
また、以下のページで利用規約に同意しておく必要があります。

pyannote/speaker-diarization-3.1

pyannote/segmentation-3.0

環境変数の設定
ディレクトリ内に .env ファイルを作成し、取得したトークンを記述します。

HF*TOKEN=hf*ここにあなたのHuggingFaceトークンを記述

Hugging Face サーバー確認のスキップについて
実行時の速度を優先するため、モデル読み込み時のサーバー確認（オンライン問い合わせ）をスキップする設定を入れています。
その結果、モデルの更新が自動取得されない点に注意してください。
更新を確認したい場合は、月1回程度、環境変数を一時的に HF_HUB_OFFLINE=0 にして実行してください。
例:
HF_HUB_OFFLINE=0 uv run python main.py 対象の会議録音.mp4

使い方

基本的には main.py を実行します。

基本的な実行（登録話者がいる場合）

声紋登録用の音声ファイルをひとつのディレクトリにまとめておき、--registry_dir でそのディレクトリを指定します。
ディレクトリ直下の対応音声ファイル（.wav, .mp3, .m4a, .flac, .mp4, .mov, .ogg, .opus, .aac, .wma）を全て自動登録します。
話者名はファイル名（拡張子を除く部分）がそのまま使われます。

例: voiceprints/ に UserA.wav, UserB.wav, ... を入れておけば、UserA, UserB として登録されます。

uv run python main.py 対象の会議録音.mp4 --registry_dir ./voiceprints

なお、登録音声は前後0.5秒をノイズ排除のためにトリミングしてから声紋抽出します。
（音声長が短すぎる場合はトリミングをスキップして元音声をそのまま使います）

登録話者なしで実行（分離と文字起こしのみ）

--registry_dir を指定しない場合は、自動的に割り振られたクラスターID（例: SPEAKER_00）が出力されます。

uv run python main.py 対象の会議録音.mp4

オプション引数

コマンドラインから様々なカスタマイズが可能です。

uv run python main.py 対象音声.mp4 [オプション]

引数

説明

デフォルト値

audio_file_path

[必須] 処理対象の音声/動画ファイルのパス

(なし)

--registry_dir

声紋登録用音声ファイルを格納したディレクトリ。直下の対応音声ファイル全てを自動登録（話者名はファイル名 stem）

None

--output_csv_path

出力するCSVファイルのパス（省略時は現在時刻で自動生成）

自動生成

--threshold

話者一致判定のしきい値。これより距離が遠いと Unknown 扱い

0.5

--num_speakers

音声内の既知の総話者数（わかっている場合指定すると精度向上）

None (自動推定)

--hf_token

Hugging Face トークン（.env より優先されます）

.env の値

--embedding_model

話者照合に使用するモデル

pyannote/embedding

--mlx_model

WhisperモデルID

mlx-community/whisper-large-v3-mlx

--pyannote_model_id

DiarizationモデルID

pyannote/speaker-diarization-3.1

処理の仕組み（照合ロジック）

--registry_dir で指定されたディレクトリ内の各音声から、声紋（埋め込みベクトル）を抽出・正規化しメモリに保持します。各音声は前後0.5秒をノイズ排除のためにトリミングしてから処理します。

対象音声を Pyannote で話者分離し、各クラスター（声のまとまり）の中で最も長い発話区間を抽出します。

その区間の音声から声紋を抽出し、事前登録された声紋とコサイン距離を計算します。

最も距離が近い（最近傍の）登録話者を候補とします。

その候補との距離が --threshold (デフォルト0.5) 以下であればその名前を割り当て、超えていれば Unknown_X として扱います。

出力

実行が完了すると、指定したパス（または自動生成されたファイル名）にCSVが出力されます。

出力例 (result.csv):

"start","end","speaker","text","cosine_distance"
"00:00:01:000","00:00:05:000","UserA","こんにちは、本日の会議を始めます。","0.123456"
"00:00:05:000","00:00:08:000","UserB","よろしくお願いします。","0.234567"
"00:00:09:000","00:00:12:000","Unknown_1","（※登録外の人の発言）音声テストです。",""

FastAPI + htmx Web UI（音声→対応表→字幕の統合）

CLI に加えて、FastAPI + htmx による Web UI が利用できます。

起動方法:

uv run python app.py

または:

uv run uvicorn app:app --host 127.0.0.1 --port 8000

起動後、ブラウザで http://127.0.0.1:8000 にアクセスしてください。

UI は 3 セクション構成（横並びカード）:

- **Step 1: 文字起こし & 話者分離** — 音声/動画 + 声紋登録ファイル群（任意）から文字起こし SRT を生成（本文は `[speaker] text` 形式）。内部的に `main.py` と同じ処理（`AudioProcessor`）を呼び出し、サイドカー CSV も `temp/` に保存します（Unknown ラベル付け用）。詳細設定 (`<details>`) でしきい値・話者数・モデル名・HF_TOKEN 上書きが可能。
- **Step 2: マッチング（対応表作成）** — 台本 (`.csv`) または手動書き起こしテキスト (`.txt`) ＋ Step 1 の SRT から、SentenceTransformer + FAISS + WLIS による対応表 CSV を生成（`subtitle_matcher.py`）。`.txt` は `id,scene_id,type,speaker,contents` 形式の台本 CSV に自動変換されます（`話者:本文` / `（ト書き）` / `# 場面` の各形式に対応）。
- **Step 3: 字幕生成** — 編集済み対応表 CSV から字幕 SRT を生成（`subtitle_exporter.py`）。

Step 1 → Step 2 の連携は **完全ステートレス**（Streamlit 版のようなセッション保持はしません）。Step 1 で生成した SRT をダウンロードし、Step 2 で「音声認識 SRTファイル」にアップロードしてください。

なお、文字起こしは **同期処理**（リクエスト中ブラウザは htmx インジケータで「処理中…」を表示）です。長尺音声で uvicorn 側のタイムアウトに当たる場合は、`--timeout-keep-alive` 等の調整を検討してください。

アップロード・生成物は `temp/` 配下に保存されます（再起動しても残ります）。

台本CSVの形式:

type,speaker,contents
dialogue,話者A,こんにちは、世界
dialogue,話者B,お疲れ様です

CLI コマンド版（個別実行）:

uv run python subtitle_matcher.py 台本.csv 音声認識.srt 対応表.csv
uv run python subtitle_exporter.py 対応表.csv 字幕.srt
