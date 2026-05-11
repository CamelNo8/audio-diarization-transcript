話者照合 ＋ 文字起こし 統合スクリプト (Audio Diarization & Transcription)

このプロジェクトは、音声・動画ファイルから文字起こしを行い、さらに**「事前に登録した話者の声」と照合して「誰が話したか」を特定・CSV出力**するPythonスクリプトです。

Pyannoteによる話者分離（Diarization）と埋め込み抽出（Embedding）、およびApple Silicon（Mac）に最適化された mlx-whisper による高速な文字起こしを組み合わせています。

主な機能

文字起こし: mlx-whisper を使用した高速・高精度な日本語の文字起こし。

話者分離 (Diarization): pyannote/speaker-diarization-3.1 を使用し、音声内で人が話している区間を検出。

話者照合・特定: pyannote/embedding を使用。事前に登録した音声ファイルの声紋特徴量を抽出し、文字起こしされた音声セグメントとコサイン距離で比較（閾値による判定付きの最近傍検索）。一致した話者の名前を付与します。

動作環境・必須要件

OS: macOS (Apple Silicon M1/M2/M3 等推奨。mlx-whisper のため)

Python: 3.9 以上

FFmpeg: システムにインストールされていること（音声の前処理に使用します）

Macの場合: brew install ffmpeg

インストール

リポジトリの準備
このディレクトリに移動します。

依存パッケージのインストール
高速なパッケージマネージャ uv の使用を推奨します。

uv pip install -r pyproject.toml

(標準の pip を使用する場合は pip install . を実行してください)

Hugging Face トークンの準備
Pyannote のモデル（speaker-diarization-3.1 など）を使用するには、Hugging Faceのアカウントとアクセストークンが必要です。
また、以下のページで利用規約に同意しておく必要があります。

pyannote/speaker-diarization-3.1

pyannote/segmentation-3.0

環境変数の設定
ディレクトリ内に .env ファイルを作成し、取得したトークンを記述します。

HF*TOKEN=hf*ここにあなたのHuggingFaceトークンを記述

使い方

基本的には main.py を実行します。

基本的な実行（登録話者がいる場合）

処理対象の音声ファイルの後に、--registry オプションで「登録名=音声ファイルへのパス」を指定します（複数指定可能）。

uv run python main.py 対象の会議録音.mp4 \
 --registry UserA=UserAのサンプル音声.wav UserB=UserBのサンプル音声.wav

登録話者なしで実行（分離と文字起こしのみ）

登録話者を指定しない場合は、自動的に割り振られたクラスターID（例: SPEAKER_00）が出力されます。

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

--registry

登録音声。名前=ファイルパス の形式で複数指定可能

[]

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

--registry で渡された各音声から、声紋（埋め込みベクトル）を抽出・正規化しメモリに保持します。

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
