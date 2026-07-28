"""プロジェクト全体で共有する定数・既定値。

閾値・モデル名・パスはここに集約する（規約4.5）。
モジュール固有の値は各モジュール側に置く。
"""

from __future__ import annotations

from pathlib import Path

# ===================================================================
# パス
# ===================================================================
#: リポジトリのルート（このファイルは <root>/src/config.py にある）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Web UI の作業ディレクトリ（アップロード音声・生成物の置き場）
TEMP_DIR = PROJECT_ROOT / "temp"

#: 未知話者クラスタのジョブ保存先
CLUSTERS_ROOT = TEMP_DIR / "clusters"

#: Jinja2 テンプレートの置き場
TEMPLATES_DIR = PROJECT_ROOT / "templates"

#: 声紋データベースの既定ルート（環境変数 VOICE_DB_ROOT で上書き可）
DEFAULT_VOICE_DB_ROOT = PROJECT_ROOT / "voice_databases"

#: 話者エンベディング永続キャッシュの既定ルート
#: （環境変数 EMBEDDING_CACHE_DIR で上書き可）
DEFAULT_EMBEDDING_CACHE_DIR = PROJECT_ROOT / "embedding_cache"

# ===================================================================
# 音声フォーマット
# ===================================================================
#: pyannote / Whisper へ渡す音声のサンプリングレート
TARGET_SAMPLE_RATE_HZ = 16000

#: 同上（モノラル）
TARGET_CHANNELS = 1

# ===================================================================
# 話者分離・話者照合
# ===================================================================
#: 同一話者と判定するコサイン距離の既定閾値
DEFAULT_SPEAKER_THRESHOLD = 0.5

#: 話者エンベディング抽出モデル
DEFAULT_EMBEDDING_MODEL = "pyannote/embedding"

#: 話者分離（ダイアライゼーション）パイプライン
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

#: 登録音声の前後を切り落とす秒数（無音・言い出しのブレを除くため）
REGISTRATION_TRIM_SEC = 0.5

#: トリミング後に最低限残したい音声長（秒）
REGISTRATION_MIN_REMAINING_SEC = 0.5

#: クラスタの代表区間として優先する最短の長さ（秒）。
#: これ未満しか無いクラスタは、その中の最長区間を使う。
MIN_REPRESENTATIVE_SEC = 1.0

#: 未登録話者に付ける仮ラベルの接頭辞。
UNKNOWN_LABEL_PREFIX = "Unknown_"

# ===================================================================
# 文字起こし
# ===================================================================
#: Whisper の既定モデル（Apple Silicon / mlx-whisper 向け表記）
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

#: 幻聴（ハルシネーション）とみなす定型フレーズ。
#: 無音・音楽だけの区間で Whisper が出力しがちな、学習データ由来の決まり文句。
#: 環境変数 HALLUCINATION_PHRASES_FILE で素材ごとに追記できる
#: （src/transcription/hallucination.py を参照）。
DEFAULT_HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご清聴ありがとうございました",
    "ご清聴ありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "最後までご視聴いただきありがとうございました",
    "チャンネル登録をお願いします",
    "チャンネル登録よろしくお願いします",
    "おわり",
    "END",
)

# ===================================================================
# 背景音 / BGM 除去
# ===================================================================
#: audio-separator の既定分離モデル
#: （HuggingFace の karaokenerds/all-uvr-models から取得）
DEFAULT_SEPARATOR_MODEL = "Kim_Vocal_2.onnx"

#: 除去の強度と audio-separator のモデルの対応。``None`` は除去しない。
#: 未知のキーも「除去しない」として扱う（``.get()`` が ``None`` を返すため）。
DENOISE_MODELS = {
    "off": None,
    "fast": DEFAULT_SEPARATOR_MODEL,
    "high": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
}

# ===================================================================
# 字幕マッチング
# ===================================================================
#: 台本・音声認識テキストをまとめる N-gram の最大長
NGRAM_MAX_N = 3

#: 近傍探索で1件あたり取得する候補数
SIMILAR_VECTOR_TOP_K = 10

#: 文埋め込みモデル（多言語 STS）
SENTENCE_EMBEDDING_MODEL = "stsb-xlm-r-multilingual"

# ===================================================================
# 名前の検証
# ===================================================================
#: DB名・話者名に使えない文字（ファイル名として不正になるもの）
INVALID_NAME_CHARS = set('/\\:*?"<>|')

# ===================================================================
# 出力ファイルの既定名
# ===================================================================
#: 文字起こしの字幕 SRT（Web の Step 1）
DEFAULT_TRANSCRIPTION_SRT_NAME = "transcription.srt"

#: 台本と音声認識の対応表 CSV（Web の Step 2）
DEFAULT_MATCHING_CSV_NAME = "対応表.csv"

#: 編集済み対応表から生成する字幕 SRT（Web の Step 3）
DEFAULT_SUBTITLE_SRT_NAME = "subtitles.srt"
