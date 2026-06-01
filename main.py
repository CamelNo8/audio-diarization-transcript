from __future__ import annotations

import argparse
import os
import sys
import shutil
import logging
from pathlib import Path

# python-dotenv を使って環境変数 (.env) をロード
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv is not installed. Environment variables from .env will not be loaded automatically.")
    print("To install: uv pip install python-dotenv")

os.environ["HF_HUB_OFFLINE"] = "1"  # [OPTIMIZED]

from speaker_identification import SpeakerIdentifier
from audio_processor import AudioProcessor, create_transcript_csv_path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

_SPEAKER_IDENTIFIER_CACHE: dict[str, SpeakerIdentifier] = {}

# 登録音声ディレクトリで対象とする拡張子
SUPPORTED_REGISTRY_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mov", ".ogg", ".opus", ".aac", ".wma",
}


def get_cached_speaker_identifier(
    model_name: str, hf_token: str, threshold: float
) -> SpeakerIdentifier:
    cached = _SPEAKER_IDENTIFIER_CACHE.get(model_name)
    if cached is None:
        try:
            cached = SpeakerIdentifier(
                model_name=model_name,
                hf_token=hf_token,
                threshold=threshold,
            )
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "0"
            cached = SpeakerIdentifier(
                model_name=model_name,
                hf_token=hf_token,
                threshold=threshold,
            )
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
        _SPEAKER_IDENTIFIER_CACHE[model_name] = cached
    else:
        cached.threshold = threshold
        cached.registry_embeddings = {}
        cached.unknown_counter = 1
    return cached


def collect_registry_files(registry_dir: Path) -> dict[str, Path]:
    """声紋登録ディレクトリ内の音声ファイルを収集し、{話者名: パス} の辞書を返す。

    話者名はファイル名の stem を使用する。同名 stem が複数ある場合はエラー。
    """
    if not registry_dir.is_dir():
        raise NotADirectoryError(f"声紋登録ディレクトリが存在しません: {registry_dir}")

    parsed: dict[str, Path] = {}
    for path in sorted(registry_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_REGISTRY_EXTENSIONS:
            continue
        name = path.stem
        if not name:
            raise ValueError(f"ファイル名が空のため登録できません: {path}")
        if name in parsed:
            raise ValueError(
                f"登録ファイル名（stem）が重複しています: {name} "
                f"({parsed[name]} と {path})"
            )
        parsed[name] = path

    if not parsed:
        raise ValueError(
            f"声紋登録ディレクトリ内に対象音声ファイルが見つかりません: {registry_dir} "
            f"(対象拡張子: {sorted(SUPPORTED_REGISTRY_EXTENSIONS)})"
        )

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="話者照合機能付き文字起こし＆話者分離スクリプト"
    )
    parser.add_argument(
        "audio_file_path", type=Path, help="文字起こし対象の音声/動画ファイルのパス"
    )
    parser.add_argument(
        "--registry_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="声紋登録用音声ファイルを格納したディレクトリ。ディレクトリ内の対応音声ファイルを全て自動登録します（話者名はファイル名 stem）",
    )
    parser.add_argument(
        "--output_csv_path",
        type=Path,
        default=None,
        help="出力するCSVファイルのパス（省略時は自動生成）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="話者一致判定のしきい値。小さいほど厳格（デフォルト: 0.5）",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face アクセストークン (未指定時は環境変数 HF_TOKEN を使用)",
    )
    parser.add_argument(
        "--embedding_model",
        default="pyannote/embedding",
        help="話者照合に使用するモデル名",
    )
    parser.add_argument(
        "--mlx_model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="mlx-whisper のモデルID",
    )
    parser.add_argument(
        "--pyannote_model_id",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote Diarization のモデルID",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="音声内の既知の話者数。未指定の場合はモデルが自動推定します。",
    )
    return parser.parse_args()


def main() -> int:
    # ffmpegの存在チェック
    if not shutil.which("ffmpeg"):
        logging.critical("Critical Error: ffmpeg is required but not found in PATH. Please install FFmpeg.")
        return 1

    args = parse_args()

    output_csv_path = args.output_csv_path
    
    if not args.audio_file_path.is_file():
        logging.critical(f"Critical Error: 対象の音声ファイルが見つかりません {args.audio_file_path}")
        return 1
    
    if not args.hf_token:
        logging.critical("Hugging Face トークンが設定されていません。--hf_token または .env に HF_TOKEN を設定してください。")
        return 1

    if output_csv_path is None:
        try:
            output_csv_path = create_transcript_csv_path(args.audio_file_path)
            logging.info(f"Output CSV path defaulting to: {output_csv_path}")
        except Exception as e:
            logging.critical(f"Critical Error: CSVパスの自動生成に失敗しました: {e}")
            return 1

    logging.info("スクリプトの実行を開始します...")

    identifier = None
    # 声紋登録ディレクトリがある場合のみ話者照合機能（SpeakerIdentifier）を初期化
    if args.registry_dir is not None:
        try:
            registry_paths = collect_registry_files(args.registry_dir)
            identifier = get_cached_speaker_identifier(  # [OPTIMIZED]
                model_name=args.embedding_model,
                hf_token=args.hf_token,
                threshold=args.threshold,
            )
            logging.info(
                f"登録話者の特徴量を抽出しています... ({len(registry_paths)} 名: {args.registry_dir})"
            )
            for registered_name, path in registry_paths.items():
                identifier.register_speaker(registered_name, path)

        except Exception as exc:
            logging.critical(f"話者照合モジュールの初期化エラー: {exc}", exc_info=True)
            return 1
    else:
        logging.info("--registry_dir オプションが指定されていないため、話者の特定（名前の割り当て）はスキップします。")

    # 音声の処理と文字起こしを実行
    try:
        with AudioProcessor(
            audio_file=args.audio_file_path,
            output_csv_path=output_csv_path,
            mlx_model_id=args.mlx_model,
            pyannote_model_id=args.pyannote_model_id,
            hf_token=args.hf_token,
            identifier=identifier,
        ) as processor:
            
            success = processor.process_and_save_to_csv(known_num_speakers=args.num_speakers)

            if success:
                logging.info(f"Processing complete. Results saved to {output_csv_path}")
                return 0
            else:
                logging.error("Processing failed. Please check the logs above for details.")
                return 1

    except Exception as e:
        logging.critical(f"実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())