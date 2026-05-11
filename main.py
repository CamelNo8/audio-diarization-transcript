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

from speaker_identification import SpeakerIdentifier
from audio_processor import AudioProcessor, create_transcript_csv_path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="話者照合機能付き文字起こし＆話者分離スクリプト"
    )
    parser.add_argument(
        "audio_file_path", type=Path, help="文字起こし対象の音声/動画ファイルのパス"
    )
    parser.add_argument(
        "--registry",
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help="登録音声。形式は '名前=ファイルパス' (複数指定可)",
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


def parse_registry_entries(entries: list[str]) -> dict[str, Path]:
    """NAME=PATH 形式の引数を辞書に変換する"""
    parsed: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"registry の形式が不正です: {entry} (NAME=PATH 形式で指定してください)")
        name, path = entry.split("=", 1)
        name, path_str = name.strip(), path.strip()
        
        if not name:
            raise ValueError(f"名前が空です: {entry}")
        if name in parsed:
            raise ValueError(f"重複した名前です: {name}")
            
        parsed[name] = Path(path_str)
    return parsed


def main() -> int:
    # ffmpegの存在チェック
    if not shutil.which("ffmpeg"):
        logging.critical("Critical Error: ffmpeg is required but not found in PATH. Please install FFmpeg.")
        return 1

    args = parse_args()

    audio_file_path = args.audio_file_path
    output_csv_path = args.output_csv_path
    
    if not audio_file_path.is_file():
        logging.critical(f"Critical Error: 対象の音声ファイルが見つかりません {audio_file_path}")
        return 1
    
    if not args.hf_token:
        logging.critical("Hugging Face トークンが設定されていません。--hf_token または .env に HF_TOKEN を設定してください。")
        return 1

    if output_csv_path is None:
        try:
            output_csv_path = create_transcript_csv_path(audio_file_path)
            logging.info(f"Output CSV path defaulting to: {output_csv_path}")
        except Exception as e:
            logging.critical(f"Critical Error: CSVパスの自動生成に失敗しました: {e}")
            return 1

    logging.info("スクリプトの実行を開始します...")

    identifier = None
    # 登録音声がある場合のみ話者照合機能（SpeakerIdentifier）を初期化
    if args.registry:
        try:
            registry_paths = parse_registry_entries(args.registry)
            identifier = SpeakerIdentifier(
                model_name=args.embedding_model,
                hf_token=args.hf_token,
                threshold=args.threshold
            )
            logging.info("登録話者の特徴量を抽出しています...")
            used_registered_names: set[str] = set()
            for alias, path in registry_paths.items():
                # 出力ラベルは登録ファイル名（拡張子除く）を使う
                registered_name = path.stem or alias
                if registered_name in used_registered_names:
                    raise ValueError(
                        f"登録ファイル名が重複しています: {registered_name}。"
                        "同名ファイルを避けるか、ファイル名を変更してください。"
                    )
                used_registered_names.add(registered_name)

                identifier.register_speaker(registered_name, path)
                logging.info(f"Registry alias '{alias}' -> speaker label '{registered_name}'")
                
        except Exception as exc:
            logging.critical(f"話者照合モジュールの初期化エラー: {exc}", exc_info=True)
            return 1
    else:
        logging.info("--registry オプションが指定されていないため、話者の特定（名前の割り当て）はスキップします。")

    # 音声の処理と文字起こしを実行
    try:
        with AudioProcessor(
            audio_file=audio_file_path,
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