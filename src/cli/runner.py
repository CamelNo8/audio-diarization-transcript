"""コマンドラインからの文字起こし実行。

``main.py`` から呼ばれる CLI 本体。引数の解析と、話者照合の準備、
:class:`~src.diarization.processor.AudioProcessor` の起動を行う。
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src.common.logging import get_logger
from src.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SPEAKER_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
)
from src.diarization.processor import AudioProcessor, create_transcript_csv_path
from src.diarization.registry import (
    collect_registry_files,
    get_cached_speaker_identifier,
)

logger = get_logger(__name__)

#: 正常終了 / 異常終了の終了コード。
EXIT_OK = 0
EXIT_ERROR = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Args:
        argv: 引数のリスト。省略時は ``sys.argv`` を使う。
    """
    parser = argparse.ArgumentParser(
        description="話者照合機能付き文字起こし＆話者分離スクリプト"
    )
    _add_io_arguments(parser)
    _add_model_arguments(parser)
    return parser.parse_args(argv)


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """入出力と話者照合の条件に関する引数を追加する。"""
    parser.add_argument(
        "audio_file_path", type=Path, help="文字起こし対象の音声/動画ファイルのパス"
    )
    parser.add_argument(
        "--registry_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "声紋登録用音声ファイルを格納したディレクトリ。"
            "ディレクトリ内の対応音声ファイルを全て自動登録します"
            "（話者名はファイル名 stem）"
        ),
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
        default=DEFAULT_SPEAKER_THRESHOLD,
        help="話者一致判定のしきい値。小さいほど厳格（デフォルト: 0.5）",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="音声内の既知の話者数。未指定の場合はモデルが自動推定します。",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """モデルと認証に関する引数を追加する。"""
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face アクセストークン (未指定時は環境変数 HF_TOKEN を使用)",
    )
    parser.add_argument(
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="話者照合に使用するモデル名",
    )
    parser.add_argument(
        "--mlx_model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=(
            "Whisper モデルID/品質"
            "（例: large-v3, medium, small。エンジンに合わせ自動変換）"
        ),
    )
    parser.add_argument(
        "--whisper_backend",
        type=str,
        choices=["auto", "mlx", "faster"],
        default="auto",
        help=(
            "文字起こしエンジン。auto=OS判定 / mlx=mlx-whisper / faster=faster-whisper"
        ),
    )
    parser.add_argument(
        "--pyannote_model_id",
        type=str,
        default=DEFAULT_DIARIZATION_MODEL,
        help="Pyannote Diarization のモデルID",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 本体。

    Args:
        argv: コマンドライン引数（プログラム名を除く）。

    Returns:
        終了コード（:data:`EXIT_OK` / :data:`EXIT_ERROR`）。
    """
    if not shutil.which("ffmpeg"):
        logger.critical(
            "Critical Error: ffmpeg is required but not found in PATH. "
            "Please install FFmpeg."
        )
        return EXIT_ERROR

    args = parse_args(argv)
    if not _validate_args(args):
        return EXIT_ERROR

    output_csv_path = args.output_csv_path
    if output_csv_path is None:
        try:
            output_csv_path = create_transcript_csv_path(args.audio_file_path)
            logger.info(f"Output CSV path defaulting to: {output_csv_path}")
        except Exception as e:
            logger.critical(f"Critical Error: CSVパスの自動生成に失敗しました: {e}")
            return EXIT_ERROR

    logger.info("スクリプトの実行を開始します...")

    try:
        identifier = _prepare_identifier(args)
    except Exception as exc:
        logger.critical(f"話者照合モジュールの初期化エラー: {exc}", exc_info=True)
        return EXIT_ERROR

    return _run(args, output_csv_path, identifier)


def _validate_args(args: argparse.Namespace) -> bool:
    """実行に必要な前提が揃っているかを確認する。"""
    if not args.audio_file_path.is_file():
        logger.critical(
            f"Critical Error: 対象の音声ファイルが見つかりません {args.audio_file_path}"
        )
        return False
    if not args.hf_token:
        logger.critical(
            "Hugging Face トークンが設定されていません。"
            "--hf_token または .env に HF_TOKEN を設定してください。"
        )
        return False
    return True


def _prepare_identifier(args: argparse.Namespace):
    """声紋登録ディレクトリがある場合だけ識別器を用意し、登録話者を読み込む。"""
    if args.registry_dir is None:
        logger.info(
            "--registry_dir オプションが指定されていないため、"
            "話者の特定（名前の割り当て）はスキップします。"
        )
        return None

    registry_paths = collect_registry_files(args.registry_dir)
    identifier = get_cached_speaker_identifier(
        model_name=args.embedding_model,
        hf_token=args.hf_token,
        threshold=args.threshold,
    )
    logger.info(
        f"登録話者の特徴量を抽出しています... "
        f"({len(registry_paths)} 名: {args.registry_dir})"
    )
    for registered_name, path in registry_paths.items():
        identifier.register_speaker(registered_name, path)
    return identifier


def _run(args: argparse.Namespace, output_csv_path: Path, identifier) -> int:
    """音声処理を実行し、終了コードを返す。"""
    try:
        with AudioProcessor(
            audio_file=args.audio_file_path,
            output_csv_path=output_csv_path,
            mlx_model_id=args.mlx_model,
            pyannote_model_id=args.pyannote_model_id,
            hf_token=args.hf_token,
            identifier=identifier,
            registry_dir=args.registry_dir,
            whisper_backend=args.whisper_backend,
        ) as processor:
            if processor.process_and_save_to_csv(known_num_speakers=args.num_speakers):
                logger.info(f"Processing complete. Results saved to {output_csv_path}")
                return EXIT_OK
            logger.error("Processing failed. Please check the logs above for details.")
            return EXIT_ERROR
    except Exception as e:
        logger.critical(f"実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return EXIT_ERROR
