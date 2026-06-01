"""mlx_whisper の生の文字起こし結果 (クラスタリング前) の start/end が
ミリ秒単位で整数に丸められているかを確認するテスト。

実行例:
    python test_transcript_ms_integer.py 三上.wav
    python test_transcript_ms_integer.py 転スラ1話調整版.wav --model mlx-community/whisper-large-v3-mlx

判定方法:
    seconds * 1000 が整数 (1e-9 以下の誤差まで) かどうかを確認する。
    すべてのセグメントの start / end が整数ms ならば "整数に丸め込まれている" と判定。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlx_whisper


# float の表現誤差を吸収するための許容値
# 例: 1.234 * 1000 = 1233.9999999999998 のようなケースを整数とみなす
MS_INTEGER_TOLERANCE = 1e-6


def is_integer_ms(seconds: float, tolerance: float = MS_INTEGER_TOLERANCE) -> bool:
    ms = seconds * 1000.0
    return abs(ms - round(ms)) <= tolerance


def fractional_ms(seconds: float) -> float:
    ms = seconds * 1000.0
    return ms - math.floor(ms)


def run_transcription(audio_path: Path, model: str) -> list[dict]:
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        verbose=False,
        language="ja",
    )
    return result.get("segments", [])


def check_segments(segments: list[dict]) -> tuple[int, int, list[tuple[int, str, float]]]:
    """整数msのセグメント数、総セグメント数、整数でなかったセグメントの一覧を返す。"""
    integer_count = 0
    violations: list[tuple[int, str, float]] = []

    for idx, seg in enumerate(segments):
        for field in ("start", "end"):
            value = float(seg[field])
            if is_integer_ms(value):
                integer_count += 1
            else:
                violations.append((idx, field, value))

    total_checks = len(segments) * 2
    return integer_count, total_checks, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path, help="検査対象の音声ファイル")
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-mlx",
        help="mlx-whisper のモデルID",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="整数でなかったセグメントを何件まで表示するか (デフォルト: 5)",
    )
    args = parser.parse_args()

    if not args.audio_file.is_file():
        print(f"Error: 音声ファイルが見つかりません: {args.audio_file}", file=sys.stderr)
        return 2

    print(f"[INFO] Transcribing: {args.audio_file}")
    print(f"[INFO] Model: {args.model}")
    segments = run_transcription(args.audio_file, args.model)

    if not segments:
        print("[WARN] セグメントが取得できませんでした。")
        return 1

    integer_count, total_checks, violations = check_segments(segments)

    print()
    print(f"総セグメント数         : {len(segments)}")
    print(f"検査した start/end の数: {total_checks}")
    print(f"整数msだった件数       : {integer_count}")
    print(f"非整数msだった件数     : {len(violations)}")

    if violations:
        print()
        print("[結論] mlx_whisper の生の出力は ms が整数に丸め込まれていません。")
        print(f"先頭 {min(args.show_samples, len(violations))} 件の非整数ms:")
        for idx, field, value in violations[: args.show_samples]:
            print(
                f"  segment[{idx}].{field} = {value!r} "
                f"(× 1000 = {value * 1000!r}, fractional={fractional_ms(value):.6f})"
            )
        # サンプルとして先頭セグメントの素の値も併せて表示
        print()
        print("[参考] 先頭セグメントの生の値:")
        first = segments[0]
        print(f"  start = {first['start']!r}")
        print(f"  end   = {first['end']!r}")
        return 1

    print()
    print("[結論] mlx_whisper の生の出力は ms が整数に丸め込まれています。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
