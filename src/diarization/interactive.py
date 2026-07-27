"""未知話者を CLI で対話的に登録する。

このモジュールの ``print`` / ``input`` はログではなく端末の操作画面そのもの
（再生の案内と入力プロンプト）なので、ロガーには置き換えない。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from src.common.audio import extract_audio
from src.common.logging import get_logger
from src.config import INVALID_NAME_CHARS
from src.diarization.clusters import (
    ClusterAssignments,
    recompute_distances_for_cluster,
    remap_remaining_unknowns,
)

logger = get_logger(__name__)

#: 「以降すべてスキップ」を表す内部の合図。
_SKIP_ALL = "__SKIP_ALL__"

#: 画面の区切り線の長さ。
_RULE_WIDTH = 60


def resolve_unknown_speakers(
    identifier,
    assignments: ClusterAssignments,
    wav_path: Optional[Path],
    registry_dir: Optional[Path],
) -> None:
    """``Unknown_NN`` のクラスタについて、代表音声を聞かせて名前を入力してもらう。

    入力された名前で声紋を永続登録し、登録するたびに残りの未知クラスタを再照合する。

    Args:
        identifier: 話者識別器。``None`` なら何もしない。
        assignments: 照合結果（この場で更新される）。
        wav_path: 代表音声の切り出し元 WAV。
        registry_dir: 声紋ファイルの保存先ディレクトリ。
    """
    if identifier is None or registry_dir is None:
        return
    if wav_path is None or not Path(wav_path).exists():
        logger.warning("一時 WAV が無いため対話的登録をスキップします。")
        return

    unknown_clusters = sorted(assignments.unknown_cluster_ids())
    if not unknown_clusters:
        return
    if not registry_dir.is_dir():
        logger.warning(
            f"registry_dir が存在しないため対話的登録をスキップします: {registry_dir}"
        )
        return

    _print_intro(len(unknown_clusters))
    skip_all = False
    for cluster_id in unknown_clusters:
        if skip_all:
            continue
        # 直前の再照合で名前が確定していたらスキップ
        if cluster_id not in assignments.unknown_cluster_ids():
            continue

        result = _resolve_one(
            cluster_id, identifier, assignments, wav_path, registry_dir
        )
        if result is _SKIP_ALL:
            skip_all = True

    _print_outro()


def _resolve_one(
    cluster_id: str,
    identifier,
    assignments: ClusterAssignments,
    wav_path: Path,
    registry_dir: Path,
) -> Optional[str]:
    """1クラスタ分の対話・登録・再照合を行う。

    Returns:
        以降すべてスキップする場合は :data:`_SKIP_ALL`、それ以外は ``None``。
    """
    clip_path = _extract_cluster_audio(wav_path, assignments, cluster_id)
    if clip_path is None:
        logger.warning(
            f"クラスタ {cluster_id} の音声切り出しに失敗したためスキップします。"
        )
        return None

    try:
        resolved = _prompt_user_for_speaker(
            cluster_id, clip_path, assignments, registry_dir
        )
    finally:
        _unlink_quietly(clip_path)

    if resolved is None:
        return None
    if resolved == _SKIP_ALL:
        return _SKIP_ALL

    name, saved_path = resolved
    _register_and_remap(identifier, assignments, cluster_id, name, saved_path)
    return None


def _register_and_remap(
    identifier,
    assignments: ClusterAssignments,
    cluster_id: str,
    name: str,
    saved_path: Path,
) -> None:
    """入力された名前で声紋を登録し、残りの未知クラスタを再照合する。"""
    try:
        identifier.register_speaker(name, saved_path)
    except Exception as e:
        # 保存ファイルを残しても識別器には未登録なので、次のクラスタへ進む
        logger.error(f"声紋登録に失敗しました ({name}): {e}")
        return

    # 新規登録後、自クラスタの埋め込みを再照合して全候補との距離を確定させる
    distance, candidates = recompute_distances_for_cluster(
        identifier, assignments, cluster_id
    )
    assignments.set_speaker(cluster_id, name, distance, candidates)
    dist_str = f"{distance:.6f}" if distance is not None else "N/A"
    print(
        f"  → クラスタ {cluster_id} を「{name}」として登録しました "
        f"(cosine_distance={dist_str})。"
    )

    for remapped_id, remapped_name, remapped_distance in remap_remaining_unknowns(
        identifier, assignments
    ):
        print(
            f"  → クラスタ {remapped_id} は再照合の結果「{remapped_name}」"
            f" (cosine_distance={remapped_distance:.6f}) に確定しました。"
        )


def _print_intro(unknown_count: int) -> None:
    """対話の開始案内を表示する。"""
    print()
    print("=" * _RULE_WIDTH)
    print(f"未登録クラスタが {unknown_count} 個あります。")
    print("各クラスタの代表音声を再生しますので、声の主の名前を入力してください。")
    print("=" * _RULE_WIDTH)


def _print_outro() -> None:
    """対話の終了案内を表示する。"""
    print("=" * _RULE_WIDTH)
    print("対話的登録を終了します。")
    print("=" * _RULE_WIDTH)
    print()


def _extract_cluster_audio(
    wav_path: Path, assignments: ClusterAssignments, cluster_id: str
) -> Optional[Path]:
    """クラスタの代表音声区間を一時 WAV に切り出して返す。失敗時は ``None``。"""
    segment = assignments.segments.get(cluster_id)
    if segment is None:
        return None

    fd, tmp_str = tempfile.mkstemp(suffix=".wav", prefix=f"cluster_{cluster_id}_")
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        extract_audio(wav_path, tmp_path, start=segment.start, end=segment.end)
    except subprocess.CalledProcessError as e:
        logger.warning(f"代表音声の切り出しに失敗: {e.stderr}")
        _unlink_quietly(tmp_path)
        return None
    return tmp_path


def _play_audio(audio_path: Path) -> None:
    """afplay でブロッキング再生する。失敗してもエラーにしない。"""
    try:
        subprocess.run(
            ["afplay", str(audio_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"afplay による再生に失敗しました: {e}")


def sanitize_speaker_name(raw: str) -> Optional[str]:
    """入力された話者名をファイル名に使える形へ整える。

    Returns:
        前後の空白を除いた名前。空、またはファイル名に使えない文字を
        含む場合は ``None``。
    """
    name = raw.strip()
    if not name:
        return None
    if any(c in INVALID_NAME_CHARS for c in name):
        return None
    return name


def _prompt_user_for_speaker(
    cluster_id: str,
    clip_path: Path,
    assignments: ClusterAssignments,
    registry_dir: Path,
) -> Optional[object]:
    """代表音声を再生し、話者名を尋ねる。

    Returns:
        ``None``（スキップ）、:data:`_SKIP_ALL`（以降全てスキップ）、
        または ``(名前, 保存先パス)`` の組。
    """
    current_label = assignments.speaker_mapping.get(cluster_id, cluster_id)
    print()
    print(f"--- クラスタ {cluster_id} (現ラベル: {current_label}) ---")
    print(f"代表音声ファイル: {clip_path}")
    _play_audio(clip_path)

    while True:
        try:
            raw = input(
                "このクラスタの声は誰ですか？\n"
                "  名前を入力 / [Enter]=スキップ / [r]=もう一度再生 / "
                "[s]=以降全てスキップ: "
            )
        except EOFError:
            return None

        cmd = raw.strip()
        if cmd == "":
            return None
        if cmd.lower() == "r":
            _play_audio(clip_path)
            continue
        if cmd.lower() == "s":
            return _SKIP_ALL

        name = sanitize_speaker_name(cmd)
        if name is None:
            print(
                "  ! 名前が空、または使用できない文字が含まれています。"
                "再入力してください。"
            )
            continue

        saved_path = _persist_registry_audio(registry_dir, name, clip_path)
        if saved_path is None:
            continue
        return (name, saved_path)


def _persist_registry_audio(
    registry_dir: Path, name: str, clip_path: Path
) -> Optional[Path]:
    """切り出した代表音声を ``registry_dir/<name>.wav`` として永続保存する。

    既存ファイルがある場合は上書きするか確認する。

    Returns:
        保存先のパス。保存しなかった場合は ``None``。
    """
    target = registry_dir / f"{name}.wav"

    if target.exists() and not _confirm_overwrite(target):
        print("  → 別の名前を入力してください。")
        return None

    try:
        registry_dir.mkdir(parents=True, exist_ok=True)
        with open(clip_path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
    except OSError as e:
        logger.error(f"声紋ファイルの保存に失敗しました ({target}): {e}")
        return None
    print(f"  → 声紋ファイルを保存しました: {target}")
    return target


def _confirm_overwrite(target: Path) -> bool:
    """既存の声紋ファイルを上書きしてよいか尋ねる。"""
    try:
        answer = input(f"  ! {target} は既に存在します。上書きしますか？ [y/N]: ")
    except EOFError:
        return False
    return answer.strip().lower() == "y"


def _unlink_quietly(path: Path) -> None:
    """一時ファイルを削除する。消せなくても処理は続行する。"""
    try:
        path.unlink()
    except OSError:
        pass
