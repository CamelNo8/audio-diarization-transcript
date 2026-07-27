"""文字起こしジョブの状態管理。

Step 1 の進捗と結果（CSV / SRT のパスと未解決クラスタ）を ``job.json`` に
永続化し、処理中の進捗確認と、Step 1 の後からの話者ラベル付けに使う。

**ディスクが唯一の真実**であり、メモリにキャッシュしない。文字起こしは
ワーカースレッドで走り、その間に別のリクエストが状態を読むため、キャッシュを
持つと書き手と読み手で食い違う（uvicorn を複数ワーカーで起動した場合はプロセスも
別になる）。書き込みは :func:`src.common.json_io.write_json` でアトミックに行う。

``job_id`` は URL から渡ってくるため、パスに使う前に必ず :func:`job_dir` で
検証する（規約7.1）。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.common.csv_io import read_rows, write_rows
from src.common.json_io import read_json, write_json
from src.config import CLUSTERS_ROOT as _CONFIGURED_CLUSTERS_ROOT

#: ジョブの保存先ルート。テストではこのモジュール属性を差し替える。
CLUSTERS_ROOT = _CONFIGURED_CLUSTERS_ROOT

#: ``job_id`` として許す文字（英数字・アンダースコア・ハイフンのみ）。
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

#: ジョブの状態。
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


def new_job_id() -> str:
    """``<日時>-<乱数>`` 形式のジョブ ID を払い出す。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def job_dir(job_id: str) -> Path:
    """ジョブの保存ディレクトリを返す。

    Raises:
        ValueError: ``job_id`` にパス区切りなど想定外の文字が含まれる場合。
    """
    safe = Path(job_id).name
    if safe != job_id or not _JOB_ID_RE.match(job_id):
        raise ValueError(f"無効な job_id: {job_id!r}")
    return CLUSTERS_ROOT / safe


def save_job(job_id: str, job: Dict[str, Any]) -> None:
    """ジョブ状態を ``job.json`` へ書く（アトミック）。"""
    write_json(job_dir(job_id) / "job.json", job)


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """ジョブ状態を読み込む。見つからない・ID が不正なら ``None``。

    ルート側で 404 相当の扱いにするため、ID が不正でも例外にはしない。
    """
    try:
        path = job_dir(job_id) / "job.json"
    except ValueError:
        return None
    job = read_json(path)
    if job is None:
        return None
    # 旧フォーマットのジョブには csv_filename が無いため csv_path から補完する
    if not job.get("csv_filename") and job.get("csv_path"):
        job["csv_filename"] = Path(job["csv_path"]).name
    return job


def update_job(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """ジョブの一部のフィールドだけを更新する。

    読んで・足して・書き戻す。ジョブ1件を触るのは同時に1スレッドだけ
    （文字起こしのワーカー、またはラベル付けのリクエスト）なので、
    ここでは追加のロックを取らない。

    Returns:
        更新後のジョブ。ジョブが無ければ ``None``。
    """
    job = load_job(job_id)
    if job is None:
        return None
    job.update(fields)
    save_job(job_id, job)
    return job


def relabel_csv(csv_path: Path, mapping: Dict[str, Tuple[str, Optional[float]]]) -> int:
    """CSV 内の speaker 列が mapping のキーに一致する行を新名で置換する。

    Args:
        csv_path: 文字起こし CSV。
        mapping: ``{unknown_label: (新しい話者名, 新しい距離 or None)}``。

    Returns:
        置換した行数。speaker 列が無い CSV では 0。
    """
    if not mapping or not csv_path.exists():
        return 0
    rows = read_rows(csv_path)
    if not rows:
        return 0
    header = rows[0]
    if "speaker" not in header:
        return 0
    sp_idx = header.index("speaker")
    dist_idx = header.index("cosine_distance") if "cosine_distance" in header else -1

    count = 0
    for row in rows[1:]:
        if sp_idx >= len(row) or row[sp_idx] not in mapping:
            continue
        new_name, new_dist = mapping[row[sp_idx]]
        row[sp_idx] = new_name
        if 0 <= dist_idx < len(row):
            row[dist_idx] = f"{new_dist:.6f}" if new_dist is not None else ""
        count += 1
    write_rows(csv_path, rows)
    return count
