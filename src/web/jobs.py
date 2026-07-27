"""未知話者ラベル付けジョブの状態管理。

Step 1 の結果（CSV / SRT のパスと未解決クラスタ）を ``job.json`` に永続化し、
Step 1 の後からでもラベル付けできるようにする。

``job_id`` は URL から渡ってくるため、パスに使う前に必ず :func:`job_dir` で
検証する（規約7.1）。
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.common.csv_io import read_rows, write_rows
from src.config import CLUSTERS_ROOT as _CONFIGURED_CLUSTERS_ROOT

#: ジョブの保存先ルート。テストではこのモジュール属性を差し替える。
CLUSTERS_ROOT = _CONFIGURED_CLUSTERS_ROOT

#: ``job_id`` として許す文字（英数字・アンダースコア・ハイフンのみ）。
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

#: job_id -> ジョブ状態のメモリキャッシュ。実体は job.json。
_JOBS: Dict[str, Dict[str, Any]] = {}


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
    """ジョブ状態をメモリキャッシュとディスクの両方へ書く。"""
    _JOBS[job_id] = job
    path = job_dir(job_id) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2, default=str)


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """ジョブ状態を読み込む。見つからない・ID が不正なら ``None``。

    ルート側で 404 相当の扱いにするため、ID が不正でも例外にはしない。
    """
    if job_id in _JOBS:
        return _JOBS[job_id]
    try:
        path = job_dir(job_id) / "job.json"
    except ValueError:
        return None
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    # 旧フォーマットのジョブには csv_filename が無いため csv_path から補完する
    if not job.get("csv_filename") and job.get("csv_path"):
        job["csv_filename"] = Path(job["csv_path"]).name
    _JOBS[job_id] = job
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
