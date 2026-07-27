"""未知話者の事後ラベル付け。

Step 1 で声紋DB に載っていなかったクラスタへ、あとから名前を与える画面。
1件ラベル付けするたびに、更新した DB で残りのクラスタを再照合する。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse

import src.voice_db.registry as vdb
from src.common.audio import extract_audio
from src.common.logging import get_logger
from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SPEAKER_THRESHOLD,
    UNKNOWN_LABEL_PREFIX,
)
from src.web import jobs, storage
from src.web.converters import csv_to_srt_with_speaker
from src.web.errors import WebInputError
from src.web.forms import parse_opt_float
from src.web.identification import load_identifier
from src.web.templating import render_error, templates

logger = get_logger(__name__)

router = APIRouter(prefix="/unknowns")

#: CSV 置換の指示。``{unknown_label: (新しい話者名, 新しい距離 or None)}``。
SpeakerMapping = Dict[str, Tuple[str, Optional[float]]]

#: DB 選択で「新規作成」を表すフォーム値。
NEW_DATABASE_CHOICE = "__new__"

_INVALID_SPEAKER_NAME = "話者名が無効です（空 or 使用不可文字）。"
_INVALID_DATABASE_NAME = "新規データベース名が無効です（空 or 使用不可文字）。"


class _ConfirmExistingDatabase(Exception):
    """新規作成しようとした DB が既にある。上書き前に確認を挟む。"""

    def __init__(self, context: Dict[str, str]) -> None:
        super().__init__()
        self.context = context


@dataclass(frozen=True)
class _LabelTarget:
    """ラベル付け 1 件の対象。"""

    job_id: str
    job: Dict[str, Any]
    cluster: Dict[str, Any]


@dataclass
class LabelForm:
    """未知クラスタにラベルを付けるフォーム。"""

    speaker_name: str = Form(...)
    db_name: str = Form(...)
    new_db_name: str = Form("")
    hf_token_override: str = Form("")
    clip_start: str = Form("")
    clip_end: str = Form("")


@router.get("/{job_id}", response_class=HTMLResponse)
async def unknowns_page(request: Request, job_id: str):
    """未知クラスタの一覧画面。"""
    job = jobs.load_job(job_id)
    if job is None:
        return render_error(request, f"ジョブが見つかりません: {job_id}")
    return templates.TemplateResponse(
        request,
        "unknowns.html",
        {"job": job, "databases": vdb.list_databases()},
    )


@router.get("/{job_id}/clip/{cluster_id}")
async def unknowns_clip(job_id: str, cluster_id: str):
    """クラスタの代表音声を返す（試聴用）。"""
    job = jobs.load_job(job_id)
    if job is None:
        return HTMLResponse("Job not found", status_code=404)
    safe_cluster = Path(cluster_id).name
    if safe_cluster != cluster_id:
        return HTMLResponse("Invalid cluster_id", status_code=400)
    clip_path = jobs.job_dir(job_id) / f"clip_{safe_cluster}.wav"
    if not clip_path.exists():
        return HTMLResponse("Clip not found", status_code=404)
    return FileResponse(path=clip_path, media_type="audio/wav")


@router.post("/{job_id}/label/{cluster_id}", response_class=HTMLResponse)
async def unknowns_label(
    request: Request,
    job_id: str,
    cluster_id: str,
    form: LabelForm = Depends(),
):
    """クラスタに名前を付けて声紋DB へ登録し、残りを再照合する。"""
    try:
        target = _require_unresolved_target(job_id, cluster_id)
        speaker_name = _require_name(form.speaker_name, _INVALID_SPEAKER_NAME)
        db_name = _resolve_target_database(form, speaker_name, cluster_id)
        db_dir = _require_database(db_name)

        _register_clip_as_speaker(target, db_name, speaker_name, form)
        identifier = _reload_identifier(target.job, db_dir, form.hf_token_override)
        message = _label_and_rematch(target, speaker_name, identifier)

        target.job["db_name"] = db_name  # 以降のラベル付けでもデフォルトに使う
        jobs.save_job(job_id, target.job)
        return _render_cluster_list(request, target.job, last_message=message)

    except _ConfirmExistingDatabase as confirm:
        return _render_cluster_list(
            request, target.job, confirm_existing=confirm.context
        )
    except WebInputError as e:
        return render_error(request, str(e))


def _render_cluster_list(request: Request, job: Dict[str, Any], **extra: Any):
    """クラスタ一覧のフラグメントを返す。"""
    return templates.TemplateResponse(
        request,
        "partials/unknowns_list.html",
        {"job": job, "databases": vdb.list_databases(), **extra},
    )


def _require_unresolved_target(job_id: str, cluster_id: str) -> _LabelTarget:
    """まだラベル付けされていないクラスタを、所属ジョブごと取り出す。

    Raises:
        WebInputError: ジョブやクラスタが無い、またはラベル付け済みの場合。
    """
    job = jobs.load_job(job_id)
    if job is None:
        raise WebInputError(f"ジョブが見つかりません: {job_id}")
    cluster = next((c for c in job["clusters"] if c["cluster_id"] == cluster_id), None)
    if cluster is None:
        raise WebInputError(f"クラスタが見つかりません: {cluster_id}")
    if cluster.get("resolved"):
        raise WebInputError(f"クラスタ {cluster_id} は既にラベル付け済みです。")
    return _LabelTarget(job_id=job_id, job=job, cluster=cluster)


def _require_name(raw: str, message: str) -> str:
    """DB名 / 話者名を検証する。

    Raises:
        WebInputError: ファイル名として使えない場合。
    """
    safe = vdb.sanitize_name(raw)
    if safe is None:
        raise WebInputError(message)
    return safe


def _resolve_target_database(
    form: LabelForm, speaker_name: str, cluster_id: str
) -> str:
    """保存先 DB 名を決める。新規作成が指定されていれば作る。

    Raises:
        _ConfirmExistingDatabase: 同名の DB が既にあり、確認が要る場合。
        WebInputError: 名前が不正、または作成に失敗した場合。
    """
    if form.db_name != NEW_DATABASE_CHOICE:
        return form.db_name

    safe_new = _require_name(form.new_db_name, _INVALID_DATABASE_NAME)
    # 同名の既存DBがあれば確認バナーを返して処理を中断
    if (vdb.get_root() / safe_new).is_dir():
        raise _ConfirmExistingDatabase(
            {
                "db_name": safe_new,
                "speaker_name": speaker_name,
                "cluster_id": cluster_id,
            }
        )
    try:
        vdb.create_database(safe_new)
    except ValueError as e:
        raise WebInputError(f"DB作成エラー: {e}") from e
    return safe_new


def _require_database(name: str) -> Path:
    """既存 DB のディレクトリを返す。

    Raises:
        WebInputError: 名前が不正、または DB が無い場合。
    """
    try:
        return vdb.database_dir(name)
    except (ValueError, FileNotFoundError) as e:
        raise WebInputError(f"DBエラー: {e}") from e


def _register_clip_as_speaker(
    target: _LabelTarget, db_name: str, speaker_name: str, form: LabelForm
) -> None:
    """クラスタ音声を（必要なら切り出して）声紋DB へ保存する。

    Raises:
        WebInputError: 音声が無い / 範囲が不正 / 切り出しや保存に失敗した場合。
    """
    clip_path, tmp_crop = _prepare_clip(target, (form.clip_start, form.clip_end))
    try:
        vdb.add_speaker_file(db_name, clip_path, dest_filename=f"{speaker_name}.wav")
    except ValueError as e:
        raise WebInputError(f"DB保存エラー: {e}") from e
    finally:
        if tmp_crop is not None:
            storage.remove_quietly(tmp_crop)


def _prepare_clip(
    target: _LabelTarget, crop_range: Tuple[str, str]
) -> Tuple[Path, Optional[Path]]:
    """保存に使う音声を用意する。

    Args:
        target: ラベル付けの対象。
        crop_range: フォームの ``(開始秒, 終了秒)``。空なら全体を使う。

    Returns:
        ``(保存に使う音声, 後始末が要る一時ファイル or None)``。

    Raises:
        WebInputError: 音声が無い / 範囲が不正 / 切り出しに失敗した場合。
    """
    job_dir = jobs.job_dir(target.job_id)
    clip_path = job_dir / target.cluster["clip_filename"]
    if not clip_path.exists():
        raise WebInputError(f"クラスタ音声が見つかりません: {clip_path}")

    start = parse_opt_float(crop_range[0])
    end = parse_opt_float(crop_range[1])
    if not ((start is not None and start > 0) or end is not None):
        return clip_path, None
    if start is not None and end is not None and end <= start:
        raise WebInputError("終了時間は開始時間より後にしてください。")

    tmp_crop = job_dir / f"_cropsave_{target.cluster['cluster_id']}.wav"
    try:
        extract_audio(clip_path, tmp_crop, start=start, end=end, to_wav16k=True)
    except subprocess.CalledProcessError as e:
        raise WebInputError(f"音声の切り出しに失敗しました: {e.stderr}") from e
    return tmp_crop, tmp_crop


def _reload_identifier(
    job: Dict[str, Any], db_dir: Path, hf_token_override: str
) -> Any:
    """更新後の DB 全体を登録した照合器を読み直す。

    Raises:
        WebInputError: トークンが無い、またはモデルを読み込めない場合。
    """
    hf_token = hf_token_override or os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise WebInputError("Hugging Face Token が設定されていません。")
    try:
        return load_identifier(
            db_dir,
            model_name=job.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            hf_token=hf_token,
            threshold=float(job.get("threshold", DEFAULT_SPEAKER_THRESHOLD)),
        )
    except Exception as e:
        logger.exception("identifier reload failed")
        raise WebInputError(f"声紋モデルの読み込みエラー: {e}") from e


def _label_and_rematch(target: _LabelTarget, speaker_name: str, identifier: Any) -> str:
    """対象を解決済みにし、残りを再照合してから CSV / SRT を更新する。

    Returns:
        画面に出す結果の通知文。
    """
    target.cluster["resolved"] = True
    target.cluster["resolved_name"] = speaker_name
    target.cluster["resolved_method"] = "manual"
    mapping: SpeakerMapping = {target.cluster["unknown_label"]: (speaker_name, None)}

    auto_resolved = _rematch_remaining(target, identifier, mapping)
    replaced = _apply_mapping(target.job, mapping)

    message = (
        f"✓ {target.cluster['unknown_label']} を「{speaker_name}」として登録しました "
        f"(CSV {replaced}行を更新)"
    )
    if auto_resolved:
        message += " · 自動再照合で確定: " + ", ".join(auto_resolved)
    return message


def _rematch_remaining(
    target: _LabelTarget, identifier: Any, mapping: SpeakerMapping
) -> List[str]:
    """残った未解決クラスタを更新後の DB で再照合する。

    閾値以下に到達したクラスタは解決済みにし、``mapping`` へ追記する。
    到達しなかったクラスタは距離だけ更新する（情報として有用なため）。

    Returns:
        自動解決したクラスタの説明文（画面表示用）。
    """
    job_dir = jobs.job_dir(target.job_id)
    messages: List[str] = []
    for cluster in target.job["clusters"]:
        if cluster.get("resolved"):
            continue
        clip_path = job_dir / cluster["clip_filename"]
        if not clip_path.exists():
            continue
        try:
            name, distance, _candidates = identifier.identify_from_audio_path(clip_path)
        except Exception as e:
            logger.warning(f"再照合失敗 (cluster {cluster['cluster_id']}): {e}")
            continue
        cluster["distance"] = distance
        if name.startswith(UNKNOWN_LABEL_PREFIX):
            continue
        cluster["resolved"] = True
        cluster["resolved_name"] = name
        cluster["resolved_method"] = "auto"
        mapping[cluster["unknown_label"]] = (name, distance)
        messages.append(f"{cluster['unknown_label']} → {name} (距離 {distance:.4f})")
    return messages


def _apply_mapping(job: Dict[str, Any], mapping: SpeakerMapping) -> int:
    """CSV の話者名を置換し、SRT を再生成する。

    Returns:
        置換した CSV の行数。
    """
    csv_path = Path(job["csv_path"])
    replaced = jobs.relabel_csv(csv_path, mapping)
    srt_path_str = job.get("srt_path")
    if srt_path_str:
        try:
            csv_to_srt_with_speaker(csv_path, Path(srt_path_str))
        except Exception:
            logger.exception("SRT 再生成に失敗しました")
    return replaced
