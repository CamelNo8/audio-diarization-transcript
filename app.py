from __future__ import annotations

import io
import os
import re
import csv
import json
import uuid
import shutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from audio_processor import AudioProcessor
from main import (
    get_cached_speaker_identifier,
    collect_registry_files,
    SUPPORTED_REGISTRY_EXTENSIONS,
)
from subtitle_matcher import run_matching_process
from subtitle_exporter import (
    load_subtitle_data,
    generate_srt_content,
    write_srt_file,
)
import voice_database as vdb

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
CLUSTERS_ROOT = TEMP_DIR / "clusters"
CLUSTERS_ROOT.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"

# job_id -> job state (in-memory cache; persisted to job.json on disk)
_JOBS: Dict[str, Dict[str, Any]] = {}

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="音声→字幕 統合アプリ")


# ===================================================================
# Helper: 文字起こしCSV (HH:MM:SS:ms) → SRT
# ===================================================================
def _colon_ms_to_comma_ms(t: str) -> str:
    if not t:
        return ""
    if "," in t:
        return t
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})[:.,](\d{1,3})$", t)
    if m:
        h, mi, s, ms = m.groups()
        return f"{int(h):02d}:{int(mi):02d}:{int(s):02d},{int(ms):03d}"
    if t.count(":") >= 3:
        head, _, tail = t.rpartition(":")
        return f"{head},{tail}"
    return t


def _csv_to_srt_with_speaker(csv_path: Path, srt_path: Path) -> int:
    """Step 1 の文字起こしCSVから speaker prefix 付きSRTを生成する。

    本文は "[speaker] text" 形式。speaker が空 / "Unknown" / "Unknown_*" の場合も
    そのまま prefix として書き出す（ユーザーが視認できるように）。
    Returns: 書き出した字幕ブロック数
    """
    if not csv_path.exists():
        return 0
    blocks = []
    idx = 1
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = _colon_ms_to_comma_ms((row.get("start") or "").strip())
            end = _colon_ms_to_comma_ms((row.get("end") or "").strip())
            body = (row.get("text") or "").strip()
            speaker = (row.get("speaker") or "").strip()
            if not start or not end or not body:
                continue
            line = f"[{speaker}] {body}" if speaker else body
            blocks.append(f"{idx}\n{start} --> {end}\n{line}\n")
            idx += 1
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return idx - 1


# ===================================================================
# Helper: 台本/書き起こし テキスト → 台本CSV (id,scene_id,type,speaker,contents)
# ===================================================================
_SPEAKER_LINE_RE = re.compile(r"^\s*([^\s:：（(][^:：]{0,30}?)\s*[:：]\s*(.+)$")
_SCENE_PAREN_RE = re.compile(r"^\s*[（(](.+)[)）]\s*$")


def _txt_to_script_csv_bytes(txt_bytes: bytes) -> bytes:
    """プレーンテキストを台本CSV(id,scene_id,type,speaker,contents)形式に変換する。

    各非空行を 1 行に変換する:
      - `# <内容>` または `（...）` `(...)` → type=scene, speaker=空, contents=中身
      - `<話者>:<本文>` / `<話者>：<本文>` → type=dialogue, speaker=話者, contents=本文
      - それ以外 → type=dialogue, speaker=空, contents=行
    id は 1 から連番、scene_id は空欄。
    """
    text = txt_bytes.decode("utf-8-sig")
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id", "scene_id", "type", "speaker", "contents"])
    idx = 1
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            writer.writerow([idx, "", "scene", "", line.lstrip("#").strip()])
            idx += 1
            continue
        m_scene = _SCENE_PAREN_RE.match(line)
        if m_scene:
            writer.writerow([idx, "", "scene", "", m_scene.group(1).strip()])
            idx += 1
            continue
        m_sp = _SPEAKER_LINE_RE.match(line)
        if m_sp:
            writer.writerow([idx, "", "dialogue", m_sp.group(1).strip(), m_sp.group(2).strip()])
            idx += 1
            continue
        writer.writerow([idx, "", "dialogue", "", line])
        idx += 1
    return buf.getvalue().encode("utf-8-sig")


def _render_error(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/error.html",
        {"message": message},
    )


# ===================================================================
# Job state (Unknown labeling)
# ===================================================================
def _new_job_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _job_dir(job_id: str) -> Path:
    safe = Path(job_id).name
    if safe != job_id or not re.match(r"^[A-Za-z0-9_\-]+$", job_id):
        raise ValueError(f"無効な job_id: {job_id!r}")
    return CLUSTERS_ROOT / safe


def _save_job(job_id: str, job: Dict[str, Any]) -> None:
    _JOBS[job_id] = job
    path = _job_dir(job_id) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2, default=str)


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    if job_id in _JOBS:
        return _JOBS[job_id]
    try:
        path = _job_dir(job_id) / "job.json"
    except ValueError:
        return None
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    _JOBS[job_id] = job
    return job


def _relabel_csv(csv_path: Path, mapping: Dict[str, Tuple[str, Optional[float]]]) -> int:
    """CSV 内の speaker 列が mapping のキーに一致する行を新名で置換する。

    mapping: { unknown_label: (new_name, new_distance_or_None) }
    Returns: 置換した行数
    """
    if not mapping or not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    header = rows[0]
    try:
        sp_idx = header.index("speaker")
    except ValueError:
        return 0
    dist_idx = header.index("cosine_distance") if "cosine_distance" in header else -1

    count = 0
    for row in rows[1:]:
        if sp_idx < len(row) and row[sp_idx] in mapping:
            new_name, new_dist = mapping[row[sp_idx]]
            row[sp_idx] = new_name
            if dist_idx >= 0 and dist_idx < len(row):
                row[dist_idx] = f"{new_dist:.6f}" if new_dist is not None else ""
            count += 1
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)
    return count


# ===================================================================
# Helper: 音声の切り出し（トリミング）
# ===================================================================
def _parse_opt_float(raw: str) -> Optional[float]:
    """フォームの開始/終了秒をパースする。空 / 不正 / 負値は None。"""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v >= 0 else None


def _crop_audio_file(
    src: Path,
    dst: Path,
    start: Optional[float],
    end: Optional[float],
    *,
    to_wav16k: bool,
) -> None:
    """src を [start, end]（src 内の絶対秒）で dst に切り出す。

    start/end は省略可（None なら先頭/末尾まで）。
    to_wav16k=True で 16kHz mono WAV に再エンコード（声紋用）、
    False なら元コーデックを尊重しつつ dst の拡張子に従う。
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vn"]
    if to_wav16k:
        cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    cmd += ["-y", str(dst)]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


# ===================================================================
# Routes
# ===================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"databases": vdb.list_databases()},
    )


@app.post("/process/transcription", response_class=HTMLResponse)
async def process_transcription(
    request: Request,
    audio_file: UploadFile = File(...),
    registry_files: List[UploadFile] = File(default=[]),
    db_choice: str = Form("none"),  # "none" / "existing" / "new"
    db_existing_name: str = Form(""),
    db_new_name: str = Form(""),
    output_srt_name: str = Form("transcription.srt"),
    threshold: float = Form(0.5),
    num_speakers: str = Form(""),
    embedding_model: str = Form("pyannote/embedding"),
    mlx_model: str = Form("mlx-community/whisper-large-v3-mlx"),
    pyannote_model_id: str = Form("pyannote/speaker-diarization-3.1"),
    hf_token_override: str = Form(""),
    denoise_mode: str = Form("off"),
):
    try:
        if not shutil.which("ffmpeg"):
            return _render_error(request, "ffmpeg が PATH に見つかりません。`brew install ffmpeg` でインストールしてください。")

        hf_token = hf_token_override or os.getenv("HF_TOKEN", "")
        if not hf_token:
            return _render_error(request, "Hugging Face Token が設定されていません。.env の HF_TOKEN または詳細設定で指定してください。")

        if not audio_file.filename:
            return _render_error(request, "音声/動画ファイルが指定されていません。")

        # 保存先
        safe_audio_name = Path(audio_file.filename).name
        audio_path = TEMP_DIR / f"upload_audio_{safe_audio_name}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # 声紋DB の選択／作成
        registry_dir: Optional[Path] = None
        if db_choice == "existing":
            if not db_existing_name:
                return _render_error(request, "既存DBが選択されていません。")
            try:
                registry_dir = vdb.database_dir(db_existing_name)
            except (ValueError, FileNotFoundError) as e:
                return _render_error(request, f"DBエラー: {e}")
        elif db_choice == "new":
            safe_new = vdb.sanitize_name(db_new_name)
            if safe_new is None:
                return _render_error(request, "新規DB名が無効です。")
            try:
                registry_dir = vdb.create_database(safe_new)
            except ValueError as e:
                # 既存ならそれを使う
                try:
                    registry_dir = vdb.database_dir(safe_new)
                except Exception:
                    return _render_error(request, f"DB作成エラー: {e}")
        # else: db_choice == "none" → registry_dir は None のまま

        # アップロードされた声紋ファイルを DB に追加
        valid_uploads = [
            rf for rf in registry_files
            if rf and rf.filename and Path(rf.filename).suffix.lower() in SUPPORTED_REGISTRY_EXTENSIONS
        ]
        if valid_uploads:
            if registry_dir is None:
                return _render_error(
                    request,
                    "声紋ファイルがアップロードされていますが、保存先DBが選択されていません。"
                    "「既存DBを使う」または「新規DBを作成」を選択してください。",
                )
            for rf in valid_uploads:
                rname = Path(rf.filename).name
                tmp_upload = TEMP_DIR / f"upload_registry_{rname}"
                with open(tmp_upload, "wb") as f:
                    shutil.copyfileobj(rf.file, f)
                try:
                    vdb.add_speaker_file(registry_dir.name, tmp_upload, dest_filename=rname)
                finally:
                    try:
                        tmp_upload.unlink()
                    except OSError:
                        pass

        # ロギング取得
        log_buffer = io.StringIO()
        log_handler = logging.StreamHandler(log_buffer)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)

        identifier = None
        srt_stem = Path(output_srt_name).stem or "transcription"
        output_srt_path = TEMP_DIR / output_srt_name
        output_csv_path = TEMP_DIR / f"{srt_stem}.csv"
        job_id = _new_job_id()
        unknown_clusters: List[Dict[str, Any]] = []
        try:
            if registry_dir is not None:
                registry_paths = collect_registry_files(registry_dir)
                identifier = get_cached_speaker_identifier(
                    model_name=embedding_model,
                    hf_token=hf_token,
                    threshold=threshold,
                )
                for name, path in registry_paths.items():
                    identifier.register_speaker(name, path)

            num_speakers_val: Optional[int] = None
            if num_speakers.strip():
                try:
                    num_speakers_val = int(num_speakers.strip())
                except ValueError:
                    pass

            denoise_models = {
                "fast": "Kim_Vocal_2.onnx",
                "high": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            }
            separator_model = denoise_models.get(denoise_mode)
            with AudioProcessor(
                audio_file=audio_path,
                output_csv_path=output_csv_path,
                mlx_model_id=mlx_model,
                pyannote_model_id=pyannote_model_id,
                hf_token=hf_token,
                identifier=identifier,
                registry_dir=registry_dir,
                interactive_unknown_resolve=False,
                denoise=separator_model is not None,
                separator_model=separator_model,
            ) as processor:
                success = processor.process_and_save_to_csv(known_num_speakers=num_speakers_val)
                # Unknown クラスタの音声を永続化（成功時のみ）
                if success:
                    unknown_clusters = processor.persist_unknown_clusters(_job_dir(job_id))
        finally:
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(prev_level)

        if not success or not output_csv_path.exists():
            return _render_error(
                request,
                "文字起こし処理に失敗しました。\n" + log_buffer.getvalue()[-2000:],
            )

        # CSV → SRT 変換（ユーザー向けダウンロード）
        _csv_to_srt_with_speaker(output_csv_path, output_srt_path)

        # 各 cluster を「未解決」状態でジョブに登録
        for c in unknown_clusters:
            c["resolved"] = False
            c["resolved_name"] = None
        job = {
            "job_id": job_id,
            "csv_path": str(output_csv_path),
            "srt_path": str(output_srt_path),
            "srt_filename": output_srt_name,
            "db_name": registry_dir.name if registry_dir is not None else None,
            "threshold": threshold,
            "embedding_model": embedding_model,
            "clusters": unknown_clusters,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _save_job(job_id, job)

        return templates.TemplateResponse(
            request,
            "partials/success_transcription.html",
            {
                "filename": output_srt_name,
                "download_url": f"/download/{output_srt_name}",
                "log_excerpt": log_buffer.getvalue()[-3000:],
                "job_id": job_id,
                "unknown_count": len(unknown_clusters),
            },
        )

    except Exception as e:
        logging.exception("transcription failed")
        return _render_error(request, f"エラーが発生しました: {e}")


@app.post("/process/matching", response_class=HTMLResponse)
async def process_matching(
    request: Request,
    script_file: UploadFile = File(...),
    job_id: str = Form(...),
    output_csv_name: str = Form("対応表.csv"),
):
    try:
        if not script_file.filename:
            return _render_error(request, "台本or書き起こしテキストファイルが指定されていません。")
        if not job_id:
            return _render_error(request, "Step 1 (文字起こし) を先に実行してください。")

        job = _load_job(job_id)
        if job is None:
            return _render_error(request, f"Step 1 のジョブが見つかりません: {job_id}")
        srt_path_str = job.get("srt_path")
        if not srt_path_str:
            return _render_error(request, "Step 1 の SRT 出力が記録されていません。")
        srt_path = Path(srt_path_str)
        if not srt_path.exists():
            return _render_error(request, f"Step 1 の SRT が見つかりません: {srt_path}")

        # 台本: .txt は CSV に自動変換、.csv はそのまま使う
        script_name = Path(script_file.filename).name
        suffix = Path(script_name).suffix.lower()
        if suffix == ".txt":
            txt_bytes = await script_file.read()
            csv_bytes = _txt_to_script_csv_bytes(txt_bytes)
            script_path = TEMP_DIR / f"upload_script_{Path(script_name).stem}.csv"
            script_path.write_bytes(csv_bytes)
        else:
            script_path = TEMP_DIR / f"upload_script_{script_name}"
            with open(script_path, "wb") as f:
                shutil.copyfileobj(script_file.file, f)

        output_path = TEMP_DIR / output_csv_name
        # マッチングはサブプロセスで実行（PyTorch MPS + FAISS-CPU の同一スレッド競合を回避）
        import asyncio
        import sys
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u",
            str(BASE_DIR / "subtitle_matcher.py"),
            str(script_path), str(srt_path), str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout_bytes, _ = await proc.communicate()
        match_log = stdout_bytes.decode("utf-8", errors="replace")
        # 親プロセスのコンソールにも流す
        sys.stdout.write(match_log)
        sys.stdout.flush()
        if proc.returncode != 0:
            return _render_error(
                request,
                f"マッチングプロセスが異常終了しました (rc={proc.returncode})\n\n"
                + match_log[-3000:],
            )

        if not output_path.exists():
            return _render_error(request, "マッチング処理に失敗しました。出力ファイルが生成されませんでした。")

        return templates.TemplateResponse(
            request,
            "partials/success_matching.html",
            {
                "filename": output_csv_name,
                "download_url": f"/download/{output_csv_name}",
            },
        )

    except Exception as e:
        logging.exception("matching failed")
        return _render_error(request, f"エラーが発生しました: {e}")


@app.post("/process/generation", response_class=HTMLResponse)
async def process_generation(
    request: Request,
    edited_csv: UploadFile = File(...),
    output_srt_name: str = Form("subtitles.srt"),
):
    try:
        if not edited_csv.filename:
            return _render_error(request, "編集済み対応表CSVが指定されていません。")

        csv_path = TEMP_DIR / f"upload_edited_{Path(edited_csv.filename).name}"
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(edited_csv.file, f)

        subtitle_data = load_subtitle_data(str(csv_path))
        if not subtitle_data:
            return _render_error(request, "字幕データの読み込みに失敗しました。CSVフォーマットを確認してください。")

        srt_content = generate_srt_content(subtitle_data)
        output_path = TEMP_DIR / output_srt_name
        write_srt_file(str(output_path), srt_content)

        if not output_path.exists():
            return _render_error(request, "SRT生成に失敗しました。")

        return templates.TemplateResponse(
            request,
            "partials/success_generation.html",
            {
                "filename": output_srt_name,
                "download_url": f"/download/{output_srt_name}",
                "count": len(subtitle_data),
            },
        )

    except Exception as e:
        logging.exception("generation failed")
        return _render_error(request, f"エラーが発生しました: {e}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("File not found", status_code=404)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )


# ===================================================================
# 声紋DB 管理 API
# ===================================================================
@app.get("/api/databases")
async def api_list_databases():
    return {"databases": vdb.list_databases(), "root": str(vdb.get_root())}


@app.post("/api/databases", response_class=HTMLResponse)
async def api_create_database(request: Request, name: str = Form(...)):
    safe = vdb.sanitize_name(name)
    if safe is None:
        return _render_error(request, "DB名が無効です（空 or 使用不可文字）。")
    try:
        vdb.create_database(safe)
    except ValueError as e:
        return _render_error(request, str(e))
    return await _render_db_list_fragment(request)


@app.delete("/api/databases/{name}", response_class=HTMLResponse)
async def api_delete_database(request: Request, name: str):
    try:
        vdb.delete_database(name)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))
    return await _render_db_list_fragment(request)


@app.get("/api/databases/{name}/speakers", response_class=HTMLResponse)
async def api_list_speakers(request: Request, name: str):
    try:
        speakers = vdb.list_speakers(name)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": name, "speakers": speakers},
    )


@app.post("/api/databases/{name}/speakers/upload", response_class=HTMLResponse)
async def api_upload_speakers(
    request: Request,
    name: str,
    files: List[UploadFile] = File(...),
):
    try:
        # DB の存在確認
        vdb.database_dir(name)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))

    added = 0
    for uf in files:
        if not uf or not uf.filename:
            continue
        if Path(uf.filename).suffix.lower() not in vdb.SUPPORTED_AUDIO_EXTENSIONS:
            continue
        rname = Path(uf.filename).name
        tmp = TEMP_DIR / f"upload_registry_{rname}"
        with open(tmp, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        try:
            vdb.add_speaker_file(name, tmp, dest_filename=rname)
            added += 1
        except ValueError as e:
            logging.warning(f"upload skipped ({rname}): {e}")
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    speakers = vdb.list_speakers(name)
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": name, "speakers": speakers},
    )


@app.delete("/api/databases/{name}/speakers/{filename}", response_class=HTMLResponse)
async def api_delete_speaker(request: Request, name: str, filename: str):
    try:
        vdb.delete_speaker(name, filename)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))
    speakers = vdb.list_speakers(name)
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": name, "speakers": speakers},
    )


@app.post("/api/databases/{name}/speakers/{filename}/rename", response_class=HTMLResponse)
async def api_rename_speaker(
    request: Request,
    name: str,
    filename: str,
    new_name: str = Form(...),
):
    """話者ラベル（ファイル名の拡張子なし部分）を変更する。"""
    try:
        vdb.rename_speaker(name, filename, new_name)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))
    speakers = vdb.list_speakers(name)
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": name, "speakers": speakers},
    )


@app.get("/api/databases/{name}/speakers/{filename}/audio")
async def api_speaker_audio(name: str, filename: str):
    try:
        path = vdb.speaker_path(name, filename)
    except (ValueError, FileNotFoundError):
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(
        path=path,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/databases/{name}/speakers/{filename}/trim", response_class=HTMLResponse)
async def api_trim_speaker(
    request: Request,
    name: str,
    filename: str,
    start: str = Form(""),
    end: str = Form(""),
):
    """登録済み話者ファイルを指定範囲で切り出して上書き保存する（純粋な声だけ残す）。"""
    try:
        path = vdb.speaker_path(name, filename)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, str(e))

    cs = _parse_opt_float(start)
    ce = _parse_opt_float(end)
    if (cs is None or cs <= 0) and ce is None:
        return _render_error(request, "切り出し範囲が指定されていません。")
    if cs is not None and ce is not None and ce <= cs:
        return _render_error(request, "終了時間は開始時間より後にしてください。")

    # 一時ファイルに切り出してから上書き（同じ拡張子を維持）
    tmp = TEMP_DIR / f"_trim_{uuid.uuid4().hex}{path.suffix}"
    try:
        _crop_audio_file(path, tmp, cs, ce, to_wav16k=False)
        shutil.move(str(tmp), str(path))
    except subprocess.CalledProcessError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        return _render_error(request, f"音声の切り出しに失敗しました: {e.stderr}")

    speakers = vdb.list_speakers(name)
    return templates.TemplateResponse(
        request,
        "partials/db_speakers.html",
        {"db_name": name, "speakers": speakers},
    )


@app.get("/databases", response_class=HTMLResponse)
async def databases_page(request: Request):
    return templates.TemplateResponse(
        request,
        "databases.html",
        {"databases": vdb.list_databases(), "root": str(vdb.get_root())},
    )


async def _render_db_list_fragment(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/db_list.html",
        {"databases": vdb.list_databases()},
    )


@app.get("/api/databases/list", response_class=HTMLResponse)
async def api_db_list_fragment(request: Request):
    return await _render_db_list_fragment(request)


@app.get("/api/databases/select-options", response_class=HTMLResponse)
async def api_db_select_options(request: Request, selected: str = ""):
    return templates.TemplateResponse(
        request,
        "partials/db_select_options.html",
        {"databases": vdb.list_databases(), "selected": selected},
    )


# ===================================================================
# Unknown 話者ラベル付け (事後ラベル付け / B2方式)
# ===================================================================
@app.get("/unknowns/{job_id}", response_class=HTMLResponse)
async def unknowns_page(request: Request, job_id: str):
    job = _load_job(job_id)
    if job is None:
        return _render_error(request, f"ジョブが見つかりません: {job_id}")
    return templates.TemplateResponse(
        request,
        "unknowns.html",
        {
            "job": job,
            "databases": vdb.list_databases(),
        },
    )


@app.get("/unknowns/{job_id}/clip/{cluster_id}")
async def unknowns_clip(job_id: str, cluster_id: str):
    job = _load_job(job_id)
    if job is None:
        return HTMLResponse("Job not found", status_code=404)
    safe_cluster = Path(cluster_id).name
    if safe_cluster != cluster_id:
        return HTMLResponse("Invalid cluster_id", status_code=400)
    clip_path = _job_dir(job_id) / f"clip_{safe_cluster}.wav"
    if not clip_path.exists():
        return HTMLResponse("Clip not found", status_code=404)
    return FileResponse(path=clip_path, media_type="audio/wav")


@app.post("/unknowns/{job_id}/label/{cluster_id}", response_class=HTMLResponse)
async def unknowns_label(
    request: Request,
    job_id: str,
    cluster_id: str,
    speaker_name: str = Form(...),
    db_name: str = Form(...),
    new_db_name: str = Form(""),
    hf_token_override: str = Form(""),
    clip_start: str = Form(""),
    clip_end: str = Form(""),
):
    job = _load_job(job_id)
    if job is None:
        return _render_error(request, f"ジョブが見つかりません: {job_id}")

    # 対象クラスタを検索
    target = next((c for c in job["clusters"] if c["cluster_id"] == cluster_id), None)
    if target is None:
        return _render_error(request, f"クラスタが見つかりません: {cluster_id}")
    if target.get("resolved"):
        return _render_error(request, f"クラスタ {cluster_id} は既にラベル付け済みです。")

    # 名前を検証
    safe_name = vdb.sanitize_name(speaker_name)
    if safe_name is None:
        return _render_error(request, "話者名が無効です（空 or 使用不可文字）。")

    # 新規DB作成モード
    if db_name == "__new__":
        safe_new = vdb.sanitize_name(new_db_name)
        if safe_new is None:
            return _render_error(request, "新規データベース名が無効です（空 or 使用不可文字）。")
        # 同名の既存DBがあれば確認バナーを返して処理を中断
        if (vdb.get_root() / safe_new).is_dir():
            return templates.TemplateResponse(
                request,
                "partials/unknowns_list.html",
                {
                    "job": job,
                    "databases": vdb.list_databases(),
                    "confirm_existing": {
                        "db_name": safe_new,
                        "speaker_name": safe_name,
                        "cluster_id": cluster_id,
                    },
                },
            )
        # 同名が無ければ新規作成
        try:
            vdb.create_database(safe_new)
        except ValueError as e:
            return _render_error(request, f"DB作成エラー: {e}")
        db_name = safe_new

    # DB 検証
    try:
        db_dir = vdb.database_dir(db_name)
    except (ValueError, FileNotFoundError) as e:
        return _render_error(request, f"DBエラー: {e}")

    # クリップを DB に保存（必要なら指定範囲だけ切り出してから保存）
    clip_path = _job_dir(job_id) / target["clip_filename"]
    if not clip_path.exists():
        return _render_error(request, f"クラスタ音声が見つかりません: {clip_path}")

    cs = _parse_opt_float(clip_start)
    ce = _parse_opt_float(clip_end)
    need_crop = (cs is not None and cs > 0) or (ce is not None)
    if need_crop and cs is not None and ce is not None and ce <= cs:
        return _render_error(request, "終了時間は開始時間より後にしてください。")

    clip_for_save = clip_path
    tmp_crop: Optional[Path] = None
    if need_crop:
        tmp_crop = _job_dir(job_id) / f"_cropsave_{cluster_id}.wav"
        try:
            _crop_audio_file(clip_path, tmp_crop, cs, ce, to_wav16k=True)
        except subprocess.CalledProcessError as e:
            return _render_error(request, f"音声の切り出しに失敗しました: {e.stderr}")
        clip_for_save = tmp_crop

    try:
        vdb.add_speaker_file(db_name, clip_for_save, dest_filename=f"{safe_name}.wav")
    except ValueError as e:
        return _render_error(request, f"DB保存エラー: {e}")
    finally:
        if tmp_crop is not None:
            try:
                tmp_crop.unlink()
            except OSError:
                pass

    # SpeakerIdentifier を読み直して新DB全体を登録
    hf_token = hf_token_override or os.getenv("HF_TOKEN", "")
    if not hf_token:
        return _render_error(request, "Hugging Face Token が設定されていません。")
    try:
        identifier = get_cached_speaker_identifier(
            model_name=job.get("embedding_model", "pyannote/embedding"),
            hf_token=hf_token,
            threshold=float(job.get("threshold", 0.5)),
        )
        for name, path in collect_registry_files(db_dir).items():
            identifier.register_speaker(name, path)
    except Exception as e:
        logging.exception("identifier reload failed")
        return _render_error(request, f"声紋モデルの読み込みエラー: {e}")

    # CSV 置換マッピング: まず手動でラベル付けしたものを追加
    csv_mapping: Dict[str, Tuple[str, Optional[float]]] = {
        target["unknown_label"]: (safe_name, None)
    }
    # 手動ラベル付け済みクラスタの記録
    target["resolved"] = True
    target["resolved_name"] = safe_name
    target["resolved_method"] = "manual"

    # 残りの Unknown を新DBで再照合
    auto_resolved_messages = []
    for c in job["clusters"]:
        if c.get("resolved"):
            continue
        rclip = _job_dir(job_id) / c["clip_filename"]
        if not rclip.exists():
            continue
        try:
            ident_name, dist, _cands = identifier.identify_from_audio_path(rclip)
        except Exception as e:
            logging.warning(f"再照合失敗 (cluster {c['cluster_id']}): {e}")
            continue
        if ident_name.startswith("Unknown_"):
            # 閾値以下に到達しなかった = まだ Unknown
            # 距離だけ更新（情報として有用）
            c["distance"] = dist
            continue
        # 閾値以下で確定
        c["resolved"] = True
        c["resolved_name"] = ident_name
        c["resolved_method"] = "auto"
        c["distance"] = dist
        csv_mapping[c["unknown_label"]] = (ident_name, dist)
        auto_resolved_messages.append(
            f"{c['unknown_label']} → {ident_name} (距離 {dist:.4f})"
        )

    # CSV を更新
    csv_path = Path(job["csv_path"])
    replaced = _relabel_csv(csv_path, csv_mapping)
    # SRT を再生成（speaker prefix を最新化）
    srt_path_str = job.get("srt_path")
    if srt_path_str:
        try:
            _csv_to_srt_with_speaker(csv_path, Path(srt_path_str))
        except Exception:
            logging.exception("SRT 再生成に失敗しました")
    job["db_name"] = db_name  # 以降のラベル付けでもデフォルトに使う
    _save_job(job_id, job)

    return templates.TemplateResponse(
        request,
        "partials/unknowns_list.html",
        {
            "job": job,
            "databases": vdb.list_databases(),
            "last_message": (
                f"✓ {target['unknown_label']} を「{safe_name}」として登録しました "
                f"(CSV {replaced}行を更新)"
                + (
                    " · 自動再照合で確定: " + ", ".join(auto_resolved_messages)
                    if auto_resolved_messages else ""
                )
            ),
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
