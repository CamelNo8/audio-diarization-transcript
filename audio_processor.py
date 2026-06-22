import numpy as np
import torch

# PyTorch 2.6+ で torch.load のデフォルトが weights_only=True に変わり、
# pyannote のチェックポイント (pytorch-lightning callback 等を pickle 含む) が
# 読めなくなった対策。pyannote/lightning は torch.load を weights_only=True 付きで
# 明示呼び出しするため、setdefault ではなく強制上書きする必要がある。
# pyannote モデルは Hugging Face 由来で信頼可能なため安全。
if not getattr(torch.load, "_pyannote_compat_patched", False):
    _ORIGINAL_TORCH_LOAD = torch.load
    def _torch_load_compat(*args, **kwargs):
        kwargs["weights_only"] = False  # 強制上書き
        return _ORIGINAL_TORCH_LOAD(*args, **kwargs)
    _torch_load_compat._pyannote_compat_patched = True
    torch.load = _torch_load_compat

from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio.core.io import Audio
from pathlib import Path
import warnings
import logging
from typing import Optional, Dict
import csv
import datetime
import subprocess
import tempfile
import shutil
import os

from speaker_identification import SpeakerIdentifier
import transcription_backend

# Hugging Face トークンに関するFutureWarningを抑制
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


def format_time(seconds: float) -> str:
    """秒数を HH:MM:SS:ms 形式（ミリ秒は3桁）に丸めて変換します。"""
    if seconds < 0:
        return "00:00:00:000"
    total_ms = int(round(seconds * 1000))
    hours, remainder_ms = divmod(total_ms, 3600 * 1000)
    minutes, remainder_ms = divmod(remainder_ms, 60 * 1000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}:{millis:03}"

def create_transcript_csv_path(audio_file_path: Path) -> Path:
    """指定された音声ファイルパスから、出力CSVファイルのPathを生成します。"""
    base_name = audio_file_path.stem
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d%H%M%S")
    output_filename = f"{base_name}-transcription-{timestamp_str}.csv"
    return Path.cwd() / output_filename


class AudioProcessor:
    """音声ファイルの前処理、話者分離・特定、文字起こしを行い、結果をCSVに出力するクラス。"""

    _PIPELINE_CACHE: Dict[tuple[str, str], Pipeline] = {}  # [OPTIMIZED]

    def __init__(
        self,
        audio_file: Path,
        output_csv_path: Path,
        mlx_model_id: str,
        pyannote_model_id: str,
        hf_token: str,
        identifier: Optional[SpeakerIdentifier] = None,
        registry_dir: Optional[Path] = None,
        interactive_unknown_resolve: bool = True,
        denoise: bool = False,
        separator_model: Optional[str] = None,
        whisper_backend: str = "auto",
    ):
        self.audio_file = audio_file
        self.output_csv_path = output_csv_path
        self.mlx_model_id = mlx_model_id
        self.whisper_backend = whisper_backend
        self.pyannote_model_id = pyannote_model_id
        self.hf_token = hf_token
        self.identifier = identifier
        self.registry_dir = registry_dir
        self.interactive_unknown_resolve = interactive_unknown_resolve
        self.denoise = denoise
        self.separator_model = separator_model

        self.temp_wav_path: Optional[Path] = None
        self.speaker_mapping: Dict[str, str] = {}  # クラスターID -> 話者名
        self.speaker_distance_mapping: Dict[str, Optional[float]] = {}  # クラスターID -> コサイン距離
        self.speaker_candidate_distance_mapping: Dict[str, Optional[Dict[str, float]]] = {}
        self._cluster_segments: Dict[str, Segment] = {}  # クラスターID -> 代表音声区間
        self._cluster_embeddings: Dict[str, np.ndarray] = {}  # クラスターID -> 抽出済み埋め込み

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.cleanup()

    def cleanup(self):
        if self.temp_wav_path and self.temp_wav_path.exists():
            try:
                logging.info(f"Cleaning up temporary file: {self.temp_wav_path}")
                self.temp_wav_path.unlink()
            except Exception as e:
                logging.error(f"Failed to delete temporary file: {e}")

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # [OPTIMIZED]
    @classmethod
    def _get_cached_pipeline(
        cls, model_id: str, hf_token: str, device: torch.device
    ) -> Pipeline:
        cache_key = (model_id, hf_token)
        pipeline = cls._PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            logging.info(f"Loading Pyannote pipeline ({model_id})...")  # [OPTIMIZED]
            pipeline = cls._load_pipeline_with_auth(model_id, hf_token)
            cls._PIPELINE_CACHE[cache_key] = pipeline  # [OPTIMIZED]
        pipeline.to(device)
        return pipeline

    @staticmethod
    def _load_pipeline_with_auth(model_id: str, hf_token: str) -> Pipeline:
        """新旧 huggingface_hub / pyannote-audio 両対応で Pipeline をロードする。

        - token / use_auth_token どちらが受け入れられるか
        - local_files_only オプションがサポートされるか
        が pyannote のバージョンで変わるため、組み合わせを順次試行する。
        """
        last_error: Optional[Exception] = None
        # 優先度順: 引数が新しい (token) / オフライン優先 → 引数が古い / オンライン
        attempts = [
            {"token": hf_token, "local_files_only": True},
            {"token": hf_token},
            {"use_auth_token": hf_token, "local_files_only": True},
            {"use_auth_token": hf_token},
        ]
        for kwargs in attempts:
            try:
                return Pipeline.from_pretrained(model_id, **kwargs)
            except TypeError as e:
                # 引数自体が受け付けられない → 次の組み合わせへ
                last_error = e
                continue
            except Exception as e:
                # ネットワーク不通等 → 次の組み合わせへ
                last_error = e
                continue
        raise RuntimeError(f"Pipeline.from_pretrained に失敗しました: {last_error}")

    def _set_speaker_metadata(
        self,
        cluster_id: str,
        speaker_name: str,
        distance: Optional[float],
        candidate_distances: Optional[Dict[str, float]],
    ) -> None:
        self.speaker_mapping[cluster_id] = speaker_name
        self.speaker_distance_mapping[cluster_id] = distance
        self.speaker_candidate_distance_mapping[cluster_id] = candidate_distances

    def prepare_audio(self):
        """ffmpegを使用して任意の音声/動画ファイルをWAV形式の一時ファイルに変換します。

        denoise=True の場合は変換後に Demucs でボーカルを抽出し、背景音/BGM を除去します。
        """
        fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.temp_wav_path = Path(temp_path_str)

        logging.info(f"Converting audio to temporary WAV format: {self.temp_wav_path}")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(self.audio_file),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-y",
                    str(self.temp_wav_path)
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            logging.info("Audio conversion successful.")
        except subprocess.CalledProcessError as e:
            logging.critical(f"FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

        if self.denoise:
            self._extract_vocals_inplace()

    # audio-separator のデフォルト分離モデル
    # （HuggingFace の karaokenerds/all-uvr-models から取得）
    DEFAULT_SEPARATOR_MODEL = "Kim_Vocal_2.onnx"

    def _extract_vocals_inplace(self) -> None:
        """audio-separator (UVR / MDX / Roformer 系) でボーカルを抽出し、
        self.temp_wav_path を抽出後のファイルに差し替える。

        失敗した場合は元の WAV のまま処理を続行する（致命的エラーにはしない）。
        モデルは HuggingFace 経由でダウンロードされるため、FB CDN が不通でも動作する。
        """
        assert self.temp_wav_path is not None
        model_name = self.separator_model or self.DEFAULT_SEPARATOR_MODEL
        logging.info(f"audio-separator で背景音を除去中（model={model_name}）...")

        try:
            from audio_separator.separator import Separator  # type: ignore
        except ImportError:
            logging.warning(
                "audio-separator が見つかりません。`uv sync` で依存をインストールしてください。"
                "元音声で処理を続行します。"
            )
            return

        outdir = self.temp_wav_path.parent / f"sep_{self.temp_wav_path.stem}"
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            separator = Separator(
                output_dir=str(outdir),
                log_level=logging.WARNING,
            )
            separator.load_model(model_filename=model_name)
            output_files = separator.separate(str(self.temp_wav_path))
        except Exception as e:
            logging.warning(f"audio-separator 失敗。元音声で続行します。\n{e}")
            shutil.rmtree(outdir, ignore_errors=True)
            return

        # output_files は ["...(Vocals)....wav", "...(Instrumental)....wav"] 形式
        vocals_file = next(
            (f for f in (output_files or []) if "vocal" in Path(f).name.lower()),
            None,
        )
        if not vocals_file:
            logging.warning(f"ボーカル抽出ファイルが見つかりません: {output_files}")
            shutil.rmtree(outdir, ignore_errors=True)
            return

        vocals_path = Path(vocals_file)
        if not vocals_path.is_absolute():
            vocals_path = outdir / vocals_path.name
        if not vocals_path.exists():
            # 念のためフォールバック検索
            cands = list(outdir.rglob("*[Vv]ocals*.wav"))
            if not cands:
                logging.warning(f"ボーカル WAV が見当たりません: {outdir}")
                shutil.rmtree(outdir, ignore_errors=True)
                return
            vocals_path = cands[0]

        # 16kHz mono へ再変換しつつ元 WAV を置換
        fd2, replaced_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd2)
        replaced_path = Path(replaced_str)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(vocals_path),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(replaced_path),
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"ボーカル出力の 16kHz 変換に失敗。元音声で続行します。\n{e.stderr}")
            shutil.rmtree(outdir, ignore_errors=True)
            try:
                replaced_path.unlink()
            except OSError:
                pass
            return

        # 既存 temp_wav_path を置き換え
        try:
            self.temp_wav_path.unlink()
        except OSError:
            pass
        self.temp_wav_path = replaced_path
        shutil.rmtree(outdir, ignore_errors=True)
        logging.info(f"ボーカル抽出済み: {self.temp_wav_path}")

    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        """全体のプロセス（前処理、話者分離・特定、文字起こし、マージ、CSV保存）を実行します。"""
        # 1. 音声の前処理 (WAV変換)
        self.prepare_audio()

        # 2. 話者分離 (Pyannote Diarization)
        try:
            # use_auth_tokenパラメータを使用して初期化
            pipeline = self._get_cached_pipeline(  # [OPTIMIZED]
                self.pyannote_model_id,
                self.hf_token,
                self._select_device(),
            )
        except Exception as e:
            logging.critical(f"Error loading Pyannote pipeline: {e}")  # [OPTIMIZED]
            return False

        logging.info("Running speaker diarization...")
        diarization_params = {}
        if known_num_speakers is not None:
            diarization_params["num_speakers"] = known_num_speakers

        try:
            diarization = pipeline(str(self.temp_wav_path), **diarization_params)
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}")
            return False

        # 3. 各クラスターごとの話者照合・特定
        if self.identifier:
            logging.info("Identifying speakers for each cluster...")
            audio_io = Audio()
            cluster_segments = []  # [OPTIMIZED]
            for cluster_id in diarization.labels():
                segments = diarization.label_timeline(cluster_id)
                # 特徴量抽出の安定性を高めるため、1秒以上のセグメントを優先
                valid_segments = [s for s in segments if s.duration >= 1.0]
                longest_segment = max(valid_segments, key=lambda s: s.duration) if valid_segments else max(segments, key=lambda s: s.duration)
                cluster_segments.append((cluster_id, longest_segment))  # [OPTIMIZED]
                self._cluster_segments[cluster_id] = longest_segment

            waveforms = []  # [OPTIMIZED]
            for cluster_id, longest_segment in cluster_segments:
                try:
                    # クラスターの代表音声区間から波形を取得
                    waveform, sr = audio_io.crop(str(self.temp_wav_path), longest_segment)
                    waveforms.append((cluster_id, waveform, sr))  # [OPTIMIZED]
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        self.identifier._next_unknown_name(),
                        None,
                        None,
                    )

            embeddings = []  # [OPTIMIZED]
            for cluster_id, waveform, sr in waveforms:
                try:
                    embedding = self.identifier.get_embedding_from_waveform(waveform, sr)
                    embeddings.append((cluster_id, embedding))  # [OPTIMIZED]
                    self._cluster_embeddings[cluster_id] = embedding
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        self.identifier._next_unknown_name(),
                        None,
                        None,
                    )

            for cluster_id, embedding in embeddings:
                try:
                    identified_name, cosine_distance, candidate_distances = (
                        self.identifier.identify_speaker_with_distances(embedding)
                    )

                    self._set_speaker_metadata(
                        cluster_id,
                        identified_name,
                        cosine_distance,
                        candidate_distances,
                    )
                    logging.info(
                        f"Cluster '{cluster_id}' identified as -> {identified_name} "
                        f"(cosine_distance={cosine_distance:.6f})"
                        if cosine_distance is not None
                        else f"Cluster '{cluster_id}' identified as -> {identified_name}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to identify speaker for cluster {cluster_id}: {e}")
                    self._set_speaker_metadata(
                        cluster_id,
                        self.identifier._next_unknown_name(),
                        None,
                        None,
                    )

            # 4. Unknown と判定されたクラスタを対話的に登録（CLI のみ）
            if self.registry_dir is not None and self.interactive_unknown_resolve:
                self._resolve_unknown_speakers_interactively()
        else:
            # 識別器が設定されていない場合は全クラスタを Unknown_NN として扱う
            # ラベル付け用に代表音声区間も記録しておく（identifier 分岐と同じロジック）
            for i, cluster_id in enumerate(sorted(diarization.labels()), start=1):
                segments = diarization.label_timeline(cluster_id)
                valid_segments = [s for s in segments if s.duration >= 1.0]
                if valid_segments:
                    longest_segment = max(valid_segments, key=lambda s: s.duration)
                else:
                    longest_segment = max(segments, key=lambda s: s.duration)
                self._cluster_segments[cluster_id] = longest_segment
                self._set_speaker_metadata(cluster_id, f"Unknown_{i:02d}", None, None)

        # 4. 全文一括文字起こし（プラットフォームに応じて mlx-whisper / faster-whisper）
        try:
            whisper_result = transcription_backend.transcribe_full(
                self.temp_wav_path,
                model_id=self.mlx_model_id,
                language="ja",
                prefer_device=self._select_device(),
                backend=self.whisper_backend,
            )
        except Exception as e:
            logging.error(f"Error during whisper transcription: {e}", exc_info=True)
            return False

        segments = whisper_result.get("segments", [])
        if not segments:
            logging.warning("No speech segments detected by Whisper.")
            return True

        # 5. マージ処理とCSV出力
        logging.info(f"Merging results and writing to {self.output_csv_path}...")
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                csv_writer.writerow(["start", "end", "speaker", "text", "cosine_distance"])

                for seg in segments:
                    w_start, w_end, w_text = seg["start"], seg["end"], seg["text"].strip()
                    if not w_text:
                        continue

                    w_segment = Segment(w_start, w_end)
                    speaker_durations = {}

                    # pyannoteのDiarization結果と被る長さを計算
                    for p_seg, _, cluster_id in diarization.itertracks(yield_label=True):
                        overlap = w_segment & p_seg
                        if overlap:
                            speaker_durations[cluster_id] = speaker_durations.get(cluster_id, 0.0) + overlap.duration

                    # 最も重複時間が長いクラスターを選び、話者名に変換
                    if speaker_durations:
                        best_cluster = max(speaker_durations, key=speaker_durations.get)
                        best_speaker = self.speaker_mapping.get(best_cluster, best_cluster)
                        best_distance = self.speaker_distance_mapping.get(best_cluster)
                        candidate_distances = self.speaker_candidate_distance_mapping.get(best_cluster)
                    else:
                        best_cluster = ""
                        best_speaker = "Unknown"
                        best_distance = None
                        candidate_distances = None

                    start_str = format_time(w_start)
                    end_str = format_time(w_end)
                    cosine_distance_str = f"{best_distance:.6f}" if best_distance is not None else ""

                    csv_writer.writerow([start_str, end_str, best_speaker, w_text, cosine_distance_str])
                    logging.info(
                        f"  [{start_str} - {end_str}] {best_speaker}: {w_text} "
                        f"(cosine_distance={cosine_distance_str or 'N/A'})"
                    )
                    if candidate_distances:
                        sorted_candidates = sorted(
                            candidate_distances.items(), key=lambda item: item[1]
                        )
                        candidates_str = ", ".join(
                            f"{name}={dist:.6f}" for name, dist in sorted_candidates
                        )
                    else:
                        candidates_str = "N/A"

                    print(
                        f"  [{start_str} - {end_str}] cluster={best_cluster} "
                        f"speaker={best_speaker} candidates: {candidates_str}"
                    )

            logging.info(f"Successfully finished writing results to {self.output_csv_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return False

    # ------------------------------------------------------------------
    # API 向け: 背景音除去 + 話者分離 + 文字起こしまでを実行し、
    # 話者照合（声紋DB）と CSV 出力は行わず、結果を dict で返す。
    # 照合は PC 側が vocals WAV を使って行う。
    # ------------------------------------------------------------------
    def process_for_api(
        self,
        known_num_speakers: Optional[int] = None,
        vocals_out: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Spark などのリモート推論サーバー用。

        実行する工程:
          1. 前処理（WAV変換、denoise=True なら audio-separator で背景音除去）
          2. pyannote 話者分離
          3. faster-whisper / mlx-whisper 文字起こし
          4. Whisper セグメントへ pyannote クラスタIDを割り当て（マージ）

        戻り値（JSON 化可能な dict）:
          {
            "segments":   [{"start", "end", "text", "cluster_id"}, ...],
            "clusters":   {cluster_id: {"rep_start", "rep_end"}, ...},  # 照合用代表区間
            "num_speakers": int,
            "vocals_path": str | None,  # vocals_out にコピーした処理済みWAVのパス
          }

        話者名は付与しない（照合は PC 側で実施するため cluster_id のみ返す）。
        """
        # 1. 前処理（denoise=True なら背景音除去まで実施）
        self.prepare_audio()

        # 2. 話者分離
        pipeline = self._get_cached_pipeline(
            self.pyannote_model_id, self.hf_token, self._select_device()
        )
        logging.info("Running speaker diarization (API)...")
        diarization_params = {}
        if known_num_speakers is not None:
            diarization_params["num_speakers"] = known_num_speakers
        diarization = pipeline(str(self.temp_wav_path), **diarization_params)

        # 各クラスタの代表区間（最長セグメント、1秒以上を優先）を記録
        clusters: Dict[str, Dict[str, float]] = {}
        for cluster_id in diarization.labels():
            timeline = diarization.label_timeline(cluster_id)
            valid = [s for s in timeline if s.duration >= 1.0]
            rep = max(valid or list(timeline), key=lambda s: s.duration)
            clusters[cluster_id] = {"rep_start": float(rep.start), "rep_end": float(rep.end)}

        # 3. 文字起こし
        whisper_result = transcription_backend.transcribe_full(
            self.temp_wav_path,
            model_id=self.mlx_model_id,
            language="ja",
            prefer_device=self._select_device(),
            backend=self.whisper_backend,
        )

        # 4. マージ（Whisper セグメント → 最重複クラスタID）
        out_segments = []
        for seg in whisper_result.get("segments", []):
            w_start, w_end = float(seg["start"]), float(seg["end"])
            w_text = seg["text"].strip()
            if not w_text:
                continue
            w_segment = Segment(w_start, w_end)
            durations: Dict[str, float] = {}
            for p_seg, _, cluster_id in diarization.itertracks(yield_label=True):
                overlap = w_segment & p_seg
                if overlap:
                    durations[cluster_id] = durations.get(cluster_id, 0.0) + overlap.duration
            best_cluster = max(durations, key=durations.get) if durations else None
            out_segments.append(
                {"start": w_start, "end": w_end, "text": w_text, "cluster_id": best_cluster}
            )

        # 処理済み（背景音除去済み）WAV を照合用に確定パスへコピー
        vocals_path: Optional[str] = None
        if vocals_out is not None and self.temp_wav_path is not None:
            vocals_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.temp_wav_path, vocals_out)
            vocals_path = str(vocals_out)

        return {
            "segments": out_segments,
            "clusters": clusters,
            "num_speakers": len(clusters),
            "vocals_path": vocals_path,
        }

    # ------------------------------------------------------------------
    # Unknown クラスタの永続化（Web UI 向け）
    # ------------------------------------------------------------------

    def persist_unknown_clusters(self, output_dir: Path) -> list:
        """Unknown と判定されたクラスタの代表音声を output_dir に保存し、
        メタ情報のリストを返す。Web UI で事後ラベル付けするために使用。

        戻り値の各要素: {
            "cluster_id": str,
            "unknown_label": str,       # 例 "Unknown_01"
            "distance": Optional[float],
            "candidate_distances": Optional[Dict[str, float]],
            "clip_filename": str,
            "segment_start": float,
            "segment_end": float,
        }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.temp_wav_path is None or not self.temp_wav_path.exists():
            logging.warning("一時 WAV が無いため Unknown クラスタを永続化できません。")
            return []

        result = []
        for cluster_id, label in self.speaker_mapping.items():
            if not label.startswith("Unknown_"):
                continue
            segment = self._cluster_segments.get(cluster_id)
            if segment is None:
                continue

            clip_filename = f"clip_{cluster_id}.wav"
            clip_path = output_dir / clip_filename
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-nostdin", "-loglevel", "error",
                        "-ss", f"{segment.start:.3f}",
                        "-to", f"{segment.end:.3f}",
                        "-i", str(self.temp_wav_path),
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", "-y",
                        str(clip_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logging.warning(f"クラスタ {cluster_id} の音声切り出し失敗: {e.stderr}")
                continue

            result.append({
                "cluster_id": cluster_id,
                "unknown_label": label,
                "distance": self.speaker_distance_mapping.get(cluster_id),
                "candidate_distances": self.speaker_candidate_distance_mapping.get(cluster_id),
                "clip_filename": clip_filename,
                "segment_start": float(segment.start),
                "segment_end": float(segment.end),
            })
        return result

    # ------------------------------------------------------------------
    # Unknown 話者の対話的登録
    # ------------------------------------------------------------------

    _INVALID_NAME_CHARS = set('/\\:*?"<>|')

    def _resolve_unknown_speakers_interactively(self) -> None:
        """speaker_mapping のうち Unknown_NN として残っているクラスタについて、
        ユーザーに代表音声を聞かせて名前を入力してもらい、声紋を永続登録する。
        登録するたびに残りの Unknown クラスタを再照合する。"""

        if self.identifier is None or self.registry_dir is None:
            return
        if self.temp_wav_path is None or not self.temp_wav_path.exists():
            logging.warning("一時 WAV が無いため対話的登録をスキップします。")
            return

        unknown_clusters = sorted(
            cid for cid, name in self.speaker_mapping.items()
            if name.startswith("Unknown_")
        )
        if not unknown_clusters:
            return

        if not self.registry_dir.is_dir():
            logging.warning(
                f"registry_dir が存在しないため対話的登録をスキップします: {self.registry_dir}"
            )
            return

        print()
        print("=" * 60)
        print(f"未登録クラスタが {len(unknown_clusters)} 個あります。")
        print("各クラスタの代表音声を再生しますので、声の主の名前を入力してください。")
        print("=" * 60)

        skip_all = False
        for cluster_id in unknown_clusters:
            # 直前の再照合で名前が確定していたらスキップ
            current = self.speaker_mapping.get(cluster_id, "")
            if not current.startswith("Unknown_"):
                continue
            if skip_all:
                continue

            clip_path = self._extract_cluster_audio(cluster_id)
            if clip_path is None:
                logging.warning(
                    f"クラスタ {cluster_id} の音声切り出しに失敗したためスキップします。"
                )
                continue

            try:
                resolved = self._prompt_user_for_speaker(cluster_id, clip_path)
            finally:
                try:
                    clip_path.unlink()
                except OSError:
                    pass

            if resolved is None:
                # スキップ
                continue
            if resolved == "__SKIP_ALL__":
                skip_all = True
                continue

            name, saved_path = resolved
            try:
                self.identifier.register_speaker(name, saved_path)
            except Exception as e:
                logging.error(f"声紋登録に失敗しました ({name}): {e}")
                # 保存ファイルを残しても識別器には未登録なので、ループは続行
                continue

            # 新規登録後、自クラスタの埋め込みを再照合して
            # 全候補との距離 (candidate_distances) を確定させる
            distance, candidates = self._recompute_distances_for_cluster(cluster_id)
            self._set_speaker_metadata(cluster_id, name, distance, candidates)
            dist_str = f"{distance:.6f}" if distance is not None else "N/A"
            print(f"  → クラスタ {cluster_id} を「{name}」として登録しました (cosine_distance={dist_str})。")

            # 残りの Unknown クラスタを新規話者と再照合
            self._remap_remaining_unknowns()

        print("=" * 60)
        print("対話的登録を終了します。")
        print("=" * 60)
        print()

    def _extract_cluster_audio(self, cluster_id: str) -> Optional[Path]:
        """クラスタの代表音声区間を一時 WAV に切り出して返す。"""
        segment = self._cluster_segments.get(cluster_id)
        if segment is None:
            return None

        fd, tmp_str = tempfile.mkstemp(suffix=".wav", prefix=f"cluster_{cluster_id}_")
        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error",
                    "-ss", f"{segment.start:.3f}",
                    "-to", f"{segment.end:.3f}",
                    "-i", str(self.temp_wav_path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-y",
                    str(tmp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"代表音声の切り出しに失敗: {e.stderr}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None
        return tmp_path

    def _play_audio(self, audio_path: Path) -> None:
        """afplay でブロッキング再生。失敗してもエラーにしない。"""
        try:
            subprocess.run(
                ["afplay", str(audio_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"afplay による再生に失敗しました: {e}")

    def _sanitize_speaker_name(self, raw: str) -> Optional[str]:
        name = raw.strip()
        if not name:
            return None
        if any(c in self._INVALID_NAME_CHARS for c in name):
            return None
        return name

    def _prompt_user_for_speaker(
        self, cluster_id: str, clip_path: Path
    ) -> Optional[object]:
        """ユーザーに名前を尋ねる。

        戻り値:
          - None: スキップ
          - "__SKIP_ALL__": 以降全てスキップ
          - (name, saved_path): 名前と永続保存先
        """
        current_label = self.speaker_mapping.get(cluster_id, cluster_id)
        print()
        print(f"--- クラスタ {cluster_id} (現ラベル: {current_label}) ---")
        print(f"代表音声ファイル: {clip_path}")
        self._play_audio(clip_path)

        while True:
            try:
                raw = input(
                    "このクラスタの声は誰ですか？\n"
                    "  名前を入力 / [Enter]=スキップ / [r]=もう一度再生 / [s]=以降全てスキップ: "
                )
            except EOFError:
                return None

            cmd = raw.strip()
            if cmd == "":
                return None
            if cmd.lower() == "r":
                self._play_audio(clip_path)
                continue
            if cmd.lower() == "s":
                return "__SKIP_ALL__"

            name = self._sanitize_speaker_name(cmd)
            if name is None:
                print("  ! 名前が空、または使用できない文字が含まれています。再入力してください。")
                continue

            saved_path = self._persist_registry_audio(name, clip_path)
            if saved_path is None:
                continue
            return (name, saved_path)

    def _persist_registry_audio(self, name: str, clip_path: Path) -> Optional[Path]:
        """切り出した代表音声を registry_dir/<name>.wav として永続保存する。
        既存ファイルがある場合は上書き確認をする。"""
        assert self.registry_dir is not None
        target = self.registry_dir / f"{name}.wav"

        if target.exists():
            try:
                overwrite = input(
                    f"  ! {target} は既に存在します。上書きしますか？ [y/N]: "
                ).strip().lower()
            except EOFError:
                overwrite = "n"
            if overwrite != "y":
                print("  → 別の名前を入力してください。")
                return None

        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            with open(clip_path, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
        except OSError as e:
            logging.error(f"声紋ファイルの保存に失敗しました ({target}): {e}")
            return None
        print(f"  → 声紋ファイルを保存しました: {target}")
        return target

    def _recompute_distances_for_cluster(
        self, cluster_id: str
    ) -> "tuple[Optional[float], Optional[Dict[str, float]]]":
        """対話的に登録したクラスタについて、全登録話者との距離を取り直す。
        candidate_distances を埋めて CSV 出力で N/A にならないようにするのが目的。"""
        if self.identifier is None:
            return None, None
        embedding = self._cluster_embeddings.get(cluster_id)
        if embedding is None:
            return None, None
        try:
            _name, distance, candidates = (
                self.identifier.identify_speaker_with_distances(embedding)
            )
        except Exception as e:
            logging.warning(f"クラスタ {cluster_id} の距離再計算に失敗: {e}")
            return None, None
        return distance, candidates

    def _remap_remaining_unknowns(self) -> None:
        """新たに話者を登録した後、残りの Unknown クラスタの埋め込みを再照合し、
        閾値以下にヒットすれば speaker_mapping を更新する。"""
        if self.identifier is None:
            return

        for cid, current_name in list(self.speaker_mapping.items()):
            if not current_name.startswith("Unknown_"):
                continue
            embedding = self._cluster_embeddings.get(cid)
            if embedding is None:
                continue
            try:
                identified_name, distance, candidates = (
                    self.identifier.identify_speaker_with_distances(embedding)
                )
            except Exception as e:
                logging.warning(f"クラスタ {cid} の再照合に失敗: {e}")
                continue

            if identified_name.startswith("Unknown_"):
                # 再照合で別の Unknown_NN が払い出されないよう、もとのラベルを維持
                continue

            self._set_speaker_metadata(cid, identified_name, distance, candidates)
            print(
                f"  → クラスタ {cid} は再照合の結果「{identified_name}」"
                f" (cosine_distance={distance:.6f}) に確定しました。"
            )
