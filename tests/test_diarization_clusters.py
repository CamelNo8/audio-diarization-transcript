"""未知話者クラスタの永続化の特性テスト。

リファクタリング前の ``audio_processor.py`` の振る舞いを固定する。
分割後の ``src/diarization/clusters.py`` に対して同じ期待値を保つ。
pyannote / Whisper は呼ばず、クラスタの状態遷移とファイル出力だけを検証する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pyannote.core import Segment

from src.diarization import clusters
from src.diarization.clusters import ClusterAssignments
from src.diarization.processor import create_transcript_csv_path


@pytest.fixture
def 割り当て():
    return ClusterAssignments()


@pytest.fixture
def 一時WAV(tmp_path):
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"dummy")
    return wav


class Test話者メタ情報の記録:
    """set_speaker — 名前・距離・全候補距離を同時に更新する。"""

    def test_3つのマッピングが同時に更新される(self, 割り当て):
        割り当て.set_speaker("SPEAKER_00", "太郎", 0.12, {"太郎": 0.12})

        assert 割り当て.speaker_mapping["SPEAKER_00"] == "太郎"
        assert 割り当て.distance_mapping["SPEAKER_00"] == 0.12
        assert 割り当て.candidate_distance_mapping["SPEAKER_00"] == {"太郎": 0.12}

    def test_距離が未確定ならNoneが入る(self, 割り当て):
        割り当て.set_speaker("SPEAKER_00", "Unknown_01", None, None)

        assert 割り当て.distance_mapping["SPEAKER_00"] is None
        assert 割り当て.candidate_distance_mapping["SPEAKER_00"] is None

    def test_Unknownのクラスタだけが記録順に取り出される(self, 割り当て):
        割り当て.set_speaker("SPEAKER_02", "Unknown_02", None, None)
        割り当て.set_speaker("SPEAKER_00", "太郎", 0.1, None)
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)

        assert 割り当て.unknown_cluster_ids() == ["SPEAKER_02", "SPEAKER_01"]


class Test代表区間の選択:
    """pick_representative_segment — 照合に使う区間を1つ選ぶ。"""

    def test_1秒以上の区間のうち最長が選ばれる(self):
        区間 = [Segment(0.0, 1.5), Segment(2.0, 5.0), Segment(6.0, 6.5)]

        assert clusters.pick_representative_segment(区間) == Segment(2.0, 5.0)

    def test_1秒以上が無ければ全体の最長が選ばれる(self):
        区間 = [Segment(0.0, 0.3), Segment(2.0, 2.8)]

        assert clusters.pick_representative_segment(区間) == Segment(2.0, 2.8)

    def test_境界としてちょうど1秒の区間は優先対象に含まれる(self):
        区間 = [Segment(0.0, 1.0), Segment(2.0, 2.9)]

        assert clusters.pick_representative_segment(区間) == Segment(0.0, 1.0)


class Test未知クラスタの永続化:
    """persist_unknown_clusters — Web UI のラベル付け用に代表音声を書き出す。"""

    @pytest.fixture
    def 切り出しを記録(self, monkeypatch):
        """extract_audio を差し替え、呼び出し内容を記録しつつ空ファイルを作る。"""
        呼び出し = []

        def _fake(src, dst, start=None, end=None, quiet=True):
            呼び出し.append(
                {"src": Path(src), "dst": Path(dst), "start": start, "end": end}
            )
            Path(dst).write_bytes(b"clip")

        monkeypatch.setattr("src.diarization.clusters.extract_audio", _fake)
        return 呼び出し

    def test_Unknownのクラスタだけが書き出される(
        self, 割り当て, 一時WAV, tmp_path, 切り出しを記録
    ):
        割り当て.set_speaker("SPEAKER_00", "太郎", 0.1, {"太郎": 0.1})
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", 0.9, {"太郎": 0.9})
        割り当て.segments = {
            "SPEAKER_00": Segment(0.0, 2.0),
            "SPEAKER_01": Segment(3.0, 5.5),
        }

        結果 = clusters.persist_unknown_clusters(割り当て, 一時WAV, tmp_path / "c")

        assert len(結果) == 1
        assert 結果[0]["cluster_id"] == "SPEAKER_01"
        assert 結果[0]["unknown_label"] == "Unknown_01"

    def test_メタ情報に距離と代表区間が入る(
        self, 割り当て, 一時WAV, tmp_path, 切り出しを記録
    ):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", 0.9, {"太郎": 0.9})
        割り当て.segments = {"SPEAKER_01": Segment(3.0, 5.5)}

        項目 = clusters.persist_unknown_clusters(割り当て, 一時WAV, tmp_path / "c")[0]

        assert 項目["distance"] == 0.9
        assert 項目["candidate_distances"] == {"太郎": 0.9}
        assert 項目["segment_start"] == 3.0
        assert 項目["segment_end"] == 5.5
        assert 項目["clip_filename"] == "clip_SPEAKER_01.wav"

    def test_代表音声が出力ディレクトリに書き出される(
        self, 割り当て, 一時WAV, tmp_path, 切り出しを記録
    ):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)
        割り当て.segments = {"SPEAKER_01": Segment(3.0, 5.5)}
        出力先 = tmp_path / "clusters"

        clusters.persist_unknown_clusters(割り当て, 一時WAV, 出力先)

        assert (出力先 / "clip_SPEAKER_01.wav").exists()
        assert 切り出しを記録[0]["start"] == 3.0
        assert 切り出しを記録[0]["end"] == 5.5

    def test_代表区間が無いクラスタは飛ばされる(
        self, 割り当て, 一時WAV, tmp_path, 切り出しを記録
    ):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)

        assert (
            clusters.persist_unknown_clusters(割り当て, 一時WAV, tmp_path / "c") == []
        )

    def test_切り出しに失敗したクラスタは飛ばされる(
        self, 割り当て, 一時WAV, tmp_path, monkeypatch
    ):
        def _fail(src, dst, start=None, end=None, quiet=True):
            raise subprocess.CalledProcessError(1, "ffmpeg", stderr="失敗")

        monkeypatch.setattr("src.diarization.clusters.extract_audio", _fail)
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)
        割り当て.segments = {"SPEAKER_01": Segment(3.0, 5.5)}

        assert (
            clusters.persist_unknown_clusters(割り当て, 一時WAV, tmp_path / "c") == []
        )

    def test_一時WAVが無ければ空リストを返す(self, 割り当て, tmp_path):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)

        assert clusters.persist_unknown_clusters(割り当て, None, tmp_path / "c") == []


class Test文字起こしCSVパスの生成:
    """create_transcript_csv_path — 音声名＋日時から出力先を決める。"""

    def test_音声のstemとtranscriptionを含む名前になる(self, tmp_path):
        path = create_transcript_csv_path(tmp_path / "会議録音.wav")

        assert path.name.startswith("会議録音-transcription-")
        assert path.suffix == ".csv"

    def test_カレントディレクトリ直下に作られる(self, tmp_path):
        path = create_transcript_csv_path(tmp_path / "sub" / "audio.m4a")

        assert path.parent == Path.cwd()
