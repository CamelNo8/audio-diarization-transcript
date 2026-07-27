"""未知話者クラスタの永続化・再照合の特性テスト。

リファクタリング前の ``audio_processor.py`` の振る舞いを固定する。
分割後の ``src/diarization/clusters.py`` に対して同じ期待値を保つ。
pyannote / Whisper は呼ばず、クラスタの状態遷移とファイル出力だけを検証する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from pyannote.core import Segment

from src.diarization import clusters
from src.diarization.clusters import ClusterAssignments
from src.diarization.interactive import sanitize_speaker_name
from src.diarization.processor import create_transcript_csv_path


class _擬似識別器:
    """SpeakerIdentifier の代わり。埋め込み→照合結果の対応表を持つ。"""

    def __init__(self, 照合結果: dict | None = None, 失敗する: bool = False):
        self.照合結果 = 照合結果 or {}
        self.失敗する = 失敗する
        self.unknown_counter = 1

    def identify_speaker_with_distances(self, embedding):
        if self.失敗する:
            raise RuntimeError("照合に失敗しました")
        key = float(embedding[0][0])
        return self.照合結果.get(key, (self._next_unknown_name(), None, None))

    def _next_unknown_name(self) -> str:
        name = f"Unknown_{self.unknown_counter:02d}"
        self.unknown_counter += 1
        return name


def _埋め込み(値: float) -> np.ndarray:
    return np.asarray([[値, 0.0]], dtype=np.float32)


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


class Testクラスタの距離再計算:
    """recompute_distances_for_cluster — 登録直後に全候補距離を埋め直す。"""

    def test_照合結果の距離と候補が返る(self, 割り当て):
        識別器 = _擬似識別器({1.0: ("太郎", 0.05, {"太郎": 0.05})})
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(1.0)}

        assert clusters.recompute_distances_for_cluster(
            識別器, 割り当て, "SPEAKER_01"
        ) == (0.05, {"太郎": 0.05})

    def test_識別器が無ければNoneの組を返す(self, 割り当て):
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(1.0)}

        assert clusters.recompute_distances_for_cluster(
            None, 割り当て, "SPEAKER_01"
        ) == (None, None)

    def test_埋め込みが無ければNoneの組を返す(self, 割り当て):
        assert clusters.recompute_distances_for_cluster(
            _擬似識別器(), 割り当て, "SPEAKER_99"
        ) == (None, None)

    def test_照合が例外を投げてもNoneの組を返す(self, 割り当て):
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(1.0)}

        assert clusters.recompute_distances_for_cluster(
            _擬似識別器(失敗する=True), 割り当て, "SPEAKER_01"
        ) == (None, None)


class Test残りの未知クラスタの再マッピング:
    """remap_remaining_unknowns — 新規登録後に他の Unknown を照合し直す。"""

    def test_閾値内にヒットしたUnknownが実名に置き換わる(self, 割り当て):
        識別器 = _擬似識別器({2.0: ("太郎", 0.2, {"太郎": 0.2})})
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(2.0)}

        更新 = clusters.remap_remaining_unknowns(識別器, 割り当て)

        assert 更新 == [("SPEAKER_01", "太郎", 0.2)]
        assert 割り当て.speaker_mapping["SPEAKER_01"] == "太郎"
        assert 割り当て.distance_mapping["SPEAKER_01"] == 0.2

    def test_再照合でもUnknownならもとのラベルが維持される(self, 割り当て):
        識別器 = _擬似識別器({2.0: ("Unknown_07", 0.9, {"太郎": 0.9})})
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(2.0)}

        assert clusters.remap_remaining_unknowns(識別器, 割り当て) == []
        assert 割り当て.speaker_mapping["SPEAKER_01"] == "Unknown_01"

    def test_既に実名のクラスタは触られない(self, 割り当て):
        識別器 = _擬似識別器({2.0: ("花子", 0.1, {"花子": 0.1})})
        割り当て.set_speaker("SPEAKER_00", "太郎", 0.05, {"太郎": 0.05})
        割り当て.embeddings = {"SPEAKER_00": _埋め込み(2.0)}

        clusters.remap_remaining_unknowns(識別器, 割り当て)

        assert 割り当て.speaker_mapping["SPEAKER_00"] == "太郎"

    def test_埋め込みが無いクラスタは飛ばされる(self, 割り当て):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)

        clusters.remap_remaining_unknowns(_擬似識別器(), 割り当て)

        assert 割り当て.speaker_mapping["SPEAKER_01"] == "Unknown_01"

    def test_識別器が無ければ何も起きない(self, 割り当て):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)

        assert clusters.remap_remaining_unknowns(None, 割り当て) == []
        assert 割り当て.speaker_mapping["SPEAKER_01"] == "Unknown_01"

    def test_照合が例外を投げてもラベルは維持される(self, 割り当て):
        割り当て.set_speaker("SPEAKER_01", "Unknown_01", None, None)
        割り当て.embeddings = {"SPEAKER_01": _埋め込み(2.0)}

        clusters.remap_remaining_unknowns(_擬似識別器(失敗する=True), 割り当て)

        assert 割り当て.speaker_mapping["SPEAKER_01"] == "Unknown_01"


class Test話者名の検証:
    """sanitize_speaker_name — 対話登録でファイル名に使える名前だけ通す。"""

    def test_前後の空白が落ちる(self):
        assert sanitize_speaker_name("  太郎  ") == "太郎"

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_空の入力はNoneになる(self, raw):
        assert sanitize_speaker_name(raw) is None

    @pytest.mark.parametrize("raw", ["太郎/花子", "a\\b", "a:b", "a*b", "a?b"])
    def test_ファイル名に使えない文字を含むとNoneになる(self, raw):
        assert sanitize_speaker_name(raw) is None


class Test文字起こしCSVパスの生成:
    """create_transcript_csv_path — 音声名＋日時から出力先を決める。"""

    def test_音声のstemとtranscriptionを含む名前になる(self, tmp_path):
        path = create_transcript_csv_path(tmp_path / "会議録音.wav")

        assert path.name.startswith("会議録音-transcription-")
        assert path.suffix == ".csv"

    def test_カレントディレクトリ直下に作られる(self, tmp_path):
        path = create_transcript_csv_path(tmp_path / "sub" / "audio.m4a")

        assert path.parent == Path.cwd()
