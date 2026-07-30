"""正解字幕からの5分クリップ選定のテスト。

予稿 4.3 の「1クリップを重なり・話者交替が多い区間、もう1つを落ち着いた区間として
意図的に選定する。クリップの切り出しは発話境界で行う」を自動化する部分。
"""

from __future__ import annotations

import pytest

from src.evaluation.clip_selector import (
    SelectionOptions,
    enumerate_windows,
    select_clip_pair,
    shift_entries,
    write_srt,
)
from src.evaluation.srt_stats import parse_srt


def _SRTを書く(path, rows: list[tuple[float, float, str]]):
    blocks = []
    for number, (start, end, text) in enumerate(rows, start=1):
        blocks.append(f"{number}\n{_時刻(start)} --> {_時刻(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _時刻(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3600 * 1000)
    minutes, rest = divmod(rest, 60 * 1000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _落ち着いた区間と騒がしい区間(tmp_path, entry_sec: float = 10.0):
    """前半300秒は1人が淡々と、後半300秒は2人が交替しながら重なる字幕を作る。"""
    rows = []
    for i in range(30):
        start = i * entry_sec
        rows.append((start, start + entry_sec, "（三上）そうだね"))
    for i in range(30, 60):
        start = i * entry_sec
        speaker = "三上" if i % 2 == 0 else "大賢者"
        other = "大賢者" if i % 2 == 0 else "三上"
        rows.append((start, start + entry_sec, f"（{speaker}）あのさ （{other}）うん"))
    return _SRTを書く(tmp_path / "gt.srt", rows)


class Test候補窓の列挙:
    """enumerate_windows — 窓の端は必ず発話境界に合わせる。"""

    def test_窓の終わりは発話境界になる(self, tmp_path):
        # 7秒の発話が並ぶので、60秒以上になる最初の境界は63秒（9件分）
        path = _SRTを書く(
            tmp_path / "gt.srt",
            [(i * 7.0, i * 7.0 + 7.0, "（三上）はい") for i in range(20)],
        )
        entries = parse_srt(path)

        windows = enumerate_windows(
            entries, SelectionOptions(length_sec=60.0, tolerance_sec=15.0)
        )

        assert windows[0].start == pytest.approx(0.0)
        assert windows[0].end == pytest.approx(63.0)
        boundaries = {entry.end for entry in entries}
        assert all(window.end in boundaries for window in windows)

    def test_窓長が許容範囲を外れる候補は作られない(self, tmp_path):
        # 100秒の発話が1件だけ続く並びでは、60秒の窓を作れない
        path = _SRTを書く(
            tmp_path / "gt.srt",
            [(i * 100.0, i * 100.0 + 100.0, "（三上）はい") for i in range(5)],
        )
        entries = parse_srt(path)

        windows = enumerate_windows(
            entries, SelectionOptions(length_sec=60.0, tolerance_sec=15.0)
        )

        assert windows == []

    def test_窓には区間内の字幕だけが含まれる(self, tmp_path):
        entries = parse_srt(_落ち着いた区間と騒がしい区間(tmp_path))

        windows = enumerate_windows(entries, SelectionOptions())

        first = windows[0]
        assert first.variables.utterance_count == 30
        assert all(first.start <= e.start and e.end <= first.end for e in first.entries)


class Test2本のクリップの選定:
    """select_clip_pair — 難しい方と落ち着いた方を、重ならないように選ぶ。"""

    def test_重なりと話者交替が多い区間がhardに選ばれる(self, tmp_path):
        entries = parse_srt(_落ち着いた区間と騒がしい区間(tmp_path))
        windows = enumerate_windows(entries, SelectionOptions())

        hard, calm = select_clip_pair(windows, SelectionOptions())

        assert hard.start == pytest.approx(300.0)
        assert hard.variables.overlap_entry_ratio == pytest.approx(1.0)
        assert calm.start == pytest.approx(0.0)
        assert calm.variables.overlap_entry_ratio == 0.0
        assert calm.variables.speaker_change_count == 0

    def test_選ばれた2本は時間が重ならない(self, tmp_path):
        entries = parse_srt(_落ち着いた区間と騒がしい区間(tmp_path))
        windows = enumerate_windows(entries, SelectionOptions())

        hard, calm = select_clip_pair(windows, SelectionOptions())

        assert hard.end <= calm.start or calm.end <= hard.start

    def test_目標話者数から離れた窓はhardに選ばれにくい(self, tmp_path):
        """4人の区間と2人の区間があるとき、目標2人ならペナルティで後者が選ばれる。"""
        rows = []
        for i in range(30):  # 0-300秒: 4人が交替（本来いちばん難しい）
            start = i * 10.0
            speaker = ["A", "B", "C", "D"][i % 4]
            rows.append((start, start + 10.0, f"（{speaker}）はい"))
        for i in range(30, 60):  # 300-600秒: 2人が交替
            start = i * 10.0
            speaker = "A" if i % 2 == 0 else "B"
            rows.append((start, start + 10.0, f"（{speaker}）はい"))
        entries = parse_srt(_SRTを書く(tmp_path / "gt.srt", rows))
        windows = enumerate_windows(entries, SelectionOptions())

        penalized = SelectionOptions(target_speakers=2, speaker_penalty=10.0)
        hard, _ = select_clip_pair(windows, penalized)

        assert hard.variables.speaker_count == 2

    def test_重ならない2本が取れなければエラーになる(self, tmp_path):
        # 全体が400秒しかないので、300秒の窓は2本取れない
        path = _SRTを書く(
            tmp_path / "gt.srt",
            [(i * 10.0, i * 10.0 + 10.0, "（三上）はい") for i in range(40)],
        )
        entries = parse_srt(path)
        windows = enumerate_windows(entries, SelectionOptions())

        with pytest.raises(ValueError, match="重ならない"):
            select_clip_pair(windows, SelectionOptions())

    def test_候補が無ければエラーになる(self):
        with pytest.raises(ValueError, match="候補"):
            select_clip_pair([], SelectionOptions())


class Test切り出した字幕の書き出し:
    """shift_entries / write_srt — クリップ単体で評価に使える正解SRTにする。"""

    def test_切り出した字幕は先頭が0秒に振り直される(self, tmp_path):
        entries = parse_srt(_落ち着いた区間と騒がしい区間(tmp_path))
        windows = enumerate_windows(entries, SelectionOptions())
        hard, _ = select_clip_pair(windows, SelectionOptions())

        shifted = shift_entries(hard.entries, offset=hard.start)

        assert shifted[0].start == pytest.approx(0.0)
        assert shifted[0].end == pytest.approx(10.0)
        assert shifted[-1].end == pytest.approx(300.0)

    def test_書き出したSRTを読み直すと同じ時刻と話者になる(self, tmp_path):
        entries = parse_srt(_落ち着いた区間と騒がしい区間(tmp_path))
        windows = enumerate_windows(entries, SelectionOptions())
        hard, _ = select_clip_pair(windows, SelectionOptions())
        shifted = shift_entries(hard.entries, offset=hard.start)

        output = tmp_path / "hard.srt"
        write_srt(shifted, output)
        reloaded = parse_srt(output)

        assert len(reloaded) == len(shifted)
        assert reloaded[0].start == pytest.approx(0.0)
        assert reloaded[0].speakers == shifted[0].speakers
        assert reloaded[-1].end == pytest.approx(300.0)
