"""窓口2（src/speaker_verification.py）の入口としての振る舞い。

重いモデル（pyannote の埋め込み）は読まない。話者分離結果の受け取りと、
照合器が無いときの扱いだけを検証する。
"""

from __future__ import annotations

from pyannote.core import Segment

from src import speaker_verification


class _擬似話者分離:
    """pyannote の話者分離結果の代わり。クラスタIDごとの区間だけを持つ。"""

    def __init__(self, 区間表: dict[str, list[Segment]]):
        self.区間表 = 区間表

    def labels(self) -> list[str]:
        return list(self.区間表)

    def label_timeline(self, cluster_id: str) -> list[Segment]:
        return self.区間表[cluster_id]


class Test区間の照合:
    """verify_segments — 話者分離の各クラスタに話者名を付ける。"""

    def test_照合器が無いときは全クラスタがUnknownになる(self):
        分離結果 = _擬似話者分離(
            {
                "SPEAKER_00": [Segment(0.0, 2.0)],
                "SPEAKER_01": [Segment(2.0, 5.0)],
            }
        )

        割り当て = speaker_verification.verify_segments(分離結果, None, None)

        assert 割り当て.speaker_mapping == {
            "SPEAKER_00": "Unknown_01",
            "SPEAKER_01": "Unknown_02",
        }

    def test_代表区間は最長の区間が選ばれる(self):
        分離結果 = _擬似話者分離({"SPEAKER_00": [Segment(0.0, 1.5), Segment(3.0, 8.0)]})

        割り当て = speaker_verification.verify_segments(分離結果, None, None)

        assert 割り当て.segments["SPEAKER_00"] == Segment(3.0, 8.0)
