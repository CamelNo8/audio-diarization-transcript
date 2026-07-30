"""AudioProcessor の CSV 出力（src/diarization/processor.py）。

「処理が完走したら CSV が存在する」ことを、発話が 1 件も無い音声でも
保証する。重いモデル（pyannote / Whisper）はモックし、出力の組み立てだけを見る。
"""

from __future__ import annotations

import pytest
from pyannote.core import Segment

from src.common.csv_io import read_rows
from src.diarization import transcript
from src.diarization.processor import AudioProcessor
from src.diarization.transcript import CSV_HEADER
from src.evaluation.srt_stats import compute_overlap_time_ratio, parse_rttm


class _ダミー話者分離:
    """diarization の代わり。build_rows からは参照されない前提。"""

    def labels(self):
        return []

    def itertracks(self, yield_label=False):
        return iter([])


@pytest.fixture
def プロセッサ(tmp_path, monkeypatch):
    出力パス = tmp_path / "result.csv"
    processor = AudioProcessor(
        audio_file=tmp_path / "input.wav",
        output_csv_path=出力パス,
        mlx_model_id="large-v3",
        pyannote_model_id="pyannote/speaker-diarization-3.1",
        hf_token="dummy-token",
    )
    monkeypatch.setattr(AudioProcessor, "prepare_audio", lambda self: None)
    monkeypatch.setattr(
        AudioProcessor, "_run_diarization", lambda self, n: _ダミー話者分離()
    )
    monkeypatch.setattr(AudioProcessor, "_assign_speakers", lambda self, d: None)
    return processor, 出力パス


class Test発話が検出されなかった場合:
    def test_ヘッダのみのCSVが生成される(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(AudioProcessor, "_transcribe", lambda self: [])

        結果 = processor.process_and_save_to_csv()

        assert 結果 is True
        assert 出力パス.is_file(), "成功を返すなら CSV が存在しなければならない"
        assert read_rows(出力パス) == [CSV_HEADER]


class Test発話が検出された場合:
    def test_セグメントが行として書き出される(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(
            AudioProcessor,
            "_transcribe",
            lambda self: [{"start": 0.0, "end": 1.5, "text": "こんにちは"}],
        )

        結果 = processor.process_and_save_to_csv()

        assert 結果 is True
        行 = read_rows(出力パス)
        assert 行[0] == CSV_HEADER
        assert len(行) == 2
        assert 行[1][3] == "こんにちは"

    def test_本文が空のセグメントは行にならない(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(
            AudioProcessor,
            "_transcribe",
            lambda self: [{"start": 0.0, "end": 1.0, "text": "   "}],
        )

        processor.process_and_save_to_csv()

        assert read_rows(出力パス) == [CSV_HEADER]


class Testハルシネーションを含む場合:
    def test_幻聴とみなした行は書き出されない(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(
            AudioProcessor,
            "_transcribe",
            lambda self: [
                {"start": 0.0, "end": 1.0, "text": "こんにちは"},
                {"start": 1.0, "end": 2.0, "text": "ご視聴ありがとうございました"},
                {"start": 2.0, "end": 3.0, "text": "はいはいはい"},
                {"start": 3.0, "end": 4.0, "text": "本題に入ります"},
                {"start": 4.0, "end": 5.0, "text": "本題に入ります"},
            ],
        )

        processor.process_and_save_to_csv()

        本文 = [行[3] for 行 in read_rows(出力パス)[1:]]
        assert 本文 == ["こんにちは", "本題に入ります"]


class Test話者分離に失敗した場合:
    def test_CSVを作らず失敗を返す(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(AudioProcessor, "_run_diarization", lambda self, n: None)

        結果 = processor.process_and_save_to_csv()

        assert 結果 is False
        assert not 出力パス.exists()


class _重なりのある話者分離:
    """3.0〜4.0 秒で2人が重なる話者分離結果の代わり。"""

    def labels(self):
        return ["SPEAKER_00", "SPEAKER_01"]

    def itertracks(self, yield_label=False):
        return iter(
            [
                (Segment(0.0, 4.0), "_", "SPEAKER_00"),
                (Segment(3.0, 5.0), "_", "SPEAKER_01"),
            ]
        )


class Test話者分離結果のRTTM出力:
    """実験の説明変数 overlap_time_ratio のため、重なりを残した結果を保存する。"""

    def test_照合後の話者名でRTTMが書き出される(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(
            AudioProcessor, "_run_diarization", lambda self, n: _重なりのある話者分離()
        )
        monkeypatch.setattr(AudioProcessor, "_transcribe", lambda self: [])
        processor.assignments.speaker_mapping = {
            "SPEAKER_00": "三上",
            "SPEAKER_01": "大賢者",
        }

        processor.process_and_save_to_csv()

        rttm = 出力パス.with_suffix(".rttm")
        assert rttm.is_file()
        segments = parse_rttm(rttm)
        assert [s.speaker for s in segments] == ["三上", "大賢者"]
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].end == pytest.approx(4.0)
        # 3.0〜4.0 の1秒が重なり、発話区間は 0.0〜5.0 の5秒
        assert compute_overlap_time_ratio(segments) == pytest.approx(1.0 / 5.0)

    def test_RTTM書き出しに失敗しても処理は完走する(self, プロセッサ, monkeypatch):
        processor, 出力パス = プロセッサ
        monkeypatch.setattr(AudioProcessor, "_transcribe", lambda self: [])

        def _失敗する(*args, **kwargs):
            raise OSError("書き込みできません")

        monkeypatch.setattr(transcript, "write_rttm", _失敗する)

        結果 = processor.process_and_save_to_csv()

        assert 結果 is True, "RTTM は実験用の副産物であり、字幕生成の成否とは無関係"
        assert 出力パス.is_file()
