"""複数音声の連結（ffmpeg）の特性テスト。

コマンドの組み立て（純粋なロジック）と、実行部分の呼び出し方を分けて固定する。
実際に ffmpeg を動かす統合テストは最後に1本だけ置く。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.common.audio import build_concat_command, concat_audio, probe_duration_sec
from src.config import TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ


class Test連結コマンドの組み立て:
    def test_入力ファイルが全て_i_で並ぶ(self, tmp_path):
        clips = [tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"]

        cmd = build_concat_command(clips, tmp_path / "out.wav")

        assert [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"] == [
            str(c) for c in clips
        ]

    def test_各入力のサンプリングレートとチャンネル数が揃えられる(self, tmp_path):
        clips = [tmp_path / "a.wav", tmp_path / "b.wav"]

        cmd = build_concat_command(clips, tmp_path / "out.wav")
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        expected = (
            f"aformat=sample_rates={TARGET_SAMPLE_RATE_HZ}"
            f":channel_layouts={'mono' if TARGET_CHANNELS == 1 else 'stereo'}"
        )
        assert filter_arg.count(expected) == len(clips)

    def test_連結する本数がフィルタに入る(self, tmp_path):
        clips = [tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"]

        cmd = build_concat_command(clips, tmp_path / "out.wav")
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        assert "concat=n=3:v=0:a=1" in filter_arg

    def test_出力先が最後に来る(self, tmp_path):
        dst = tmp_path / "out.wav"

        cmd = build_concat_command([tmp_path / "a.wav"], dst)

        assert cmd[-1] == str(dst)

    def test_1本だけでも組み立てられる(self, tmp_path):
        cmd = build_concat_command([tmp_path / "a.wav"], tmp_path / "out.wav")
        filter_arg = cmd[cmd.index("-filter_complex") + 1]

        assert "concat=n=1:v=0:a=1" in filter_arg

    def test_0本ならエラーになる(self, tmp_path):
        with pytest.raises(ValueError, match="1本もありません"):
            build_concat_command([], tmp_path / "out.wav")


class Test連結の実行:
    def test_組み立てたコマンドで_ffmpeg_が呼ばれる(self, tmp_path, monkeypatch):
        clips = [tmp_path / "a.wav", tmp_path / "b.wav"]
        for c in clips:
            c.write_bytes(b"dummy")
        dst = tmp_path / "out.wav"
        called = {}

        def fake_run(cmd, **kwargs):
            called["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)

        concat_audio(clips, dst)

        assert called["cmd"] == build_concat_command(clips, dst)

    def test_存在しないファイルがあれば分かるエラーになる(self, tmp_path):
        missing = tmp_path / "missing.wav"

        with pytest.raises(FileNotFoundError, match="missing.wav"):
            concat_audio([missing], tmp_path / "out.wav")

    def test_ffmpegが失敗したら標準エラーを添えて失敗する(self, tmp_path, monkeypatch):
        clip = tmp_path / "a.wav"
        clip.write_bytes(b"dummy")

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="壊れた入力です")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="壊れた入力です"):
            concat_audio([clip], tmp_path / "out.wav")


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg / ffprobe が無い環境ではスキップする",
)
class Test実際のffmpegで連結する:
    def _make_tone(self, path: Path, seconds: float, sample_rate: int) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={seconds}:sample_rate={sample_rate}",
                "-y",
                str(path),
            ],
            check=True,
        )

    def test_サンプリングレートが違う2本を繋ぐと長さが合計になる(self, tmp_path):
        first = tmp_path / "first.wav"
        second = tmp_path / "second.wav"
        self._make_tone(first, 1.0, 44100)
        self._make_tone(second, 2.0, 16000)
        dst = tmp_path / "merged.wav"

        concat_audio([first, second], dst)

        assert probe_duration_sec(dst) == pytest.approx(3.0, abs=0.05)
