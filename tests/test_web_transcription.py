"""Step 1（文字起こし）ルートの特性テスト。

重いモデルはモックし、入力検証と生成物だけを見る（規約2.5）。
"""

from __future__ import annotations

import json
import shutil

import pytest

WAV_BYTES = b"RIFF----WAVEfmt "


class Test文字起こし:
    """Step 1。重いモデルはモックし、入力検証と生成物だけを見る。"""

    def test_ffmpegが無ければエラー表示になる(
        self, client, 文字起こしをモックにする, monkeypatch
    ):
        monkeypatch.setattr(shutil, "which", lambda name: None)

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert response.status_code == 200
        assert "ffmpeg" in response.text

    def test_HFトークンが無ければエラー表示になる(
        self, client, 文字起こしをモックにする, monkeypatch
    ):
        monkeypatch.delenv("HF_TOKEN", raising=False)

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert response.status_code == 200
        assert "Hugging Face Token" in response.text

    def test_成功するとSRTとジョブが作られる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        work_dir = 作業ディレクトリ
        clusters_root = ジョブ保存先

        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"output_srt_name": "結果.srt"},
        )

        assert response.status_code == 200
        assert (work_dir / "結果.srt").is_file()
        assert (work_dir / "結果.csv").is_file()
        job_files = list(clusters_root.glob("*/job.json"))
        assert len(job_files) == 1
        job = json.loads(job_files[0].read_text(encoding="utf-8"))
        assert job["srt_filename"] == "結果.srt"
        assert len(job["clusters"]) == 1

    def test_話者数の指定がプロセッサへ渡る(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"num_speakers": "3"},
        )

        assert 文字起こしをモックにする.last.known_num_speakers == 3

    def test_数値でない話者数は指定なしとして扱われる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"num_speakers": "たくさん"},
        )

        assert 文字起こしをモックにする.last.known_num_speakers is None

    @pytest.mark.parametrize(
        ("quality", "custom", "expected"),
        [
            ("medium", "", "medium"),
            ("__custom__", "my/whisper", "my/whisper"),
            ("__custom__", "", "mlx-community/whisper-large-v3-mlx"),
            ("", "", "mlx-community/whisper-large-v3-mlx"),
        ],
    )
    def test_Whisperモデルは品質指定とカスタム入力から決まる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
        quality,
        custom,
        expected,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"whisper_quality": quality, "whisper_custom_model": custom},
        )

        assert 文字起こしをモックにする.last.kwargs["mlx_model_id"] == expected

    def test_保存先DB未選択で声紋をアップロードするとエラーになる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        response = client.post(
            "/process/transcription",
            files=[
                ("audio_file", ("a.wav", WAV_BYTES, "audio/wav")),
                ("registry_files", ("太郎.wav", WAV_BYTES, "audio/wav")),
            ],
            data={"db_choice": "none"},
        )

        assert response.status_code == 200
        assert "保存先DB" in response.text

    def test_新規DBを指定すると声紋が登録される(
        self,
        client,
        文字起こしをモックにする,
        話者照合をモックにする,
        作業ディレクトリ,
        ジョブ保存先,
        声紋DBルート,
    ):
        db_root = 声紋DBルート

        response = client.post(
            "/process/transcription",
            files=[
                ("audio_file", ("a.wav", WAV_BYTES, "audio/wav")),
                ("registry_files", ("太郎.wav", WAV_BYTES, "audio/wav")),
            ],
            data={"db_choice": "new", "db_new_name": "新DB"},
        )

        assert response.status_code == 200
        assert (db_root / "新DB" / "太郎.wav").is_file()
        assert 話者照合をモックにする.registered == [
            ("太郎", db_root / "新DB" / "太郎.wav")
        ]

    def test_存在しない既存DBを選ぶとエラーになる(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        声紋DBルート,
    ):
        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"db_choice": "existing", "db_existing_name": "無いDB"},
        )

        assert response.status_code == 200
        assert "DBエラー" in response.text
