"""Step 1（文字起こし）ルートの特性テスト。

重いモデルはモックし、入力検証と生成物だけを見る（規約2.5）。
Step 1 は非同期ジョブなので、POST は進捗パネルを返し、実際の結果は
``/process/transcription/<job_id>/status`` から取る。
"""

from __future__ import annotations

import json
import shutil

import pytest

import src.web.jobs as jobs

WAV_BYTES = b"RIFF----WAVEfmt "


def _唯一のジョブID(clusters_root) -> str:
    job_files = list(clusters_root.glob("*/job.json"))
    assert len(job_files) == 1, f"ジョブが1件ではない: {job_files}"
    return job_files[0].parent.name


class Test入力の検証:
    """検証は同期で行う。不正な入力ではジョブを作らずエラーを返す。"""

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

    def test_入力エラーではジョブが作られない(
        self, client, 文字起こしをモックにする, ジョブ保存先, monkeypatch
    ):
        monkeypatch.setattr(shutil, "which", lambda name: None)

        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert list(ジョブ保存先.glob("*/job.json")) == []


class Test非同期ジョブの受付:
    """POST は待たずに返り、進捗はポーリングで取る。"""

    def test_受付時に進捗パネルが返る(
        self,
        client,
        文字起こしをモックにする,
        ワーカーを起動しない,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        assert response.status_code == 200
        assert "文字起こし中" in response.text

    def test_進捗パネルはポーリングを指示する(
        self,
        client,
        文字起こしをモックにする,
        ワーカーを起動しない,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        response = client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        job_id = _唯一のジョブID(ジョブ保存先)
        assert f"/process/transcription/{job_id}/status" in response.text
        assert "hx-trigger" in response.text

    def test_受付直後のジョブは実行中になっている(
        self,
        client,
        文字起こしをモックにする,
        ワーカーを起動しない,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        job = jobs.load_job(_唯一のジョブID(ジョブ保存先))
        assert job["status"] == jobs.STATUS_RUNNING
        assert job["clusters"] == [], "完了前に開かれても壊れないよう空で入れておく"


class Test進捗の問い合わせ:
    def test_実行中は進捗パネルが返る(
        self,
        client,
        文字起こしをモックにする,
        ワーカーを起動しない,
        作業ディレクトリ,
        ジョブ保存先,
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )
        job_id = _唯一のジョブID(ジョブ保存先)

        response = client.get(f"/process/transcription/{job_id}/status")

        assert response.status_code == 200
        assert "文字起こし中" in response.text
        assert "hx-trigger" in response.text, "実行中はポーリングが続くこと"

    def test_完了すると成功パネルが返りポーリングが止まる(
        self, client, 文字起こしをモックにする, 作業ディレクトリ, ジョブ保存先
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
            data={"output_srt_name": "結果.srt"},
        )
        job_id = _唯一のジョブID(ジョブ保存先)

        response = client.get(f"/process/transcription/{job_id}/status")

        assert response.status_code == 200
        assert "文字起こし完了" in response.text
        assert "hx-trigger" not in response.text, "完了断片にポーリングを残さないこと"
        assert "/download/結果.srt" in response.text

    def test_失敗するとエラーパネルが返りポーリングが止まる(
        self, client, 文字起こしをモックにする, 作業ディレクトリ, ジョブ保存先
    ):
        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )
        job_id = _唯一のジョブID(ジョブ保存先)
        jobs.update_job(job_id, status=jobs.STATUS_ERROR, error="ボーカル抽出に失敗")

        response = client.get(f"/process/transcription/{job_id}/status")

        assert response.status_code == 200
        assert "ボーカル抽出に失敗" in response.text
        assert "hx-trigger" not in response.text

    def test_存在しないジョブはエラー表示になる(self, client, ジョブ保存先):
        response = client.get("/process/transcription/20260728-000000-zzzzzz/status")

        assert response.status_code == 200
        assert "見つかりません" in response.text

    def test_不正なジョブIDでもエラー表示になる(self, client, ジョブ保存先):
        # パス区切りは経路解決で弾かれるため、ここで見るのは書式違反のほう
        response = client.get("/process/transcription/not.a.job/status")

        assert response.status_code == 200
        assert "見つかりません" in response.text


class Test文字起こし:
    """処理本体。ワーカーは同期実行に差し替えてある。"""

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
        assert job["status"] == jobs.STATUS_DONE
        assert job["srt_filename"] == "結果.srt"
        assert len(job["clusters"]) == 1

    def test_処理が例外で落ちてもジョブにエラーが残る(
        self,
        client,
        文字起こしをモックにする,
        作業ディレクトリ,
        ジョブ保存先,
        monkeypatch,
    ):
        # ワーカースレッドで起きた例外は応答に乗らないため、ジョブに残す必要がある
        def 落ちる(*args, **kwargs):
            raise RuntimeError("GPU が見つかりません")

        monkeypatch.setattr("src.web.routes.transcription._run_transcription", 落ちる)

        client.post(
            "/process/transcription",
            files={"audio_file": ("a.wav", WAV_BYTES, "audio/wav")},
        )

        job = jobs.load_job(_唯一のジョブID(ジョブ保存先))
        assert job["status"] == jobs.STATUS_ERROR
        assert "GPU が見つかりません" in job["error"]

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
