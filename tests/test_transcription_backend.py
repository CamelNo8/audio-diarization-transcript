"""文字起こしバックエンドのモデルID変換とバックエンド選択の特性テスト。

リファクタリング前の ``transcription_backend.py`` の振る舞いを固定する。
分割後の ``src/transcription/`` に対して同じ期待値を保つ。
Whisper の実推論は行わず、変換・分岐のロジックだけを検証する。
"""

from __future__ import annotations

import pytest
import torch

from src.transcription import backend as tb
from src.transcription import model_ids


class Test実行環境の判定:
    """is_apple_silicon — mlx-whisper が使えるかの判定に使う。"""

    @pytest.mark.parametrize("machine", ["arm64", "aarch64"])
    def test_macOSのARMならTrueになる(self, monkeypatch, machine):
        monkeypatch.setattr(tb.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(tb.platform, "machine", lambda: machine)

        assert tb.is_apple_silicon() is True

    @pytest.mark.parametrize(
        ("system", "machine"),
        [("Darwin", "x86_64"), ("Linux", "aarch64"), ("Windows", "AMD64")],
    )
    def test_それ以外の環境ではFalseになる(self, monkeypatch, system, machine):
        monkeypatch.setattr(tb.platform, "system", lambda: system)
        monkeypatch.setattr(tb.platform, "machine", lambda: machine)

        assert tb.is_apple_silicon() is False


class TestFasterWhisperのモデル名変換:
    """_to_faster_whisper_model — 任意の指定をサイズ名へ寄せる。"""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("large-v3", "large-v3"),
            ("mlx-community/whisper-large-v3-mlx", "large-v3"),
            ("mlx-community/whisper-large-v3-turbo", "large-v3-turbo"),
            ("openai/whisper-medium", "medium"),
            ("tiny", "tiny"),
            ("LARGE-V2", "large-v2"),
        ],
    )
    def test_モデル指定からサイズ名が取り出される(self, model_id, expected):
        assert model_ids.to_faster_whisper_model(model_id) == expected

    def test_判別できない指定はlarge_v3になる(self):
        assert model_ids.to_faster_whisper_model("unknown-model") == "large-v3"

    def test_distil指定はlarge_v3として解決される既存挙動(self):
        # FASTER_WHISPER_SIZES で "large-v3" が "distil-large-v3" より前に
        # あるため先に一致する。
        # 既存の振る舞いなのでそのまま固定する（別途報告済み）。
        assert model_ids.to_faster_whisper_model("distil-large-v3") == "large-v3"


class TestMLXリポジトリ変換:
    """_to_mlx_repo — 品質キーを mlx-community のリポジトリへ変換する。"""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("large-v3", "mlx-community/whisper-large-v3-mlx"),
            ("turbo", "mlx-community/whisper-large-v3-turbo"),
            ("medium", "mlx-community/whisper-medium-mlx"),
            ("tiny", "mlx-community/whisper-tiny-mlx"),
        ],
    )
    def test_品質キーがmlxリポジトリになる(self, model_id, expected):
        assert model_ids.to_mlx_repo(model_id) == expected

    def test_既にmlxリポジトリならそのまま返る(self):
        repo = "mlx-community/whisper-large-v2-mlx"

        assert model_ids.to_mlx_repo(repo) == repo

    def test_判別できない指定はlarge_v3のmlxリポジトリになる(self):
        assert model_ids.to_mlx_repo("unknown") == "mlx-community/whisper-large-v3-mlx"


class TestHuggingFaceリポジトリ変換:
    """_to_hf_whisper_repo — transformers バックエンド用のリポジトリを決める。"""

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("large-v3", "openai/whisper-large-v3"),
            ("turbo", "openai/whisper-large-v3-turbo"),
            ("small", "openai/whisper-small"),
        ],
    )
    def test_品質キーがopenaiリポジトリになる(self, model_id, expected):
        assert model_ids.to_hf_whisper_repo(model_id) == expected

    @pytest.mark.parametrize(
        "repo", ["openai/whisper-medium", "distil-whisper/distil-large-v3"]
    )
    def test_フルリポジトリ指定はそのまま返る(self, repo):
        assert model_ids.to_hf_whisper_repo(repo) == repo

    def test_判別できない指定はlarge_v3になる(self):
        assert model_ids.to_hf_whisper_repo("unknown") == "openai/whisper-large-v3"


class Testバックエンドの選択:
    """_resolve_backend — 明示指定を尊重し、auto なら環境から決める。"""

    @pytest.mark.parametrize("backend", ["mlx", "faster", "transformers"])
    def test_明示指定はそのまま採用される(self, backend):
        assert tb.resolve_backend(backend) == backend

    def test_大文字の指定も受け付ける(self):
        assert tb.resolve_backend("MLX") == "mlx"

    @pytest.mark.parametrize("backend", ["auto", None, ""])
    def test_autoやNoneはApple_Siliconならmlxになる(self, monkeypatch, backend):
        monkeypatch.setattr(tb, "is_apple_silicon", lambda: True)

        assert tb.resolve_backend(backend) == "mlx"

    @pytest.mark.parametrize("backend", ["auto", None, ""])
    def test_autoやNoneはApple_Silicon以外ならfasterになる(self, monkeypatch, backend):
        monkeypatch.setattr(tb, "is_apple_silicon", lambda: False)

        assert tb.resolve_backend(backend) == "faster"

    def test_未知のバックエンド名は環境判定にフォールバックする(self, monkeypatch):
        monkeypatch.setattr(tb, "is_apple_silicon", lambda: False)

        assert tb.resolve_backend("openai") == "faster"


class Testデバイスの選択:
    """select_whisper_device — CTranslate2 は cuda / cpu のみ対応。"""

    def test_cudaを希望されたらcudaになる(self):
        assert tb.select_whisper_device(torch.device("cuda")) == "cuda"

    def test_mpsを希望されてもCUDAが無ければcpuになる(self, monkeypatch):
        monkeypatch.setattr(tb.torch.cuda, "is_available", lambda: False)

        assert tb.select_whisper_device(torch.device("mps")) == "cpu"

    def test_希望なしでCUDAが使えるならcudaになる(self, monkeypatch):
        monkeypatch.setattr(tb.torch.cuda, "is_available", lambda: True)

        assert tb.select_whisper_device(None) == "cuda"

    def test_希望なしでCUDAが使えなければcpuになる(self, monkeypatch):
        monkeypatch.setattr(tb.torch.cuda, "is_available", lambda: False)

        assert tb.select_whisper_device(None) == "cpu"
