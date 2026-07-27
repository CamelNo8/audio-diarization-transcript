"""文字起こしバックエンドのモデルID変換とバックエンド選択の特性テスト。

リファクタリング前の ``transcription_backend.py`` の振る舞いを固定する。
分割後の ``src/transcription/`` に対して同じ期待値を保つ。
Whisper の実推論は行わず、変換・分岐のロジックだけを検証する。
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from src.transcription import backend as tb
from src.transcription import faster_whisper as fw
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

    @pytest.mark.parametrize(
        "model_id", ["distil-large-v3", "distil-large-v2", "DISTIL-LARGE-V3"]
    )
    def test_distil指定はdistilのまま解決される(self, model_id):
        assert model_ids.to_faster_whisper_model(model_id) == model_id.lower()

    def test_サイズ名は他を含むものほど先に並んでいる(self):
        """並び順そのものが仕様。

        判定が部分一致（``size in lowered``）のため、``distil-large-v3`` が
        ``large-v3`` より後ろにあると永久に選ばれない。順序を崩したら
        気付けるようにここで固定する。
        """
        sizes = model_ids.FASTER_WHISPER_SIZES

        for i, size in enumerate(sizes):
            後ろに並ぶ候補 = sizes[i + 1 :]
            自分を含む候補 = [他 for 他 in 後ろに並ぶ候補 if size in 他]
            assert 自分を含む候補 == [], (
                f"{size!r} は {自分を含む候補} より後ろに置く必要がある"
            )


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

    @pytest.mark.parametrize("model_id", ["distil-large-v3", "distil-large-v2"])
    def test_distil指定は警告つきでlarge_v3のmlxリポジトリになる(
        self, model_id, caplog
    ):
        # mlx-community に distil の対応リポジトリが無いため既定へ落とす。
        # 黙って別モデルに差し替えないよう、警告を出すことまで固定する。
        with caplog.at_level("WARNING"):
            repo = model_ids.to_mlx_repo(model_id)

        assert repo == "mlx-community/whisper-large-v3-mlx"
        assert "distil" in caplog.text.lower()


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

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("distil-large-v3", "distil-whisper/distil-large-v3"),
            ("distil-large-v2", "distil-whisper/distil-large-v2"),
        ],
    )
    def test_distilの品質キーがdistilリポジトリになる(self, model_id, expected):
        assert model_ids.to_hf_whisper_repo(model_id) == expected


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


class TestFasterWhisperのモデルキャッシュ:
    """_get_model — 読み込みが重いため、要求した設定でキャッシュが効くこと。

    CTranslate2 が CUDA 非対応ビルドの環境（aarch64 など）では CPU へ
    フォールバックする。このとき要求時のキーにもキャッシュを張らないと、
    毎回 CUDA 初期化を試みては失敗し直すことになる。
    """

    @pytest.fixture(autouse=True)
    def キャッシュを空にする(self, monkeypatch):
        monkeypatch.setattr(fw, "_MODEL_CACHE", {})

    @staticmethod
    def _WhisperModelを差し替える(monkeypatch, *, cuda対応: bool):
        """faster_whisper.WhisperModel を数える偽物に差し替える。"""
        呼び出し = []

        class _偽モデル:
            def __init__(self, model_name, device, compute_type):
                呼び出し.append((model_name, device, compute_type))
                if device == "cuda" and not cuda対応:
                    raise RuntimeError(
                        "This CTranslate2 package was not compiled with CUDA support"
                    )

        偽モジュール = types.ModuleType("faster_whisper")
        偽モジュール.WhisperModel = _偽モデル
        monkeypatch.setitem(sys.modules, "faster_whisper", 偽モジュール)
        return 呼び出し

    def test_同じ設定の二回目は読み込まれない(self, monkeypatch):
        呼び出し = self._WhisperModelを差し替える(monkeypatch, cuda対応=True)

        最初 = fw._get_model("large-v3", "cuda")
        二回目 = fw._get_model("large-v3", "cuda")

        assert 最初 is 二回目
        assert len(呼び出し) == 1

    def test_CPUへ落ちたあとも二回目は読み込まれない(self, monkeypatch):
        呼び出し = self._WhisperModelを差し替える(monkeypatch, cuda対応=False)

        最初 = fw._get_model("large-v3", "cuda")
        呼び出し.clear()
        二回目 = fw._get_model("large-v3", "cuda")

        assert 最初 is 二回目
        assert 呼び出し == [], "CUDA 初期化を再試行してはならない"

    def test_CPUへ落ちた場合はCPU設定でも同じモデルが返る(self, monkeypatch):
        self._WhisperModelを差し替える(monkeypatch, cuda対応=False)

        cuda要求 = fw._get_model("large-v3", "cuda")
        cpu要求 = fw._get_model("large-v3", "cpu")

        assert cuda要求 is cpu要求
