"""話者照合（埋め込み正規化・コサイン距離による判定）の特性テスト。

リファクタリング前の ``speaker_identification.py`` の振る舞いを固定する。
分割後の ``src/diarization/speaker_identifier.py`` に対して同じ期待値を保つ。
埋め込みモデル（pyannote）は読み込まず、モックに差し替える。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.diarization.speaker_identifier import SpeakerIdentifier


class _擬似キャッシュ:
    """EmbeddingCache の代わり。計算関数をそのまま呼び、呼び出しを記録する。"""

    def __init__(self):
        self.calls: list[tuple[str, Path]] = []

    def get_or_compute(self, namespace, audio_path, compute):
        self.calls.append((namespace, Path(audio_path)))
        return compute(Path(audio_path))


@pytest.fixture
def 識別器(monkeypatch):
    """モデルを読み込まない SpeakerIdentifier を作る。"""
    monkeypatch.setattr(
        SpeakerIdentifier, "_load_model", lambda self, model_name, hf_token: None
    )
    return SpeakerIdentifier(
        model_name="pyannote/embedding", hf_token="dummy", threshold=0.5
    )


def _単位ベクトル(*values: float) -> np.ndarray:
    """(1, N) 形の L2 正規化済みベクトルを作る。"""
    v = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Test埋め込みの正規化:
    """_normalize_embedding — L2 ノルムで割り、(1, N) 形に揃える。"""

    def test_ノルムが1になる(self, 識別器):
        normalized = 識別器._normalize_embedding(np.array([3.0, 4.0]))

        assert np.linalg.norm(normalized) == pytest.approx(1.0)
        assert normalized[0].tolist() == pytest.approx([0.6, 0.8])

    def test_1次元入力が2次元に整形される(self, 識別器):
        assert 識別器._normalize_embedding(np.array([1.0, 0.0, 0.0])).shape == (1, 3)

    def test_torchテンソルも受け付ける(self, 識別器):
        normalized = 識別器._normalize_embedding(torch.tensor([3.0, 4.0]))

        assert isinstance(normalized, np.ndarray)
        assert normalized[0].tolist() == pytest.approx([0.6, 0.8])

    def test_float32に変換される(self, 識別器):
        assert 識別器._normalize_embedding(np.array([1.0, 1.0])).dtype == np.float32

    def test_ノルムが0のベクトルはValueErrorになる(self, 識別器):
        with pytest.raises(ValueError, match="ノルムが 0"):
            識別器._normalize_embedding(np.zeros(4))


class TestUnknown名の採番:
    """_next_unknown_name — 未登録話者に 01 から連番を振る。"""

    def test_呼ぶたびに連番が進む(self, 識別器):
        名前 = [識別器._next_unknown_name() for _ in range(3)]

        assert 名前 == ["Unknown_01", "Unknown_02", "Unknown_03"]

    def test_10件目以降も2桁で0埋めされる(self, 識別器):
        識別器.unknown_counter = 10

        assert 識別器._next_unknown_name() == "Unknown_10"


class Test話者の照合:
    """identify_speaker_with_distances — 最近傍と閾値で判定する。"""

    def test_登録話者がいなければUnknownと距離なしを返す(self, 識別器):
        name, distance, candidates = 識別器.identify_speaker_with_distances(
            _単位ベクトル(1.0, 0.0)
        )

        assert (name, distance, candidates) == ("Unknown_01", None, None)

    def test_最も距離が近い登録話者が選ばれる(self, 識別器):
        識別器.registry_embeddings = {
            "太郎": _単位ベクトル(1.0, 0.0),
            "花子": _単位ベクトル(0.0, 1.0),
        }

        name, distance, candidates = 識別器.identify_speaker_with_distances(
            _単位ベクトル(1.0, 0.0)
        )

        assert name == "太郎"
        assert distance == pytest.approx(0.0, abs=1e-6)
        assert set(candidates) == {"太郎", "花子"}
        assert candidates["花子"] == pytest.approx(1.0, abs=1e-6)

    def test_距離が閾値ちょうどなら登録話者と判定される(self, 識別器):
        # 直交ベクトル同士のコサイン距離は 1.0
        識別器.threshold = 1.0
        識別器.registry_embeddings = {"太郎": _単位ベクトル(1.0, 0.0)}

        name, distance, _ = 識別器.identify_speaker_with_distances(
            _単位ベクトル(0.0, 1.0)
        )

        assert name == "太郎"
        assert distance == pytest.approx(1.0, abs=1e-6)

    def test_距離が閾値を超えるとUnknownになるが距離は返る(self, 識別器):
        識別器.threshold = 0.99
        識別器.registry_embeddings = {"太郎": _単位ベクトル(1.0, 0.0)}

        name, distance, candidates = 識別器.identify_speaker_with_distances(
            _単位ベクトル(0.0, 1.0)
        )

        assert name == "Unknown_01"
        assert distance == pytest.approx(1.0, abs=1e-6)
        assert candidates == {"太郎": pytest.approx(1.0, abs=1e-6)}

    def test_identify_speakerは名前と距離だけを返す(self, 識別器):
        識別器.registry_embeddings = {"太郎": _単位ベクトル(1.0, 0.0)}

        結果 = 識別器.identify_speaker(_単位ベクトル(1.0, 0.0))

        assert len(結果) == 2
        assert 結果[0] == "太郎"


class Test話者の登録:
    """register_speaker — キャッシュ経由で埋め込みを取得して登録する。"""

    def test_キャッシュ経由で埋め込みが登録される(self, 識別器, tmp_path):
        音声 = tmp_path / "太郎.wav"
        音声.write_bytes(b"dummy")
        キャッシュ = _擬似キャッシュ()
        識別器._cache = キャッシュ
        識別器._compute_registration_embedding = lambda path: _単位ベクトル(1.0, 0.0)

        識別器.register_speaker("太郎", 音声)

        assert "太郎" in 識別器.registry_embeddings
        assert キャッシュ.calls[0][1] == 音声

    def test_存在しないファイルはFileNotFoundErrorになる(self, 識別器, tmp_path):
        with pytest.raises(FileNotFoundError):
            識別器.register_speaker("太郎", tmp_path / "無い.wav")


class Testキャッシュ名前空間:
    """_cache_namespace — 前処理を変えたら旧キャッシュと分離されるようにする。"""

    def test_モデル名とトリミング量とスキーマ版を含む(self, 識別器):
        namespace = 識別器._cache_namespace

        assert namespace.startswith("pyannote/embedding__trim")
        assert namespace.endswith("__v1")

    def test_モデルが違えば名前空間も変わる(self, 識別器):
        別モデル = 識別器._cache_namespace
        識別器.model_name = "pyannote/wespeaker"

        assert 識別器._cache_namespace != 別モデル
