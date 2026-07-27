"""話者エンベディング永続キャッシュの特性テスト（現 embedding_cache.py）。

Phase 3 で src/diarization/embedding_cache.py へ移設する。
キャッシュのキーはファイル内容ハッシュであり、内容が同じなら
別パスでもヒットする（＝再計算しない）ことが本機能の要。
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_cache import EmbeddingCache, default_cache_dir


class _埋め込み計算のスパイ:
    """compute_fn の呼び出し回数を数えるテストダブル（実モデルの代役）。"""

    def __init__(self, vector: np.ndarray | None = None):
        self.call_count = 0
        self.vector = (
            vector
            if vector is not None
            else np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        )

    def __call__(self, audio_path):
        self.call_count += 1
        return self.vector


@pytest.fixture
def 音声ファイル(tmp_path):
    path = tmp_path / "太郎.wav"
    path.write_bytes(b"dummy audio content")
    return path


class Testキャッシュのヒットとミス:
    def test_初回は計算され結果が保存される(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()

        result = cache.get_or_compute("model-v1", 音声ファイル, compute)

        assert compute.call_count == 1
        np.testing.assert_array_equal(result, compute.vector)
        assert cache.stats() == {"hits": 0, "misses": 1, "enabled": True}

    def test_2回目は計算されずキャッシュから返る(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()

        cache.get_or_compute("model-v1", 音声ファイル, compute)
        result = cache.get_or_compute("model-v1", 音声ファイル, compute)

        assert compute.call_count == 1, "2回目は compute_fn を呼ばないこと"
        np.testing.assert_array_equal(result, compute.vector)
        assert cache.stats()["hits"] == 1

    def test_内容が同じなら別パスのファイルでもヒットする(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()
        同内容の別ファイル = tmp_path / "コピー.wav"
        同内容の別ファイル.write_bytes(音声ファイル.read_bytes())

        cache.get_or_compute("model-v1", 音声ファイル, compute)
        cache.get_or_compute("model-v1", 同内容の別ファイル, compute)

        assert compute.call_count == 1, "キーはパスではなく内容ハッシュ"

    def test_内容が違えば再計算される(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()
        別内容 = tmp_path / "花子.wav"
        別内容.write_bytes(b"another audio content")

        cache.get_or_compute("model-v1", 音声ファイル, compute)
        cache.get_or_compute("model-v1", 別内容, compute)

        assert compute.call_count == 2

    def test_名前空間が違えば別エントリになる(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()

        cache.get_or_compute("model-v1", 音声ファイル, compute)
        cache.get_or_compute("model-v2", 音声ファイル, compute)

        assert compute.call_count == 2, "モデルが変われば再計算される"

    def test_名前空間の記号はディレクトリ名として安全な文字に置換される(
        self, tmp_path, 音声ファイル
    ):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")

        cache.get_or_compute(
            "pyannote/embedding:v1", 音声ファイル, _埋め込み計算のスパイ()
        )

        作られたディレクトリ = [p.name for p in (tmp_path / "cache").iterdir()]
        assert 作られたディレクトリ == ["pyannote_embedding_v1"]


class Testキャッシュの無効化:
    def test_無効化すると毎回計算される(self, tmp_path, 音声ファイル):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache", enabled=False)
        compute = _埋め込み計算のスパイ()

        cache.get_or_compute("model-v1", 音声ファイル, compute)
        cache.get_or_compute("model-v1", 音声ファイル, compute)

        assert compute.call_count == 2
        assert not (tmp_path / "cache").exists(), "無効時はファイルを作らない"

    def test_環境変数でも無効化できる(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EMBEDDING_CACHE_OFF", "1")

        cache = EmbeddingCache(cache_dir=tmp_path / "cache")

        assert cache.enabled is False


class Test異常時のフォールバック:
    def test_音声ファイルが無くても計算にフォールバックする(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()

        result = cache.get_or_compute("model-v1", tmp_path / "missing.wav", compute)

        assert compute.call_count == 1, "ハッシュ計算に失敗しても処理を止めない"
        np.testing.assert_array_equal(result, compute.vector)

    def test_破損したキャッシュエントリは無視して再計算される(
        self, tmp_path, 音声ファイル
    ):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        compute = _埋め込み計算のスパイ()
        cache.get_or_compute("model-v1", 音声ファイル, compute)
        エントリ = next((tmp_path / "cache" / "model-v1").glob("*.npy"))
        エントリ.write_bytes(b"broken")

        result = cache.get_or_compute("model-v1", 音声ファイル, compute)

        assert compute.call_count == 2
        np.testing.assert_array_equal(result, compute.vector)


class Test保存先の決定:
    def test_環境変数で保存先を指定できる(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "任意の場所"))

        assert default_cache_dir() == (tmp_path / "任意の場所").resolve()

    def test_環境変数が無ければリポジトリ直下が既定になる(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_CACHE_DIR", raising=False)

        assert default_cache_dir().name == "embedding_cache"
