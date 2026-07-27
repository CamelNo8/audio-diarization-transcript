"""話者識別器の生成（src/diarization/registry.py）。

モデルがローカルに無い初回だけオフライン指定を一時的に外す。
「一時的に」である以上、元の値へ戻さなければならない。
重い埋め込みモデルは読み込まず、生成の成否だけをモックで再現する。
"""

from __future__ import annotations

import os

import pytest

from src.diarization import registry


@pytest.fixture(autouse=True)
def 識別器のキャッシュを空にする(monkeypatch):
    monkeypatch.setattr(registry, "_SPEAKER_IDENTIFIER_CACHE", {})


def _識別器を差し替える(monkeypatch, *, 初回失敗: bool, 二回目も失敗: bool = False):
    """SpeakerIdentifier を、呼ばれた時点の環境変数を記録する偽物にする。"""
    観測されたオフライン指定 = []

    def _偽識別器(model_name, hf_token, threshold):
        観測されたオフライン指定.append(os.environ.get("HF_HUB_OFFLINE"))
        if len(観測されたオフライン指定) == 1 and 初回失敗:
            raise RuntimeError("model not found locally")
        if len(観測されたオフライン指定) == 2 and 二回目も失敗:
            raise RuntimeError("network unreachable")
        return f"識別器({model_name})"

    monkeypatch.setattr(registry, "SpeakerIdentifier", _偽識別器)
    return 観測されたオフライン指定


class Testオフライン指定の復元:
    @pytest.mark.parametrize("元の値", ["1", "0"])
    def test_フォールバック後も元の値に戻る(self, monkeypatch, 元の値):
        monkeypatch.setenv("HF_HUB_OFFLINE", 元の値)
        観測 = _識別器を差し替える(monkeypatch, 初回失敗=True)

        registry.get_cached_speaker_identifier("pyannote/embedding", "token", 0.5)

        assert 観測 == [元の値, "0"], "2回目は必ずオンラインで試みる"
        assert os.environ["HF_HUB_OFFLINE"] == 元の値

    def test_未設定だった場合は未設定に戻る(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        _識別器を差し替える(monkeypatch, 初回失敗=True)

        registry.get_cached_speaker_identifier("pyannote/embedding", "token", 0.5)

        assert "HF_HUB_OFFLINE" not in os.environ

    def test_2回目も失敗したら例外を送出しつつ元の値に戻る(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        _識別器を差し替える(monkeypatch, 初回失敗=True, 二回目も失敗=True)

        with pytest.raises(RuntimeError):
            registry.get_cached_speaker_identifier("pyannote/embedding", "token", 0.5)

        assert os.environ["HF_HUB_OFFLINE"] == "0"

    def test_初回で成功したら環境変数に触れない(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        観測 = _識別器を差し替える(monkeypatch, 初回失敗=False)

        registry.get_cached_speaker_identifier("pyannote/embedding", "token", 0.5)

        assert 観測 == ["1"]
        assert os.environ["HF_HUB_OFFLINE"] == "1"


class Test識別器のキャッシュ:
    def test_同じモデル名なら再利用され登録話者がリセットされる(self, monkeypatch):
        class _偽識別器:
            def __init__(self, model_name, hf_token, threshold):
                self.threshold = threshold
                self.registry_embeddings = {}
                self.unknown_counter = 1

        monkeypatch.setattr(registry, "SpeakerIdentifier", _偽識別器)

        最初 = registry.get_cached_speaker_identifier("pyannote/embedding", "t", 0.5)
        最初.registry_embeddings["アイ"] = object()
        最初.unknown_counter = 7
        二回目 = registry.get_cached_speaker_identifier("pyannote/embedding", "t", 0.3)

        assert 二回目 is 最初
        assert 二回目.registry_embeddings == {}
        assert 二回目.unknown_counter == 1
        assert 二回目.threshold == 0.3
