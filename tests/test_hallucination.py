"""文字起こしのハルシネーション判定のテスト。"""

from __future__ import annotations

import pytest

from src.transcription.hallucination import HallucinationFilter, load_phrases


@pytest.fixture
def フィルタ() -> HallucinationFilter:
    return HallucinationFilter()


class Test定型フレーズ:
    @pytest.mark.parametrize(
        "text",
        [
            "ご視聴ありがとうございました",
            "ご清聴ありがとうございました",
            "チャンネル登録をお願いします",
        ],
    )
    def test_リストのフレーズは除去される(self, フィルタ, text):
        assert フィルタ.reason_to_drop(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "ご視聴ありがとうございました。",
            "  ご視聴ありがとうございました！ ",
            "ご視聴、ありがとうございました",
        ],
    )
    def test_句読点や空白が付いていても除去される(self, フィルタ, text):
        assert フィルタ.reason_to_drop(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "今日はご視聴ありがとうございました、と彼は言った",
            "ありがとうございます",
            "おはようございます",
        ],
    )
    def test_通常の文は残る(self, フィルタ, text):
        assert フィルタ.reason_to_drop(text) is None

    def test_追加のフレーズを渡せる(self):
        フィルタ = HallucinationFilter(phrases=["おやすみなさい"])

        assert フィルタ.reason_to_drop("おやすみなさい") is not None

    def test_環境変数のファイルからフレーズを読める(self, tmp_path, monkeypatch):
        path = tmp_path / "phrases.txt"
        path.write_text("# コメント行\n\n提供でお送りしました\n", encoding="utf-8")
        monkeypatch.setenv("HALLUCINATION_PHRASES_FILE", str(path))

        phrases = load_phrases()

        assert "提供でお送りしました" in phrases
        assert "# コメント行" not in phrases
        # 既定のフレーズも残る
        assert "ご視聴ありがとうございました" in phrases

    def test_ファイルが無くても既定のフレーズだけで動く(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_PHRASES_FILE", str(tmp_path / "none.txt"))

        assert "ご視聴ありがとうございました" in load_phrases()


class Test同一文字列の反復:
    @pytest.mark.parametrize("text", ["はいはいはい", "あーあーあーあー", "ああああ"])
    def test_短い単位の3回以上の反復は除去される(self, フィルタ, text):
        assert フィルタ.reason_to_drop(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "そうそう",  # 2回なので残す
            "はいはい",
            "ありがとうありがとう",  # 単位が長い
            "そうですね",
        ],
    )
    def test_2回まで_または単位が長い反復は残る(self, フィルタ, text):
        assert フィルタ.reason_to_drop(text) is None

    def test_反復のあとに別の語が続く場合は残る(self, フィルタ):
        assert フィルタ.reason_to_drop("はいはいはい、わかりました") is None


class Test直前の行との連続重複:
    def test_同じテキストの2回目以降が除去される(self, フィルタ):
        assert フィルタ.reason_to_drop("それでは始めます") is None
        assert フィルタ.reason_to_drop("それでは始めます") is not None
        assert フィルタ.reason_to_drop("それでは始めます") is not None

    def test_間に別のテキストが入れば再び採用される(self, フィルタ):
        assert フィルタ.reason_to_drop("それでは始めます") is None
        assert フィルタ.reason_to_drop("よろしくお願いします") is None
        assert フィルタ.reason_to_drop("それでは始めます") is None

    def test_除去された行は直前の行として記憶されない(self, フィルタ):
        """幻聴を挟んでも、その前の行との比較が続く。"""
        assert フィルタ.reason_to_drop("こんにちは") is None
        assert フィルタ.reason_to_drop("ご視聴ありがとうございました") is not None
        assert フィルタ.reason_to_drop("こんにちは") is not None
