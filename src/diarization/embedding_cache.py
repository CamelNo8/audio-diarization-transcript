"""話者エンベディングの永続キャッシュ。

同じ音声ファイル（＝同じ内容）に対する話者埋め込みは、モデルと前処理が
変わらない限り常に同じ値になる。声紋DBが大きくなると、文字起こし・ラベル付けの
たびに DB 全話者を再エンベディングするコスト（pyannote 推論 = GPU）が O(N) で
効いてくる。本モジュールはファイル内容ハッシュをキーに埋め込みベクトルを
ディスクへ永続化し、2 回目以降の再計算を回避する。

キャッシュ構造:
  <cache_dir>/<namespace>/<sha256>.npy   # 正規化済み埋め込み (shape (1, D), float32)

namespace には呼び出し側がモデル名・前処理パラメータ・スキーマ版を畳み込んだ
文字列を渡す（これらが変わると別名前空間になり、自動的にキャッシュが分離される）。

環境変数:
  EMBEDDING_CACHE_DIR  キャッシュ保存先（既定: <repo>/embedding_cache）
  EMBEDDING_CACHE_OFF  "1" でキャッシュ無効化（常に compute_fn を呼ぶ）
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.common.logging import get_logger
from src.config import DEFAULT_EMBEDDING_CACHE_DIR

logger = get_logger(__name__)

_HASH_CHUNK = 1 << 20  # 1 MiB
_SAFE_NAMESPACE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_cache_dir() -> Path:
    """キャッシュの保存先を返す（環境変数 EMBEDDING_CACHE_DIR で上書き可）。"""
    env = os.getenv("EMBEDDING_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_EMBEDDING_CACHE_DIR


def _hash_file(path: Path) -> str:
    """ファイル内容の sha256（16 進）を返す。大きめファイルも逐次読みで対応。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_namespace(namespace: str) -> str:
    """namespace をディレクトリ名に使える文字列へ整える。"""
    safe = _SAFE_NAMESPACE_RE.sub("_", namespace.strip())
    return safe or "default"


class EmbeddingCache:
    """ファイル内容ハッシュをキーにした埋め込みの永続キャッシュ。"""

    def __init__(
        self, cache_dir: Optional[Path] = None, enabled: Optional[bool] = None
    ):
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else default_cache_dir()
        )
        if enabled is None:
            enabled = os.getenv("EMBEDDING_CACHE_OFF", "") != "1"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _entry_path(self, namespace: str, file_hash: str) -> Path:
        return self.cache_dir / _safe_namespace(namespace) / f"{file_hash}.npy"

    def get_or_compute(
        self,
        namespace: str,
        audio_path: Path,
        compute_fn: Callable[[Path], np.ndarray],
    ) -> np.ndarray:
        """audio_path の埋め込みをキャッシュ経由で取得する。

        キャッシュヒット時は .npy をロード、ミス時は compute_fn(audio_path) を呼んで
        結果を保存する。キャッシュ無効時（EMBEDDING_CACHE_OFF=1）は常に計算する。
        """
        audio_path = Path(audio_path)
        if not self.enabled:
            return compute_fn(audio_path)

        try:
            file_hash = _hash_file(audio_path)
        except OSError as e:
            logger.warning(
                f"埋め込みキャッシュ: ハッシュ計算に失敗（計算にフォールバック）: {e}"
            )
            return compute_fn(audio_path)

        entry = self._entry_path(namespace, file_hash)
        cached = self._load_entry(entry, audio_path, file_hash)
        if cached is not None:
            return cached

        emb = compute_fn(audio_path)
        self.misses += 1
        self._store_entry(entry, emb, audio_path, file_hash)
        return emb

    def _load_entry(
        self, entry: Path, audio_path: Path, file_hash: str
    ) -> Optional[np.ndarray]:
        """キャッシュエントリを読む。無い／壊れている場合は ``None``。"""
        if not entry.exists():
            return None
        try:
            emb = np.load(entry)
        except (OSError, ValueError) as e:
            logger.warning(
                f"埋め込みキャッシュ: 破損エントリを無視して再計算します ({entry}): {e}"
            )
            return None
        self.hits += 1
        logger.info(f"埋め込みキャッシュ HIT: {audio_path.name} [{file_hash[:12]}]")
        return emb

    def _store_entry(
        self, entry: Path, emb: np.ndarray, audio_path: Path, file_hash: str
    ) -> None:
        """計算結果をキャッシュへ保存する。保存に失敗しても処理は続行する。"""
        try:
            entry.parent.mkdir(parents=True, exist_ok=True)
            # np.save は拡張子が .npy でないと .npy を付け足すため、
            # tmp も .npy で終わらせる。
            tmp = entry.parent / f".{file_hash}.{os.getpid()}.tmp.npy"
            np.save(tmp, emb)
            os.replace(tmp, entry)  # 原子的に確定（部分書き込みを避ける）
            logger.info(
                f"埋め込みキャッシュ STORE: {audio_path.name} [{file_hash[:12]}]"
            )
        except OSError as e:
            logger.warning(
                f"埋め込みキャッシュ: 保存に失敗しました"
                f"（今回は計算結果を返します）: {e}"
            )

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "enabled": self.enabled}


# プロセス内で共有する既定キャッシュ（明示注入しない呼び出し側向け）
_DEFAULT_CACHE: Optional[EmbeddingCache] = None


def get_default_cache() -> EmbeddingCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = EmbeddingCache()
    return _DEFAULT_CACHE
