"""JSON ファイルの読み書き。

ジョブ状態は、処理中のワーカースレッドが書いている最中に、進捗を確認する
リクエストが読む。上書き中の中途半端な内容を読まないよう、書き込みは
一時ファイル経由でアトミックに差し替える。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.common.files import remove_quietly


def write_json(path: Path, data: Any) -> None:
    """JSON をアトミックに書き出す。

    同じディレクトリの一時ファイルへ書いてから :func:`os.replace` で差し替える。
    読み手からは「差し替え前の内容」か「差し替え後の内容」のどちらかしか見えない。

    Args:
        path: 書き出し先。親ディレクトリが無ければ作る。
        data: JSON にできる値。できない値は ``str()`` で文字列化する。

    Raises:
        TypeError: ``str()`` でも表現できない値が含まれる場合。
            このとき ``path`` の元の内容は変わらない。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同一ディレクトリに作る（別ファイルシステムだと os.replace が使えないため）
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        remove_quietly(tmp_path)
        raise


def read_json(path: Path) -> Optional[Any]:
    """JSON を読む。ファイルが無ければ ``None``。

    Raises:
        json.JSONDecodeError: 中身が JSON として読めない場合。
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
