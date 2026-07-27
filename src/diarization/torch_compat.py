"""pyannote のチェックポイントを読むための torch 互換パッチ。

PyTorch 2.6+ で ``torch.load`` のデフォルトが ``weights_only=True`` に変わり、
pyannote のチェックポイント（pytorch-lightning の callback 等を pickle で含む）が
読めなくなった。pyannote / lightning は ``weights_only=True`` を明示して呼ぶため、
既定値の変更では足りず強制的に上書きする必要がある。
pyannote モデルは Hugging Face 由来で信頼できるため安全。

``pipeline`` と ``speaker_identifier`` の両方が pyannote を読み込むため、
パッチはこのモジュールに置いて共有する。
"""

from __future__ import annotations

import torch

#: 二重適用を避けるための目印。
_PATCH_FLAG = "_pyannote_compat_patched"


def patch_torch_load() -> None:
    """``torch.load`` を ``weights_only=False`` 固定にする。

    適用済みの場合は何もしない。
    """
    if getattr(torch.load, _PATCH_FLAG, False):
        return

    original_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs["weights_only"] = False  # 強制上書き
        return original_load(*args, **kwargs)

    setattr(_torch_load_compat, _PATCH_FLAG, True)
    torch.load = _torch_load_compat
