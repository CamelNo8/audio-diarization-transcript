"""Web 層のエラー型。"""

from __future__ import annotations


class WebInputError(Exception):
    """利用者の入力が受け付けられないことを表す。

    メッセージはそのまま画面に出す前提で書く（規約8.2）。ルート側でこれを
    捕捉し、:func:`src.web.templating.render_error` でフラグメントに変換する。
    """
