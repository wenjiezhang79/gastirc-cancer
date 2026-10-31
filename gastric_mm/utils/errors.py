"""面向研究者的报错。

``UserError`` 用于"不是代码 bug、而是配置或数据需要研究者处理"的情况：
报错信息说明原因，``hint`` 说明下一步该怎么做。命令行入口捕获后只打印这两部分，
不打印 Python 调用栈。
"""

from __future__ import annotations


class UserError(Exception):
    """需要研究者处理的问题（配置、数据、权重文件等）。"""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - 简单拼接
        if self.hint:
            return f"{self.message}\n  → 怎么办：{self.hint}"
        return self.message


class CaseError(Exception):
    """单个病例处理失败（不影响其他病例）。信息中不得包含病人级路径。"""
