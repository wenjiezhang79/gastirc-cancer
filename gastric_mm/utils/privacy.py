"""隐私保护工具。

规则（见 CLAUDE.md 第 3 节）：
- 日志、报告、状态表中只出现 ``study_id``，不出现病人级路径（切片、CT 文件夹、分割文件）。
- 第三方库的异常信息里可能带有文件路径，写入日志前用 :func:`scrub` 处理。
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

# 形如 /a/b/c 或 C:\a\b 的路径片段
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][^\s'\"<>|:;,()\[\]{}]+){2,}")

PLACEHOLDER = "<路径已隐去>"


def _variants(text: str) -> list:
    """一个敏感路径的若干变体：原样、绝对路径、文件名、去扩展名的文件名、上级目录名。"""
    out = set()
    if not text:
        return []
    text = str(text)
    out.add(text)
    try:
        out.add(os.path.abspath(text))
    except Exception:  # pragma: no cover
        pass
    base = os.path.basename(text.rstrip("/\\"))
    if base:
        out.add(base)
        stem = base
        for ext in (".nii.gz", ".mrk.json"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
        stem = os.path.splitext(stem)[0]
        if len(stem) >= 4:
            out.add(stem)
    parent = os.path.basename(os.path.dirname(text.rstrip("/\\")))
    if parent and len(parent) >= 4:
        out.add(parent)
    # 长的先替换，避免部分替换
    return sorted((v for v in out if v), key=len, reverse=True)


def scrub(text: str, sensitive: Optional[Iterable[str]] = None) -> str:
    """去掉文本中的病人级路径。

    参数
    ----
    text: 待处理文本（异常信息、调用栈等）。
    sensitive: 已知敏感字符串（如该病例的 slide_path、ct_dir），连同文件名、上级目录名一并替换。
    """
    if text is None:
        return ""
    text = str(text)
    for s in sensitive or []:
        for v in _variants(s):
            # 太短或纯数字的片段不替换，避免误伤正常文字
            if len(v) < 4:
                continue
            text = text.replace(v, PLACEHOLDER)

    def _repl(m: "re.Match") -> str:
        s = m.group(0)
        # 代码文件路径（调用栈）保留，便于定位 bug
        if "gastric_mm" in s or "site-packages" in s or "dist-packages" in s or "/lib/python" in s:
            return s
        return PLACEHOLDER

    return _PATH_RE.sub(_repl, text)


# 病人身份相关的列名关键字：清单中出现即报错
PHI_COLUMN_KEYWORDS = (
    "name", "姓名", "住院号", "病案号", "mrn", "patient_id", "patientid", "身份证", "id_card",
    "idcard", "phone", "电话", "手机", "address", "地址", "birth", "生日", "出生",
)


def looks_like_phi_column(col: str) -> bool:
    c = str(col).strip().lower()
    return any(k in c for k in PHI_COLUMN_KEYWORDS)
