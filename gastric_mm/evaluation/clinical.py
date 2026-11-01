"""v4 临床变量派生规则（集中在这一个模块；docs/decisions.md 第十五节第 3、4 小节）。

治疗信息、分期等**只用于统计分析，不作为任何模型的输入**。

- 分期分组 ``stage_group``：I / II / III / IV。优先读 ``ctnm_stage``（如 cIIIA、IIB、IVA）；为空时由 ``ct_t / ct_n / ct_m`` 按
  AJCC 第 8 版胃癌临床分期（cTNM）推出：cT1–2 N0 → I；cT1–2 N+ → IIA；cT3–4a N0 → IIB；cT3–4a N+ → III；cT4b M0 → IVA；M1 → IVB。
  预先规定：**IVA 并入 III**（例数少），IVB（M1）为 IV。
- 手术患者治疗模式 ``tx_pattern``：新辅助=1 → ``neoadj_surgery``；新辅助=0 且辅助化疗=1 → ``surgery_adjuvant``（参照组）；
  新辅助=0 且辅助化疗=0 → ``surgery_only``；其余缺失。
- 不可切除患者一线方案 ``tx_firstline``：取 ``first_line``；在内部训练集中不足 ``min_n`` 例的类别并入 ``chemo_other``（"化疗 + 其他"），
  合并规则由内部训练集确定后用于所有队列。参照组为单纯化疗 ``chemo``。
- 主要病理缓解 ``mpr`` = TRG 0–1。
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

STAGE_GROUPS = ["I", "II", "III", "IV"]
TX_PATTERNS = ["surgery_adjuvant", "surgery_only", "neoadj_surgery"]       # 第一个为参照组
TX_PATTERN_NAMES = {"surgery_only": "单纯手术", "surgery_adjuvant": "手术 + 辅助化疗", "neoadj_surgery": "新辅助 + 手术"}
FIRSTLINE = ["chemo", "none", "chemo_her2", "chemo_io", "chemo_cldn", "other", "chemo_other"]   # 第一个为参照组
FIRSTLINE_NAMES = {"none": "未接受系统治疗", "chemo": "单纯化疗", "chemo_her2": "化疗 + 抗 HER2", "chemo_io": "化疗 + 免疫",
                   "chemo_cldn": "化疗 + Claudin18.2 靶向", "other": "其他", "chemo_other": "化疗 + 其他（合并）"}


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def _norm(v) -> Optional[str]:
    if _blank(v):
        return None
    s = str(v).strip().upper()
    return s[1:] if s.startswith(("C", "Y", "P")) and len(s) > 1 and s[1] in "TNMI" else s


def stage_from_text(v) -> Optional[str]:
    """ctnm_stage 文本（如 cIIIA、IIB、IVA、4）→ I / II / III / IV（IVA 并入 III）。"""
    s = _norm(v)
    if s is None:
        return None
    arabic = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    if s[0] in arabic:
        s = arabic[s[0]] + s[1:]
    m = re.match(r"^(IV|III|II|I)([ABC]?)", s)
    if not m:
        return None
    roman, sub = m.group(1), m.group(2)
    if roman == "IV":
        return "III" if sub == "A" else "IV"
    return roman


def stage_from_tnm(t, n, m) -> Optional[str]:
    """AJCC 第 8 版胃癌临床分期（cTNM），IVA 并入 III。"""
    t, n, m = _norm(t), _norm(n), _norm(m)
    if m == "M1":
        return "IV"
    if t is None or t == "TX" or n is None or n == "NX" or m == "MX":
        return None
    if not (t.startswith("T") and len(t) >= 2 and t[1] in "1234"):
        return None
    tn = int(t[1])
    t4b = t.startswith("T4B")
    npos = not n.startswith("N0")
    if t4b:
        return "III"                         # IVA → 并入 III
    if tn <= 2:
        return "II" if npos else "I"
    return "III" if npos else "II"           # cT3–4a：N0 → IIB，N+ → III


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[c], errors="coerce")


def _txt(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return df[c].map(lambda v: None if _blank(v) else str(v).strip())


def tx_pattern(neoadj, adj_chemo) -> Optional[str]:
    if neoadj is None or (isinstance(neoadj, float) and np.isnan(neoadj)):
        return None
    if int(neoadj) == 1:
        return "neoadj_surgery"
    if adj_chemo is None or (isinstance(adj_chemo, float) and np.isnan(adj_chemo)):
        return None
    return "surgery_adjuvant" if int(adj_chemo) == 1 else "surgery_only"


def derive_clinical(man: pd.DataFrame) -> pd.DataFrame:
    """由（已校验的）清单派生 v4 统计分析用的临床变量。返回以 study_id 为键的表。"""
    out = pd.DataFrame({"study_id": man["study_id"].astype(str)}, index=man.index)
    out["population"] = _txt(man, "population")
    out["age"] = _num(man, "age")
    sex = _txt(man, "sex").map(lambda v: None if v is None else v.upper()[:1])
    out["sex"] = sex.map(lambda v: {"M": "M", "F": "F", "男": "M", "女": "F"}.get(v) if v else None)
    out["ecog"] = _num(man, "ecog")
    st_txt = _txt(man, "ctnm_stage").map(stage_from_text)
    st_tnm = [stage_from_tnm(t, n, m) for t, n, m in zip(_txt(man, "ct_t"), _txt(man, "ct_n"), _txt(man, "ct_m"))]
    out["stage_group"] = [a if a is not None else b for a, b in zip(st_txt, st_tnm)]
    out["ct_n_pos"] = _txt(man, "ct_n").map(lambda v: None if _norm(v) in (None, "NX") else float(not _norm(v).startswith("N0")))
    for c in ("neoadj", "adj_chemo", "immuno", "pcr", "adj_rt", "d2"):
        out[c] = _num(man, c)
    for c in ("first_line", "neoadj_regimen", "immuno_setting", "surgery_type"):
        out[c] = _txt(man, c)
    out["trg"] = _num(man, "trg")
    out["mpr"] = out["trg"].map(lambda v: np.nan if np.isnan(v) else float(v <= 1))
    out["n_lines"] = _num(man, "n_lines")
    out["diag_year"] = _num(man, "diag_year")
    res = out["population"] == "resected"
    out["tx_pattern"] = [tx_pattern(a, b) if r else None for a, b, r in zip(out["neoadj"], out["adj_chemo"], res)]
    out.loc[out["population"] != "unresectable", "first_line"] = None
    return out.reset_index(drop=True)


def firstline_mapping(first_line: pd.Series, min_n: int) -> Dict[str, str]:
    """由内部训练集确定一线方案的合并规则：不足 min_n 例的靶向 / 免疫 / 其他类别并入 chemo_other（未接受系统治疗、单纯化疗不合并）。"""
    cnt = first_line.dropna().value_counts().to_dict()
    mp = {}
    for k in FIRSTLINE:
        if k in ("chemo_other",):
            continue
        if k in ("none", "chemo") or cnt.get(k, 0) >= min_n:
            mp[k] = k
        else:
            mp[k] = "chemo_other"
    return mp
