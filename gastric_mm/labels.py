"""训练标签派生规则（全项目唯一来源）。

由清单原始字段生成训练标签；缺失记为 NaN，训练时对应任务的损失被屏蔽。

规则（见 CLAUDE.md 5.1 与 6.4）：
- Lauren：intestinal=0, diffuse=1, mixed=2；补充二分类 lauren_intestinal（肠型=1，弥漫/混合=0）。
- HER2 阳性：IHC 3+，或 IHC 2+ 且 FISH 阳性。IHC 2+ 而无 FISH：按配置（默认缺失）。
  IHC 缺失时：FISH 阳性 → 阳性；否则缺失。
  补充三分类 her2_3cls：0=HER2 0；1=HER2-low（IHC 1+，或 2+ 且 FISH 阴性）；2=阳性。
- MSI：MSI-H=1，MSS=0。EBV：pos=1，neg=0。
- PD-L1：每个阈值 t 生成 pdl1_cps{t} = (CPS ≥ t)。
- Claudin18.2：cldn182_pos = (pct ≥ 阈值，默认 75)。
- Ki-67：ki67_frac = pct/100（回归）；阈值非空时生成 ki67_high。
- 标签来源：<任务>_source 列优先，否则用 label_source。来源为 post_neoadj 且
  labels.use_post_neoadj=false 时，该任务的分型/标志物标签置为缺失（生存标签不受影响）。
- 生存：os/dfs/pfs 的 time（月）与 event（0/1）原样复制；时间或事件任一缺失则两者都置缺失。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config

LAUREN_CLASSES = ["intestinal", "diffuse", "mixed"]


@dataclass(frozen=True)
class TaskDef:
    name: str          # 标签列名
    kind: str          # multiclass / binary / regression / survival
    source_task: str   # 用于查找 <source_task>_source 列
    main: bool         # 是否主分析定义
    n_classes: int = 2


def task_definitions(cfg: Config) -> List[TaskDef]:
    lc = cfg.labels
    tasks = [
        TaskDef("lauren", "multiclass", "lauren", True, 3),
        TaskDef("lauren_intestinal", "binary", "lauren", False),
        TaskDef("her2_pos", "binary", "her2", True),
        TaskDef("her2_3cls", "multiclass", "her2", False, 3),
        TaskDef("msi_h", "binary", "msi", True),
        TaskDef("ebv_pos", "binary", "ebv", True),
    ]
    for t in lc.pdl1_thresholds:
        tasks.append(TaskDef(f"pdl1_cps{int(t)}", "binary", "pdl1", int(t) == int(lc.pdl1_main_threshold)))
    tasks.append(TaskDef("cldn182_pos", "binary", "cldn182", True))
    tasks.append(TaskDef("ki67_frac", "regression", "ki67", False))
    if lc.get("ki67_threshold") is not None:
        tasks.append(TaskDef("ki67_high", "binary", "ki67", True))
    for ep in ("os", "dfs", "pfs"):
        tasks.append(TaskDef(ep, "survival", ep, ep == "os"))
    return tasks


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_col(df, name), errors="coerce").astype(float)


def _str(df: pd.DataFrame, name: str) -> pd.Series:
    s = _col(df, name)
    return s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" else str(v).strip())


def _source(df: pd.DataFrame, task: str) -> pd.Series:
    specific = _str(df, f"{task}_source")
    general = _str(df, "label_source")
    return specific.where(specific.notna(), general)


def derive_her2(ihc: pd.Series, fish: pd.Series, two_plus_without_fish: str = "missing"):
    """返回 (her2_pos, her2_3cls) 两个 float Series（NaN = 缺失）。"""
    pos = pd.Series(np.nan, index=ihc.index, dtype=float)
    three = pd.Series(np.nan, index=ihc.index, dtype=float)
    for i in ihc.index:
        a, f = ihc[i], fish[i]
        if a == "3+":
            pos[i], three[i] = 1, 2
        elif a == "2+":
            if f == "pos":
                pos[i], three[i] = 1, 2
            elif f == "neg":
                pos[i], three[i] = 0, 1
            else:
                if two_plus_without_fish == "negative":
                    pos[i], three[i] = 0, 1
        elif a == "1+":
            pos[i], three[i] = 0, 1
        elif a == "0":
            pos[i], three[i] = 0, 0
        elif a is None:
            if f == "pos":
                pos[i], three[i] = 1, 2
    return pos, three


def derive_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """由（已校验的）清单派生训练标签。返回含 study_id、center 与各任务标签列的表。"""
    lc = cfg.labels
    out = pd.DataFrame({"study_id": _str(df, "study_id"), "center": _str(df, "center")}, index=df.index)

    lauren = _str(df, "lauren")
    out["lauren"] = lauren.map(lambda v: float(LAUREN_CLASSES.index(v)) if v in LAUREN_CLASSES else np.nan)
    out["lauren_intestinal"] = lauren.map(
        lambda v: 1.0 if v == "intestinal" else (0.0 if v in ("diffuse", "mixed") else np.nan)
    )

    her2_pos, her2_3 = derive_her2(_str(df, "her2_ihc"), _str(df, "her2_fish"),
                                   str(lc.get("her2_2plus_without_fish", "missing")))
    out["her2_pos"] = her2_pos
    out["her2_3cls"] = her2_3

    msi = _str(df, "msi")
    out["msi_h"] = msi.map(lambda v: 1.0 if v == "MSI-H" else (0.0 if v == "MSS" else np.nan))
    ebv = _str(df, "ebv")
    out["ebv_pos"] = ebv.map(lambda v: 1.0 if v == "pos" else (0.0 if v == "neg" else np.nan))

    cps = _num(df, "pdl1_cps")
    for t in lc.pdl1_thresholds:
        out[f"pdl1_cps{int(t)}"] = np.where(cps.isna(), np.nan, (cps >= float(t)).astype(float))

    cl = _num(df, "cldn182_pct")
    out["cldn182_pos"] = np.where(cl.isna(), np.nan, (cl >= float(lc.cldn182_threshold)).astype(float))

    ki = _num(df, "ki67_pct")
    out["ki67_frac"] = ki / 100.0
    if lc.get("ki67_threshold") is not None:
        out["ki67_high"] = np.where(ki.isna(), np.nan, (ki >= float(lc.ki67_threshold)).astype(float))

    # 标签来源屏蔽
    if not lc.get("use_post_neoadj", False):
        for t in task_definitions(cfg):
            if t.kind == "survival":
                continue
            src = _source(df, t.source_task)
            mask = src == "post_neoadj"
            if t.name in out.columns:
                out.loc[mask.values, t.name] = np.nan

    # 生存
    for ep in ("os", "dfs", "pfs"):
        tm = _num(df, f"{ep}_months")
        ev = _num(df, f"{ep}_event")
        bad = tm.isna() | ev.isna()
        out[f"{ep}_time"] = tm.where(~bad)
        out[f"{ep}_event"] = ev.where(~bad)

    out["population"] = _str(df, "population")
    return out


def label_columns(cfg: Config) -> Dict[str, TaskDef]:
    """标签列名 → 任务定义（生存任务对应 <ep>_time / <ep>_event 两列）。"""
    return {t.name: t for t in task_definitions(cfg)}


def label_summary(labels: pd.DataFrame, cfg: Config) -> List[dict]:
    """按中心统计每个任务的可用标签数与阳性率。"""
    rows = []
    centers = [c for c in cfg.manifest.centers if c in set(labels["center"].dropna())]
    for t in task_definitions(cfg):
        for c in centers + ["全部"]:
            sub = labels if c == "全部" else labels[labels["center"] == c]
            row = {"task": t.name, "main": "是" if t.main else "", "center": c}
            if t.kind == "survival":
                tm = sub[f"{t.name}_time"]
                ev = sub[f"{t.name}_event"]
                n = int(tm.notna().sum())
                row.update(n_labeled=n, n_positive=int((ev == 1).sum()),
                           rate=(float((ev == 1).sum()) / n if n else None),
                           note="事件数 / 事件率")
            elif t.kind == "regression":
                v = sub[t.name]
                n = int(v.notna().sum())
                row.update(n_labeled=n, n_positive=None,
                           rate=(float(v.median()) if n else None), note="中位数")
            elif t.kind == "multiclass":
                v = sub[t.name]
                n = int(v.notna().sum())
                counts = [int((v == k).sum()) for k in range(t.n_classes)]
                row.update(n_labeled=n, n_positive=None, rate=None,
                           note="各类数量 " + "/".join(str(x) for x in counts))
            else:
                v = sub[t.name]
                n = int(v.notna().sum())
                p = int((v == 1).sum())
                row.update(n_labeled=n, n_positive=p, rate=(p / n if n else None), note="阳性数 / 阳性率")
            rows.append(row)
    return rows


def positive_count(labels: pd.DataFrame, col: str) -> Optional[int]:
    if col not in labels.columns:
        return None
    return int((labels[col] == 1).sum())
