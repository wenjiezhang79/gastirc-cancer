"""按任务计算预测指标（交叉验证折外预测与最终测试共用）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import Config
from ..labels import task_definitions
from ..training.task_spec import TaskSpec


def merged(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return pred.merge(labels, on="study_id", how="left", suffixes=("", "_label"))


def _row(task, metric, est, n, n_pos=None, main=False, note=""):
    v, lo, hi = est if isinstance(est, tuple) else (est, float("nan"), float("nan"))
    return {"task": task, "metric": metric, "value": v, "lo": lo, "hi": hi, "n": int(n),
            "n_pos": None if n_pos is None else int(n_pos), "main": main, "note": note}


def evaluate(df: pd.DataFrame, spec: TaskSpec, cfg: Config, n_boot: int = 1000, seed: int = 0,
             thresholds: Optional[Dict[str, float]] = None) -> List[dict]:
    """df：预测与派生标签合并后的表。返回指标行列表（含 bootstrap 95% CI）。"""
    mains = {t.name for t in task_definitions(cfg) if t.main}
    rows: List[dict] = []
    for t, c in spec.multiclass.items():
        cols = [f"{t}_p{k}" for k in range(c)]
        y = df[t].to_numpy(float)
        P = df[cols].to_numpy(float)
        ok = np.isfinite(y)
        fn = lambda yy, pp: M.multiclass_auc(yy, pp)[0]  # noqa: E731
        rows.append(_row(t, "宏平均AUC", M.bootstrap_ci(fn, [y[ok], P[ok]], n_boot, seed, strata=y[ok]),
                         ok.sum(), main=t in mains))
        _, per = M.multiclass_auc(y, P)
        for k, a in enumerate(per):
            rows.append(_row(t, f"类别{k}_AUC", a, ok.sum(), int((y == k).sum())))
        rows.append(_row(t, "平衡准确率", M.balanced_accuracy(y, P), ok.sum()))
        if t == "lauren":
            yb = df["lauren_intestinal"].to_numpy(float)
            okb = np.isfinite(yb)
            rows.append(_row("lauren_intestinal", "AUC",
                             M.bootstrap_ci(M.auc_binary, [yb[okb], P[okb, 0]], n_boot, seed, strata=yb[okb]),
                             okb.sum(), int((yb == 1).sum()), note="由三分类概率换算"))
    for t in spec.binary:
        y = df[t].to_numpy(float)
        s = df[f"{t}_prob"].to_numpy(float)
        ok = np.isfinite(y)
        n_pos = int((y[ok] == 1).sum())
        rows.append(_row(t, "AUC", M.bootstrap_ci(M.auc_binary, [y[ok], s[ok]], n_boot, seed, strata=y[ok]),
                         ok.sum(), n_pos, main=t in mains))
        rows.append(_row(t, "PR-AUC", M.pr_auc(y, s), ok.sum(), n_pos, note=f"阳性率 {n_pos / max(ok.sum(), 1):.3f}"))
        if thresholds and t in thresholds and np.isfinite(thresholds[t]):
            se, sp = M.sens_spec(y, s, thresholds[t])
            rows.append(_row(t, "敏感度", se, ok.sum(), n_pos, note=f"阈值 {thresholds[t]:.3f}"))
            rows.append(_row(t, "特异度", sp, ok.sum(), n_pos, note=f"阈值 {thresholds[t]:.3f}"))
    for t in spec.regression:
        rows.append(_row(t, "Spearman", M.spearman(df[t], df[f"{t}_pred"]), df[t].notna().sum()))
    for ep in spec.survival:
        tm, ev, r = (df[f"{ep}_time"].to_numpy(float), df[f"{ep}_event"].to_numpy(float),
                     df[f"{ep}_risk"].to_numpy(float))
        ok = np.isfinite(tm) & np.isfinite(ev)
        rows.append(_row(ep, "C-index", M.bootstrap_ci(M.c_index, [tm[ok], ev[ok], r[ok]], n_boot, seed),
                         ok.sum(), int((ev[ok] == 1).sum()), main=ep in mains, note="事件数见 n_pos"))
        if spec.n_pops(ep) > 1:
            for pop in ("resected", "unresectable"):
                m = ok & (df["population"] == pop).to_numpy()
                if m.sum() >= 5:
                    rows.append(_row(ep, f"C-index（{pop}）",
                                     M.bootstrap_ci(M.c_index, [tm[m], ev[m], r[m]], n_boot, seed),
                                     m.sum(), int((ev[m] == 1).sum())))
    return rows


def primary_value(df: pd.DataFrame, spec: TaskSpec, task: str) -> float:
    """单个任务的主要指标：多分类=宏平均 AUC，二分类=AUC，生存=C-index。"""
    if task in spec.multiclass:
        cols = [f"{task}_p{k}" for k in range(spec.multiclass[task])]
        return M.multiclass_auc(df[task].to_numpy(float), df[cols].to_numpy(float))[0]
    if task in spec.binary:
        return M.auc_binary(df[task].to_numpy(float), df[f"{task}_prob"].to_numpy(float))
    if task in spec.survival:
        return M.c_index(df[f"{task}_time"], df[f"{task}_event"], df[f"{task}_risk"])
    if task in spec.regression:
        return M.spearman(df[task], df[f"{task}_pred"])
    return float("nan")


def metrics_table_md(rows: List[dict], title: str = "") -> str:
    L = [f"### {title}", ""] if title else []
    L += ["| 任务 | 主分析 | 指标 | 值 | 95% CI | n | 阳性/事件 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f"{r['lo']:.3f}–{r['hi']:.3f}" if np.isfinite(r["lo"]) else ""
        v = f"{r['value']:.3f}" if r["value"] is not None and np.isfinite(r["value"]) else "—"
        L.append(f"| {r['task']} | {'是' if r['main'] else ''} | {r['metric']} | {v} | {ci} | {r['n']} | "
                 f"{'' if r['n_pos'] is None else r['n_pos']} | {r['note']} |")
    L.append("")
    return "\n".join(L)
