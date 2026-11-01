"""论文用图（matplotlib，无界面后端）：ROC、带风险人数表的 KM 曲线、森林图、校准曲线、决策曲线、列线图。

每个函数把图保存为 PNG（300 dpi）与 PDF；文件名由调用方给出（只用 study_id 或任务名，不含任何病人信息）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .survival import kaplan_meier, n_at_risk

COLORS = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#566573"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    return plt


def _save(fig, path: Path) -> List[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in (".png", ".pdf"):
        p = path.with_suffix(ext)
        fig.savefig(p, dpi=300 if ext == ".png" else None, bbox_inches="tight")
        out.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def roc_plot(curves: Sequence[Tuple[str, np.ndarray, np.ndarray]], path: Path, title: str = "") -> List[Path]:
    """curves：[(名称, y, score)]，图例中注明 AUC。"""
    from .classification import delong_auc
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for k, (name, y, s) in enumerate(curves):
        y, s = np.asarray(y, float), np.asarray(s, float)
        ok = np.isfinite(y) & np.isfinite(s)
        y, s = y[ok], s[ok]
        if len(np.unique(y)) < 2:
            continue
        order = np.argsort(-s, kind="mergesort")
        ys = y[order]
        tpr = np.r_[0, np.cumsum(ys) / ys.sum()]
        fpr = np.r_[0, np.cumsum(1 - ys) / (1 - ys).sum()]
        a = delong_auc(y, s)
        ax.plot(fpr, tpr, color=COLORS[k % len(COLORS)], lw=1.4,
                label=f"{name}  AUC {a['auc']:.3f} ({a['lo']:.3f}–{a['hi']:.3f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel("1 − 特异度")
    ax.set_ylabel("敏感度")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    return _save(fig, path)


def km_plot(groups: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path, title: str = "", xlabel: str = "时间（月）",
            ylabel: str = "生存率", risk_times: Optional[Sequence[float]] = None, annotation: str = "") -> List[Path]:
    """groups：{组名: (time, event)}；下方附各时间点的风险人数表。"""
    plt = _plt()
    fig = plt.figure(figsize=(4.2, 3.9))
    ax = fig.add_axes([0.14, 0.33, 0.82, 0.6])
    tmax = max((float(np.nanmax(t)) for t, _ in groups.values() if len(t)), default=1.0)
    risk_times = list(risk_times) if risk_times is not None else list(np.linspace(0, tmax, 6).round(0))
    for k, (name, (t, e)) in enumerate(groups.items()):
        if len(t) == 0:
            continue
        km = kaplan_meier(t, e)
        c = COLORS[k % len(COLORS)]
        ax.step(km.time, km.surv, where="post", color=c, lw=1.4, label=f"{name}（n={len(t)}）")
        ax.fill_between(km.time, km.lo, km.hi, step="post", color=c, alpha=0.12, lw=0)
        cens = np.asarray(t)[np.asarray(e) == 0]
        ax.plot(cens, [km.at(x) for x in cens], "|", color=c, ms=5, mew=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, tmax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    if annotation:
        ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=7)
    tab = fig.add_axes([0.14, 0.05, 0.82, 0.18])
    tab.axis("off")
    names = list(groups)
    tab.text(-0.02, 1.0, "风险人数", transform=tab.transAxes, ha="right", va="top", fontsize=7)
    for k, name in enumerate(names):
        t, _ = groups[name]
        y = 0.75 - k * 0.3
        tab.text(-0.02, y, name, transform=tab.transAxes, ha="right", va="center", fontsize=7,
                 color=COLORS[k % len(COLORS)])
        for x, n in zip(risk_times, n_at_risk(t, risk_times)):
            tab.text(x / tmax if tmax else 0, y, str(n), transform=tab.transAxes, ha="center", va="center", fontsize=7)
    tab.text(0.5, -0.25, xlabel, transform=tab.transAxes, ha="center", fontsize=8)
    return _save(fig, path)


def forest_plot(rows: Sequence[dict], path: Path, title: str = "", xlabel: str = "HR（95% CI）") -> List[Path]:
    """rows：[{label, hr, lo, hi, n?, note?}]；对数坐标。"""
    plt = _plt()
    rows = [r for r in rows if r.get("hr") is not None]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(rows) + 1.0))
    for i, r in enumerate(rows):
        y = len(rows) - i
        if np.isfinite(r.get("hr", np.nan)):
            ax.plot([r["lo"], r["hi"]], [y, y], color="#333333", lw=1)
            ax.plot(r["hr"], y, "s", color=COLORS[0], ms=4)
            txt = f"{r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})"
        else:
            txt = r.get("note", "—")
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvline(1.0, color="#999999", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([f"{r['label']}" + (f"（n={r['n']}）" if r.get("n") is not None else "") for r in rows], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, path)


def calibration_plot(series: Sequence[Tuple[str, list]], path: Path, title: str = "", xlabel: str = "预测概率",
                     ylabel: str = "实际比例") -> List[Path]:
    """series：[(名称, [{pred, obs, lo, hi}])]。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for k, (name, pts) in enumerate(series):
        if not pts:
            continue
        x = [p["pred"] for p in pts]
        y = [p["obs"] for p in pts]
        err = [[p["obs"] - p["lo"] for p in pts], [p["hi"] - p["obs"] for p in pts]]
        ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[k % len(COLORS)], ms=3, lw=1, capsize=2, label=name)
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    return _save(fig, path)


def dca_plot(curves: Dict[str, "object"], path: Path, title: str = "", max_threshold: float = 0.8) -> List[Path]:
    """curves：{名称: decision_curve 的 DataFrame}；同时画"全部处理 / 全部不处理"。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ref = None
    lo_y = 0.0
    for k, (name, df) in enumerate(curves.items()):
        d = df[df.threshold <= max_threshold]
        ax.plot(d.threshold, d.model, color=COLORS[k % len(COLORS)], lw=1.3, label=name)
        ref = d
        lo_y = min(lo_y, float(d.model.min()))
    if ref is not None:
        ax.plot(ref.threshold, ref.treat_all, color="#888888", lw=0.9, ls="--", label="全部处理")
        ax.plot(ref.threshold, ref.treat_none, color="#444444", lw=0.9, ls=":", label="全部不处理")
        top = max(float(ref.treat_all.max()), max(float(df.model.max()) for df in curves.values()))
        ax.set_ylim(max(-0.05, lo_y - 0.02), top * 1.1 + 1e-3)
    ax.set_xlabel("阈值概率")
    ax.set_ylabel("净获益")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, path)


def nomogram_plot(terms: Sequence[dict], horizons: Sequence[float], base_surv: Dict[float, float], path: Path,
                  title: str = "") -> List[Path]:
    """简化列线图。terms：[{name, coef, values: [(标签, 取值)]}]（连续变量给若干刻度，分类变量给各水平的 0/1）；
    base_surv：{时间点: 线性预测为 0 时的生存率}。每个变量按 |coef × 取值范围| 换算为 0–100 分。"""
    plt = _plt()
    spans = []
    for tm in terms:
        v = np.array([x for _, x in tm["values"]], float) * tm["coef"]
        spans.append(v.max() - v.min())
    scale = 100.0 / max(max(spans), 1e-12)
    n_rows = len(terms) + 2 + len(horizons)
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * n_rows + 0.6))
    ax.axis("off")
    y = n_rows
    ax.plot([0, 100], [y, y], color="k", lw=0.8)
    for p in range(0, 101, 10):
        ax.text(p, y + 0.25, str(p), ha="center", fontsize=6)
    ax.text(-3, y, "分数", ha="right", va="center", fontsize=8)
    mins = []
    for tm in terms:
        y -= 1
        v = np.array([x for _, x in tm["values"]], float) * tm["coef"]
        m0 = v.min()
        mins.append(m0)
        pts = (v - m0) * scale
        ax.plot([pts.min(), pts.max()], [y, y], color="k", lw=0.8)
        for (lab, _), p in zip(tm["values"], pts):
            ax.plot([p, p], [y, y + 0.12], color="k", lw=0.6)
            ax.text(p, y - 0.32, lab, ha="center", fontsize=6)
        ax.text(-3, y, tm["name"], ha="right", va="center", fontsize=8)
    y -= 1
    total_max = 100 * len(terms)
    ax.plot([0, 100], [y, y], color="k", lw=0.8)
    for p in np.linspace(0, total_max, 6):
        ax.text(p / total_max * 100, y - 0.32, f"{p:.0f}", ha="center", fontsize=6)
    ax.text(-3, y, "总分", ha="right", va="center", fontsize=8)
    lp_min = float(np.sum(mins))
    for h in horizons:
        y -= 1
        s0 = base_surv[h]
        ax.plot([0, 100], [y, y], color="k", lw=0.8)
        for prob in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
            if not (0 < s0 < 1):
                break
            lp = np.log(np.log(prob) / np.log(s0))
            total = (lp - lp_min) * scale
            pos = total / total_max * 100
            if 0 <= pos <= 100:
                ax.plot([pos, pos], [y, y + 0.12], color="k", lw=0.6)
                ax.text(pos, y - 0.32, f"{prob:.1f}", ha="center", fontsize=6)
        ax.text(-3, y, f"{h:g} 个月生存率", ha="right", va="center", fontsize=8)
    ax.set_xlim(-30, 105)
    ax.set_ylim(y - 1, n_rows + 1)
    ax.set_title(title)
    return _save(fig, path)
