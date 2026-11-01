"""离散时间生存：时间分段与生存概率换算。

- 分段边界取训练部分**事件时间**的分位数（默认 4 段），第一段从 0 开始，最后一段到无穷。
- 每个终点、每个人群（手术 / 不可切除）各有一套边界，只用当前折训练部分计算。
- 生存概率 S(t)：段内按指数衰减插值（分段常数风险）。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def make_edges(times: np.ndarray, events: np.ndarray, n_bins: int) -> List[float]:
    t = np.asarray(times, float)
    e = np.asarray(events, float)
    ok = np.isfinite(t) & np.isfinite(e)
    t, e = t[ok], e[ok]
    src = t[e == 1] if (e == 1).sum() >= n_bins else t
    if len(src) == 0:
        src = np.array([1.0])
    qs = np.quantile(src, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    inner = []
    prev = 0.0
    for q in qs:
        q = max(float(q), prev + 1e-3)
        inner.append(q)
        prev = q
    return [0.0] + inner + [float("inf")]


def assign_bins(times: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    t = np.asarray(times, float)
    idx = np.searchsorted(np.asarray(edges[1:-1], float), t, side="right")
    idx = idx.astype(int)
    idx[~np.isfinite(t)] = -1
    return idx


def empirical_hazard_logits(bin_idx: np.ndarray, events: np.ndarray, n_bins: int) -> np.ndarray:
    """各段经验风险的 logit（用于初始化基线）。与损失一致：在第 j 段删失的人不计入第 j 段风险集。"""
    j = np.asarray(bin_idx)
    e = np.asarray(events, float)
    ok = j >= 0
    j, e = j[ok], e[ok]
    out = np.zeros(n_bins)
    for k in range(n_bins):
        at_risk = (j > k).sum() + ((j == k) & (e == 1)).sum()
        ev = ((j == k) & (e == 1)).sum()
        h = (ev + 0.5) / (at_risk + 1.0)
        out[k] = np.log(h / (1 - h))
    return out


def survival_at(surv: np.ndarray, edges: Sequence[float], horizons: Sequence[float]) -> np.ndarray:
    """surv [B, K]：各段结束时的生存概率 → 指定时间点（月）的生存概率 [B, H]。"""
    surv = np.asarray(surv, float)
    edges = list(edges)
    widths = [edges[k + 1] - edges[k] for k in range(len(edges) - 2)]
    last_w = widths[-1] if widths else 12.0
    out = np.zeros((surv.shape[0], len(horizons)))
    for hi, t in enumerate(horizons):
        k = int(np.searchsorted(np.asarray(edges[1:-1]), t, side="right"))
        s_start = surv[:, k - 1] if k > 0 else np.ones(surv.shape[0])
        s_end = surv[:, k]
        w = (edges[k + 1] - edges[k]) if np.isfinite(edges[k + 1]) else last_w
        # 有限时间段内 frac ≤ 1；最后一段（到无穷）按该段风险继续外推
        frac = max(0.0, (t - edges[k]) / w) if w > 0 else 1.0
        ratio = np.clip(s_end / np.maximum(s_start, 1e-12), 1e-12, 1.0)
        out[:, hi] = s_start * ratio ** frac
    return out


def fit_bins(times: Dict[str, np.ndarray], events: Dict[str, np.ndarray], pops: Dict[str, np.ndarray],
             endpoints: Sequence[str], n_pops: Dict[str, int], n_bins: int) -> Dict[str, List[List[float]]]:
    """返回 {终点: [人群0边界, 人群1边界, ...]}。"""
    out = {}
    for ep in endpoints:
        per = []
        for p in range(n_pops[ep]):
            m = pops[ep] == p
            per.append(make_edges(times[ep][m], events[ep][m], n_bins))
        out[ep] = per
    return out
