"""分类评价（纯 numpy / scipy）：DeLong 检验、校准（曲线、Brier、截距与斜率、截距重新校准）、决策曲线（二分类与生存）。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .survival import kaplan_meier

Z95 = 1.959963984540054


def _clean(y, *scores):
    y = np.asarray(y, float)
    ss = [np.asarray(s, float) for s in scores]
    ok = np.isfinite(y)
    for s in ss:
        ok &= np.isfinite(s)
    return y[ok], [s[ok] for s in ss]


# ---------------- DeLong ----------------
def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.r_[pos, neg])
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n              # 每个阳性的结构分量
    v01 = 1 - (tz[m:] - ty) / m          # 每个阴性的结构分量
    return auc, v10, v01


def delong_auc(y, score) -> Dict[str, float]:
    """单个模型的 AUC 与 DeLong 95% CI。"""
    y, (s,) = _clean(y, score)
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    auc, v10, v01 = _delong_components(y, s)
    var = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    se = float(np.sqrt(var))
    return {"auc": float(auc), "se": se, "lo": float(max(0.0, auc - Z95 * se)), "hi": float(min(1.0, auc + Z95 * se))}


def delong_test(y, score_a, score_b) -> Dict[str, float]:
    """同一批病例上两个模型 AUC 之差（a − b）的 DeLong 检验（配对）。"""
    y, (a, b) = _clean(y, score_a, score_b)
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return {k: float("nan") for k in ("auc_a", "auc_b", "diff", "se", "lo", "hi", "z", "p")}
    aa, v10a, v01a = _delong_components(y, a)
    ab, v10b, v01b = _delong_components(y, b)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / len(v10a) + s01 / len(v01a)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = aa - ab
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else float("nan")
    p = float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"auc_a": float(aa), "auc_b": float(ab), "diff": float(diff), "se": se, "lo": float(diff - Z95 * se),
            "hi": float(diff + Z95 * se), "z": float(z), "p": p, "n": int(len(y)), "n_pos": int((y == 1).sum())}


# ---------------- 校准 ----------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def _logistic_fit(X, y, offset=None, max_iter=100):
    """无惩罚逻辑回归（Newton），X 含截距列。返回系数。"""
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(X @ b + off)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-10 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def calibration(y, prob, n_bins: int = 10) -> Dict[str, object]:
    """校准：分箱曲线（按预测概率分位数）、Brier 分数、校准截距（斜率固定为 1 时的截距，calibration-in-the-large）与校准斜率。"""
    y, (p,) = _clean(y, prob)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "brier": float("nan"), "intercept": float("nan"), "slope": float("nan"), "curve": []}
    lp = _logit(p)
    slope = float(_logistic_fit(np.c_[np.ones(len(y)), lp], y)[1])
    intercept = float(_logistic_fit(np.ones((len(y), 1)), y, offset=lp)[0])
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    curve = []
    for k in range(len(q) - 1):
        m = idx == k
        if m.sum():
            obs = y[m].mean()
            se = np.sqrt(max(obs * (1 - obs), 1e-12) / m.sum())
            curve.append({"pred": float(p[m].mean()), "obs": float(obs), "lo": float(max(0, obs - Z95 * se)),
                          "hi": float(min(1, obs + Z95 * se)), "n": int(m.sum())})
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)), "intercept": intercept, "slope": slope,
            "mean_pred": float(p.mean()), "obs_rate": float(y.mean()), "curve": curve}


def recalibrate_intercept(y, prob) -> Dict[str, object]:
    """敏感性分析：只用该队列数据调整整体偏移（截距），返回调整量与调整后的校准指标。"""
    y, (p,) = _clean(y, prob)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"shift": float("nan"), "after": calibration(y, p)}
    a = float(_logistic_fit(np.ones((len(y), 1)), y, offset=_logit(p))[0])
    return {"shift": a, "after": calibration(y, _sigmoid(_logit(p) + a))}


def survival_calibration(time, event, surv_prob, t: float, n_bins: int = 5) -> list:
    """生存模型在 t 时刻的校准：按预测生存概率分组，各组 KM 实际生存率 vs 平均预测。"""
    T, E, S = (np.asarray(x, float) for x in (time, event, surv_prob))
    ok = np.isfinite(T) & np.isfinite(E) & np.isfinite(S)
    T, E, S = T[ok], E[ok], S[ok]
    if len(T) < 2 * n_bins:
        return []
    q = np.unique(np.quantile(S, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(q, S, side="right") - 1, 0, len(q) - 2)
    out = []
    for k in range(len(q) - 1):
        m = idx == k
        if m.sum() >= 2:
            km = kaplan_meier(T[m], E[m])
            lo, hi = km.ci_at(t)
            out.append({"pred": float(S[m].mean()), "obs": km.at(t), "lo": lo, "hi": hi, "n": int(m.sum())})
    return out


# ---------------- 决策曲线 ----------------
def decision_curve(y, prob, thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """二分类决策曲线：净获益 = 真阳性率 − 假阳性率 × pt / (1 − pt)；同时给出"全部按阳性处理""全部不处理"。"""
    y, (p,) = _clean(y, prob)
    th = np.asarray(thresholds if thresholds is not None else np.arange(0.01, 0.99, 0.01))
    n = len(y)
    prev = y.mean() if n else float("nan")
    rows = []
    for pt in th:
        pos = p >= pt
        tp = (pos & (y == 1)).sum() / n
        fp = (pos & (y == 0)).sum() / n
        odds = pt / (1 - pt)
        rows.append({"threshold": float(pt), "model": float(tp - fp * odds), "treat_all": float(prev - (1 - prev) * odds),
                     "treat_none": 0.0})
    return pd.DataFrame(rows)


def decision_curve_survival(time, event, risk_at_t, t: float, thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """生存决策曲线（t 时刻）：risk_at_t 为预测的 t 时刻前发生事件的概率；按 KM 估计各决策下的实际事件率。"""
    T, E, R = (np.asarray(x, float) for x in (time, event, risk_at_t))
    ok = np.isfinite(T) & np.isfinite(E) & np.isfinite(R)
    T, E, R = T[ok], E[ok], R[ok]
    th = np.asarray(thresholds if thresholds is not None else np.arange(0.01, 0.99, 0.01))
    n = len(T)
    ev_all = 1 - kaplan_meier(T, E).at(t) if n else float("nan")
    rows = []
    for pt in th:
        pos = R >= pt
        odds = pt / (1 - pt)
        if pos.sum() >= 1:
            ev = 1 - kaplan_meier(T[pos], E[pos]).at(t)
            frac = pos.mean()
            nb = ev * frac - (1 - ev) * frac * odds
        else:
            nb = 0.0
        rows.append({"threshold": float(pt), "model": float(nb), "treat_all": float(ev_all - (1 - ev_all) * odds),
                     "treat_none": 0.0})
    return pd.DataFrame(rows)
