"""评估指标（纯 numpy / scipy 实现，不依赖 scikit-learn）。

- 二分类：AUC（Mann–Whitney，含并列）、PR-AUC（平均精确率）、敏感度/特异度、约登阈值
- 多分类：一对其余的各类 AUC 与宏平均 AUC、平衡准确率
- 生存：Harrell C-index（风险越高 = 生存越短）
- 回归：Spearman 相关
- bootstrap 百分位置信区间、配对 bootstrap 差值

缺失值（NaN）在计算前自动去掉；无法计算时（如只有一类）返回 NaN。
v4 在此基础上扩展 DeLong、时间依赖 AUC、校准、决策曲线。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr


def _clean(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        ok &= np.all(np.isfinite(a.reshape(len(a), -1)), axis=1)
    return [a[ok] for a in arrs]


def auc_binary(y, score) -> float:
    y, s = _clean(y, score)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y, score) -> float:
    """平均精确率（average precision）。"""
    y, s = _clean(y, score)
    n1 = int((y == 1).sum())
    if n1 == 0 or n1 == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    # 按分数阈值分组（并列分数一起处理）
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n1
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def sens_spec(y, score, threshold: float) -> Tuple[float, float]:
    y, s = _clean(y, score)
    pred = s >= threshold
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    sens = float((pred & (y == 1)).sum() / n1) if n1 else float("nan")
    spec = float((~pred & (y == 0)).sum() / n0) if n0 else float("nan")
    return sens, spec


def youden_threshold(y, score) -> float:
    """约登指数（敏感度 + 特异度 − 1）最大的阈值。"""
    y, s = _clean(y, score)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    best, thr = -1.0, float(np.median(s))
    for t in np.unique(s):
        se, sp = sens_spec(y, s, t)
        if se + sp - 1 > best:
            best, thr = se + sp - 1, float(t)
    return thr


def multiclass_auc(y, prob) -> Tuple[float, list]:
    """一对其余：返回 (宏平均 AUC, 各类 AUC)。prob 形状 [n, C]。"""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(prob), axis=1)
    y, prob = y[ok], prob[ok]
    per = [auc_binary((y == c).astype(float), prob[:, c]) for c in range(prob.shape[1])]
    valid = [a for a in per if np.isfinite(a)]
    return (float(np.mean(valid)) if valid else float("nan")), per


def balanced_accuracy(y, prob) -> float:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(prob), axis=1)
    y, pred = y[ok].astype(int), prob[ok].argmax(1)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else float("nan")


def c_index(time, event, risk) -> float:
    """Harrell C-index：可比较对为 t_i < t_j 且 i 发生事件；风险高者先发生事件记 1，风险相同记 0.5。"""
    t, e, r = _clean(time, event, risk)
    if (e == 1).sum() == 0 or len(t) < 2:
        return float("nan")
    num, den = 0.0, 0.0
    idx = np.flatnonzero(e == 1)
    for start in range(0, len(idx), 512):
        ii = idx[start:start + 512]
        comp = t[None, :] > t[ii, None]                   # j 的时间更长
        dr = r[ii, None] - r[None, :]
        num += float(((dr > 0) & comp).sum() + 0.5 * ((dr == 0) & comp).sum())
        den += float(comp.sum())
    return num / den if den > 0 else float("nan")


def spearman(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ---------------- bootstrap ----------------
def bootstrap_indices(n: int, n_boot: int, seed: int = 0, strata: Optional[np.ndarray] = None):
    """生成 bootstrap 抽样下标；给出 strata 时在各层内分别有放回抽样（保持阳性比例）。"""
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        if strata is None:
            yield rng.randint(0, n, n)
        else:
            parts = []
            for s in np.unique(strata):
                m = np.flatnonzero(strata == s)
                parts.append(rng.choice(m, size=len(m), replace=True))
            yield np.concatenate(parts)


def bootstrap_ci(fn: Callable, arrays: Sequence[np.ndarray], n_boot: int = 1000, seed: int = 0,
                 strata: Optional[np.ndarray] = None, alpha: float = 0.05) -> Tuple[float, float, float]:
    """返回 (点估计, 下限, 上限)；fn(*arrays) 返回标量。"""
    arrays = [np.asarray(a) for a in arrays]
    point = fn(*arrays)
    vals = []
    for idx in bootstrap_indices(len(arrays[0]), n_boot, seed, strata):
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_diff(fn: Callable, arrays_a: Sequence[np.ndarray], arrays_b: Sequence[np.ndarray],
                          n_boot: int = 1000, seed: int = 0, strata: Optional[np.ndarray] = None,
                          alpha: float = 0.05) -> Dict[str, float]:
    """同一批病例上两个模型指标之差 (a − b) 的配对 bootstrap 置信区间与单侧比例。"""
    arrays_a = [np.asarray(a) for a in arrays_a]
    arrays_b = [np.asarray(a) for a in arrays_b]
    d0 = fn(*arrays_a) - fn(*arrays_b)
    diffs = []
    for idx in bootstrap_indices(len(arrays_a[0]), n_boot, seed, strata):
        d = fn(*[a[idx] for a in arrays_a]) - fn(*[a[idx] for a in arrays_b])
        if np.isfinite(d):
            diffs.append(d)
    if not diffs:
        return {"diff": float(d0), "lo": float("nan"), "hi": float("nan"), "p_le_0": float("nan")}
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(d0), "lo": float(lo), "hi": float(hi), "p_le_0": float(np.mean(np.array(diffs) <= 0))}
