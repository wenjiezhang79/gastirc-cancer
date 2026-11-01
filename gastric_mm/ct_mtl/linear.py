"""影像组学基线用的 L1 正则化（LASSO）线性模型：纯 numpy 实现，不新增依赖。

- 二分类：L1 逻辑回归（正样本加权，同深度模型的 max_pos_weight 规则）
- 多分类：L1 多项逻辑回归（Lauren）
- 生存：L1 Cox 部分似然（Breslow 处理并列），可按人群分层（各人群各自的基线风险、共享系数）；
  基线累积风险用 Breslow 估计，输出各时间点生存概率
- 回归：L1 线性回归
求解：近端梯度（FISTA，回溯线搜索），截距不惩罚；λ 在正则化路径上由内层 K 折交叉验证选择（验证集损失最小）。
特征预处理（只用训练部分拟合）：剔除缺失过多/近似常数的特征 → 中位数填补 → z 分数 → 相关系数去冗余。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- 预处理 ----------------
class Preprocessor:
    def __init__(self, corr_threshold: float = 0.9, max_missing: float = 0.2):
        self.corr_threshold = float(corr_threshold)
        self.max_missing = float(max_missing)

    def fit(self, X: np.ndarray, names: Sequence[str]) -> "Preprocessor":
        X = np.asarray(X, float)
        miss = np.isnan(X).mean(0)
        med = np.nanmedian(np.where(np.isnan(X).all(0), 0.0, X), axis=0)
        Xf = np.where(np.isnan(X), med, X)
        sd = Xf.std(0)
        ok = (miss <= self.max_missing) & (sd > 1e-8 * (np.abs(Xf).mean(0) + 1e-12)) & (sd > 1e-12)
        idx = np.flatnonzero(ok)
        Z = (Xf[:, idx] - Xf[:, idx].mean(0)) / Xf[:, idx].std(0)
        keep: List[int] = []
        if len(idx):
            C = np.abs(np.corrcoef(Z, rowvar=False)) if len(idx) > 1 else np.ones((1, 1))
            for j in range(len(idx)):
                if all(C[j, k] <= self.corr_threshold for k in keep):
                    keep.append(j)
        sel = idx[keep]
        self.names = [names[i] for i in sel]
        self.cols = sel
        self.median = med[sel]
        self.mean = Xf[:, sel].mean(0)
        self.std = Xf[:, sel].std(0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)[:, self.cols]
        X = np.where(np.isnan(X), self.median, X)
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"names": self.names, "median": self.median.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict, all_names: Sequence[str]) -> "Preprocessor":
        p = cls()
        pos = {n: i for i, n in enumerate(all_names)}
        missing = [n for n in d["names"] if n not in pos]
        if missing:
            raise KeyError(f"特征表中缺少模型使用的特征：{missing[:5]}")
        p.names = list(d["names"])
        p.cols = np.array([pos[n] for n in p.names], dtype=int)
        p.median, p.mean, p.std = (np.array(d[k], float) for k in ("median", "mean", "std"))
        return p


# ---------------- 损失与梯度 ----------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def loss_grad(kind: str, X: np.ndarray, y, W: np.ndarray, b: np.ndarray, aux: dict) -> Tuple[float, np.ndarray, np.ndarray]:
    n = len(X)
    if kind == "binary":
        w = aux["w"]
        z = X @ W[:, 0] + b[0]
        p = _sigmoid(z)
        eps = 1e-12
        L = -np.sum(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / w.sum()
        r = w * (p - y) / w.sum()
        return L, (X.T @ r)[:, None], np.array([r.sum()])
    if kind == "multiclass":
        z = X @ W + b
        z = z - z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        Y = np.eye(W.shape[1])[y.astype(int)]
        L = -np.mean(np.log(P[np.arange(n), y.astype(int)] + 1e-12))
        R = (P - Y) / n
        return L, X.T @ R, R.sum(0)
    if kind == "regression":
        r = X @ W[:, 0] + b[0] - y
        return 0.5 * np.mean(r ** 2), (X.T @ r / n)[:, None], np.array([r.mean()])
    if kind == "survival":
        lp = X @ W[:, 0]
        L, g = _cox(lp, aux["time"], aux["event"], aux["strata"])
        return L, (X.T @ g)[:, None], np.zeros(1)
    raise ValueError(kind)


def _cox(lp, time, event, strata) -> Tuple[float, np.ndarray]:
    """分层 Cox 负部分对数似然（Breslow），除以总事件数；返回 (损失, 对 lp 的梯度)。"""
    tot, grad, n_ev = 0.0, np.zeros_like(lp), max(1.0, float(event.sum()))
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        if event[m].sum() == 0:
            continue
        order = m[np.argsort(-time[m], kind="mergesort")]
        t, e, l = time[order], event[order], lp[order]
        lmax = l.max()
        el = np.exp(l - lmax)
        cs = np.cumsum(el)
        # 并列时间：风险集包含所有时间 ≥ t 的人 → 取同一时间组的最后一个累计和
        last = np.searchsorted(-t, -t, side="right") - 1
        risk = cs[last]
        tot += -np.sum(e * (l - lmax - np.log(risk)))
        # 梯度：-e_i + exp(l_j) Σ_{i: t_i ≤ t_j, e_i} 1/risk_i
        inv = np.where(e > 0, 1.0 / risk, 0.0)
        # 对每个 j，求所有风险集包含 j（t_i ≤ t_j）的事件 i 的 1/risk_i 之和；按时间降序后 i 在 j 之后（含并列）
        acc = np.cumsum(inv[::-1])[::-1]
        first = np.searchsorted(-t, -t, side="left")
        g = -e + el * acc[first]
        grad[order] = g
    return tot / n_ev, grad / n_ev


def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def fit_l1(kind: str, X: np.ndarray, y, lam: float, aux: dict, n_out: int = 1, W0=None, b0=None,
           max_iter: int = 500, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    p = X.shape[1]
    W = np.zeros((p, n_out)) if W0 is None else W0.copy()
    b = np.zeros(n_out) if b0 is None else b0.copy()
    if b0 is None and kind == "binary":
        w = aux["w"]
        pm = np.clip(np.sum(w * y) / w.sum(), 1e-4, 1 - 1e-4)
        b[0] = np.log(pm / (1 - pm))
    elif b0 is None and kind == "multiclass":
        cnt = np.bincount(y.astype(int), minlength=n_out) + 1.0
        b = np.log(cnt / cnt.sum())
    elif b0 is None and kind == "regression":
        b[0] = float(np.mean(y))
    step = 1.0
    Wy, by, tk = W.copy(), b.copy(), 1.0
    prev = np.inf
    for _ in range(int(max_iter)):
        L, gW, gb = loss_grad(kind, X, y, Wy, by, aux)
        while True:
            Wn = _soft(Wy - step * gW, step * lam)
            bn = by - step * gb
            Ln = loss_grad(kind, X, y, Wn, bn, aux)[0]
            d = np.concatenate([(Wn - Wy).ravel(), bn - by])
            if Ln <= L + np.concatenate([gW.ravel(), gb]) @ d + 0.5 / step * (d @ d) + 1e-12 or step < 1e-8:
                break
            step *= 0.5
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Wy = Wn + (tk - 1) / tn * (Wn - W)
        by = bn + (tk - 1) / tn * (bn - b)
        W, b, tk = Wn, bn, tn
        obj = Ln + lam * np.abs(W).sum()
        if abs(prev - obj) < tol * max(1.0, abs(obj)):
            break
        prev = obj
    return W, b


def lambda_max(kind: str, X, y, aux: dict, n_out: int) -> float:
    W0, b0 = fit_l1(kind, X, y, 1e9, aux, n_out, max_iter=200)
    _, gW, _ = loss_grad(kind, X, y, np.zeros_like(W0), b0, aux)
    return float(np.abs(gW).max()) + 1e-12


def kfold_ids(n: int, k: int, rng: np.random.RandomState, strata: Optional[np.ndarray] = None) -> np.ndarray:
    ids = np.zeros(n, dtype=int)
    if strata is None:
        strata = np.zeros(n)
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        m = m[rng.permutation(len(m))]
        ids[m] = np.arange(len(m)) % k
    return ids


def subset_aux(aux: dict, idx) -> dict:
    return {k: v[idx] for k, v in aux.items()}


def fit_with_cv(kind: str, X, y, aux: dict, n_out: int, n_lambdas: int = 20, min_ratio: float = 0.01,
                inner_folds: int = 5, max_iter: int = 500, seed: int = 0) -> dict:
    """在 λ 路径上做内层 K 折交叉验证，选验证损失最小的 λ，再用全部训练数据重拟合。"""
    lmax = lambda_max(kind, X, y, aux, n_out)
    lams = lmax * np.logspace(0, np.log10(min_ratio), n_lambdas)
    rng = np.random.RandomState(seed)
    strata = y.astype(int) if kind in ("binary", "multiclass") else (aux["event"].astype(int) if kind == "survival" else None)
    k = int(min(inner_folds, max(2, np.bincount(strata).min() if strata is not None and len(np.unique(strata)) > 1 else inner_folds)))
    fold = kfold_ids(len(X), k, rng, strata)
    cv = np.zeros((k, len(lams)))
    for f in range(k):
        tr, va = fold != f, fold == f
        W, b = None, None
        for j, lam in enumerate(lams):
            W, b = fit_l1(kind, X[tr], y[tr], lam, subset_aux(aux, tr), n_out, W, b, max_iter)
            cv[f, j] = loss_grad(kind, X[va], y[va], W, b, subset_aux(aux, va))[0]
    mean = np.nanmean(cv, 0)
    j = int(np.nanargmin(mean))
    W, b = None, None
    for lam in lams[: j + 1]:
        W, b = fit_l1(kind, X, y, lam, aux, n_out, W, b, max_iter)
    return {"W": W, "b": b, "lambda": float(lams[j]), "lambda_index": j, "cv_loss": mean.tolist(),
            "n_nonzero": int((np.abs(W).sum(1) > 0).sum())}


def breslow(lp, time, event, strata) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """各层的基线累积风险：{层: (事件时间, H0)}。"""
    out = {}
    for s in np.unique(strata):
        m = strata == s
        t, e, el = time[m], event[m], np.exp(lp[m])
        ts = np.unique(t[e > 0])
        H = []
        acc = 0.0
        for u in ts:
            acc += float(e[(t == u)].sum()) / float(el[t >= u].sum())
            H.append(acc)
        out[int(s)] = (ts, np.array(H))
    return out


def survival_from_breslow(lp: np.ndarray, strata: np.ndarray, base: Dict[int, Tuple], horizons) -> np.ndarray:
    S = np.full((len(lp), len(horizons)), np.nan)
    for i, (l, s) in enumerate(zip(lp, strata)):
        if int(s) not in base:
            continue
        ts, H = base[int(s)]
        for j, h in enumerate(horizons):
            k = np.searchsorted(ts, h, side="right") - 1
            H0 = H[k] if k >= 0 else 0.0
            S[i, j] = np.exp(-H0 * np.exp(l))
    return S
