"""生存分析（纯 numpy / scipy）：KM、log-rank、反向 KM 中位随访、Cox 比例风险模型、Breslow 基线、时间依赖 AUC。

数值在 tests/test_stats.py 中用教科书数据核对（Freireich 6-MP 白血病试验：log-rank χ² ≈ 16.8，Efron Cox 系数 ≈ −1.572）。
时间单位与输入一致（本项目为月）。缺失值（NaN）的病例由调用方先去掉；本模块函数会对 NaN 报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


def _arr(*xs):
    out = [np.asarray(x, dtype=float) for x in xs]
    for a in out:
        if not np.all(np.isfinite(a)):
            raise ValueError("生存分析输入中有缺失值（NaN），请先去掉缺失的病例")
    return out


# ---------------- Kaplan–Meier ----------------
@dataclass
class KM:
    time: np.ndarray        # 各事件时间（升序，含 0）
    surv: np.ndarray        # 该时间点之后的生存率
    lo: np.ndarray          # 95% CI（log-log）
    hi: np.ndarray
    n_risk: np.ndarray
    n_event: np.ndarray
    n: int

    def at(self, t: float) -> float:
        """t 时刻的生存率（右连续阶梯函数）。"""
        i = np.searchsorted(self.time, t, side="right") - 1
        return float(self.surv[max(i, 0)])

    def ci_at(self, t: float):
        i = max(np.searchsorted(self.time, t, side="right") - 1, 0)
        return float(self.lo[i]), float(self.hi[i])

    def median(self) -> float:
        below = np.flatnonzero(self.surv <= 0.5)
        return float(self.time[below[0]]) if len(below) else float("nan")


def kaplan_meier(time, event) -> KM:
    t, e = _arr(time, event)
    ut = np.unique(t[e > 0])
    n_risk = np.array([(t >= u).sum() for u in ut], dtype=float)
    d = np.array([((t == u) & (e > 0)).sum() for u in ut], dtype=float)
    s = np.cumprod(1 - d / n_risk) if len(ut) else np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(d / (n_risk * (n_risk - d)))
        # log(-log S) 变换的置信区间
        se_ll = np.sqrt(gw) / np.abs(np.log(s))
        z = 1.959963984540054
        lo = s ** np.exp(z * se_ll)
        hi = s ** np.exp(-z * se_ll)
    lo = np.where(np.isfinite(lo), lo, np.where(s > 0, s, 0.0))
    hi = np.where(np.isfinite(hi), hi, np.where(s > 0, s, 0.0))
    return KM(time=np.r_[0.0, ut], surv=np.r_[1.0, s], lo=np.r_[1.0, lo], hi=np.r_[1.0, hi],
              n_risk=np.r_[len(t), n_risk], n_event=np.r_[0, d], n=len(t))


def n_at_risk(time, times: Sequence[float]) -> List[int]:
    t = np.asarray(time, float)
    return [int((t >= u).sum()) for u in times]


def median_followup(time, event) -> float:
    """反向 KM 法的中位随访时间（把删失当作"事件"）。"""
    t, e = _arr(time, event)
    return kaplan_meier(t, 1 - e).median()


# ---------------- log-rank ----------------
def logrank(time, event, group) -> Dict[str, float]:
    """k 组 log-rank 检验。返回 chi2、df、p。"""
    t, e = _arr(time, event)
    g = np.asarray(group)
    levels = [x for x in pd.unique(g)]
    k = len(levels)
    if k < 2:
        return {"chi2": float("nan"), "df": 0, "p": float("nan")}
    G = np.stack([(g == lv).astype(float) for lv in levels], 1)
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for u in np.unique(t[e > 0]):
        at = t >= u
        n = at.sum()
        d = ((t == u) & (e > 0)).sum()
        nj = G[at].sum(0)
        dj = G[(t == u) & (e > 0)].sum(0)
        O += dj
        E += d * nj / n
        if n > 1:
            f = d * (n - d) / (n * n * (n - 1))
            V += f * (np.diag(nj) * n - np.outer(nj, nj))
    oe = (O - E)[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2 = float(oe @ np.linalg.solve(Vr, oe))
    except np.linalg.LinAlgError:
        chi2 = float("nan")
    return {"chi2": chi2, "df": k - 1, "p": float(sps.chi2.sf(chi2, k - 1)) if np.isfinite(chi2) else float("nan"),
            "observed": dict(zip(map(str, levels), O.tolist())), "expected": dict(zip(map(str, levels), E.tolist()))}


# ---------------- Cox 比例风险模型 ----------------
@dataclass
class CoxResult:
    names: List[str]
    beta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    loglik0: float
    n: int
    n_events: int
    converged: bool
    robust: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def hr(self):
        return np.exp(self.beta)

    def table(self) -> pd.DataFrame:
        z = self.beta / self.se
        return pd.DataFrame({"term": self.names, "coef": self.beta, "se": self.se, "hr": np.exp(self.beta),
                             "hr_lo": np.exp(self.beta - 1.959963984540054 * self.se),
                             "hr_hi": np.exp(self.beta + 1.959963984540054 * self.se),
                             "z": z, "p": 2 * sps.norm.sf(np.abs(z))})

    def lrt(self) -> Dict[str, float]:
        chi2 = 2 * (self.loglik - self.loglik0)
        return {"chi2": float(chi2), "df": len(self.beta), "p": float(sps.chi2.sf(chi2, len(self.beta)))}

    def wald(self, idx: Sequence[int]) -> Dict[str, float]:
        """若干系数同时为 0 的 Wald 检验（如交互作用的多个项）。"""
        idx = list(idx)
        b = self.beta[idx]
        V = self.cov[np.ix_(idx, idx)]
        chi2 = float(b @ np.linalg.solve(V, b))
        return {"chi2": chi2, "df": len(idx), "p": float(sps.chi2.sf(chi2, len(idx)))}


def _cox_terms(beta, X, t, e, w, strata, ties):
    """返回 对数偏似然、梯度、Hessian（Efron / Breslow 并列处理，按层累加）。"""
    p = X.shape[1]
    ll, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    eta = X @ beta
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        order = m[np.argsort(-t[m], kind="mergesort")]
        ts, es, Xs, ws, eta_s = t[order], e[order], X[order], w[order], eta[order]
        c = eta_s.max() if len(eta_s) else 0.0
        r = ws * np.exp(eta_s - c)
        S0 = np.cumsum(r)
        S1 = np.cumsum(r[:, None] * Xs, 0)
        S2 = np.cumsum(r[:, None, None] * Xs[:, :, None] * Xs[:, None, :], 0)
        uniq = np.unique(ts[es > 0])
        for u in uniq:
            last = np.searchsorted(-ts, -u, side="right") - 1      # 风险集：时间 ≥ u
            dmask = (ts == u) & (es > 0)
            d_w = ws[dmask]
            dw = d_w.sum()
            nd = int(dmask.sum())
            xd = (d_w[:, None] * Xs[dmask]).sum(0)
            ll += float((d_w * (eta_s[dmask] - c)).sum())
            grad += xd
            if ties == "efron" and nd > 1:
                rd = r[dmask]
                D0, D1 = rd.sum(), (rd[:, None] * Xs[dmask]).sum(0)
                D2 = (rd[:, None, None] * Xs[dmask][:, :, None] * Xs[dmask][:, None, :]).sum(0)
                mean_w = dw / nd
                for j in range(nd):
                    f = j / nd
                    a0 = S0[last] - f * D0
                    a1 = S1[last] - f * D1
                    a2 = S2[last] - f * D2
                    ll -= mean_w * np.log(a0)
                    grad -= mean_w * a1 / a0
                    hess -= mean_w * (a2 / a0 - np.outer(a1, a1) / a0 ** 2)
            else:
                a0, a1, a2 = S0[last], S1[last], S2[last]
                ll -= dw * np.log(a0)
                grad -= dw * a1 / a0
                hess -= dw * (a2 / a0 - np.outer(a1, a1) / a0 ** 2)
    return ll, grad, hess


def cox_fit(X, time, event, names: Optional[Sequence[str]] = None, strata=None, weights=None, ties: str = "efron",
            robust: Optional[bool] = None, max_iter: int = 50, tol: float = 1e-9) -> CoxResult:
    """Cox 比例风险模型（Newton–Raphson + 步长减半）。

    strata：分层变量（各层各自的基线风险，如按中心分层）；weights：病例权重（如逆概率加权）；
    robust：是否用稳健（三明治）方差，默认有权重时用。
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    t, e = _arr(time, event)
    if not np.all(np.isfinite(X)):
        raise ValueError("Cox 回归的协变量中有缺失值，请先去掉缺失病例")
    n, p = X.shape
    names = list(names) if names is not None else [f"x{i}" for i in range(p)]
    strata = np.zeros(n, int) if strata is None else pd.factorize(np.asarray(strata))[0]
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    robust = (weights is not None) if robust is None else robust
    beta = np.zeros(p)
    ll0, _, _ = _cox_terms(beta, X, t, e, w, strata, ties)
    ll = ll0
    converged = False
    for _ in range(max_iter):
        ll_b, g, H = _cox_terms(beta, X, t, e, w, strata, ties)
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-H, g, rcond=None)[0]
        lam = 1.0
        while True:
            nb = beta + lam * step
            ll_n = _cox_terms(nb, X, t, e, w, strata, ties)[0]
            if ll_n >= ll_b - 1e-12 or lam < 1e-6:
                break
            lam /= 2
        beta = nb
        if abs(ll_n - ll_b) < tol * (1 + abs(ll_b)):
            ll = ll_n
            converged = True
            break
        ll = ll_n
    _, g, H = _cox_terms(beta, X, t, e, w, strata, ties)
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-H)
    if robust:
        U = cox_score_residuals(beta, X, t, e, w, strata)
        cov = cov @ (U.T @ U) @ cov
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return CoxResult(names=names, beta=beta, se=se, cov=cov, loglik=float(ll), loglik0=float(ll0), n=n,
                     n_events=int(e.sum()), converged=converged, robust=bool(robust),
                     extra={"strata": strata, "X": X, "time": t, "event": e, "weights": w, "ties": ties})


def cox_score_residuals(beta, X, t, e, w, strata) -> np.ndarray:
    """得分残差（Breslow 近似，Lin & Wei 1989），用于稳健（三明治）方差。返回 w_i × U_i，形状 [n, p]：

    U_i = δ_i (x_i − x̄(t_i)) − Σ_{事件 j: t_j ≤ t_i} w_j exp(η_i) / S0(t_j) × (x_i − x̄(t_j))，
    其中 S0(t) = Σ_{t_k ≥ t} w_k exp(η_k)，x̄(t) 为风险集的加权均值。
    """
    n, p = X.shape
    U = np.zeros((n, p))
    eta = X @ beta
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        ex = np.exp(eta[m] - eta[m].max())
        tm, em, Xm, wm = t[m], e[m], X[m], w[m]
        ev = np.flatnonzero(em > 0)
        S0 = np.array([(wm * ex)[tm >= tm[j]].sum() for j in ev])
        xbar = np.array([((wm * ex)[tm >= tm[j], None] * Xm[tm >= tm[j]]).sum(0) for j in ev]) / S0[:, None] \
            if len(ev) else np.zeros((0, p))
        Ui = np.zeros((len(m), p))
        for k, j in enumerate(ev):
            Ui[j] += Xm[j] - xbar[k]
            at = tm >= tm[j]
            Ui[at] -= (wm[j] * ex[at] / S0[k])[:, None] * (Xm[at] - xbar[k])
        U[m] = wm[:, None] * Ui
    return U


def cox_ph_test(res: CoxResult) -> pd.DataFrame:
    """比例风险假设检验：标度 Schoenfeld 残差对时间秩的回归斜率是否为 0（近似 Grambsch–Therneau / cox.zph 单变量检验）。"""
    X, t, e, w, strata = (res.extra[k] for k in ("X", "time", "event", "weights", "strata"))
    beta = res.beta
    eta = X @ beta
    rows_t, rows_r = [], []
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        for i in m[e[m] > 0]:
            at = m[t[m] >= t[i]]
            r = w[at] * np.exp(eta[at] - eta[at].max())
            xbar = (r[:, None] * X[at]).sum(0) / r.sum()
            rows_t.append(t[i])
            rows_r.append(X[i] - xbar)
    if len(rows_r) < 3:
        return pd.DataFrame({"term": res.names, "rho": np.nan, "chi2": np.nan, "p": np.nan})
    R = np.array(rows_r)
    d = len(R)
    scaled = beta + d * R @ res.cov
    g = sps.rankdata(np.array(rows_t))
    out = []
    for j, nm in enumerate(res.names):
        y = scaled[:, j]
        if np.std(y) == 0:
            out.append({"term": nm, "rho": np.nan, "chi2": np.nan, "p": np.nan})
            continue
        rho, p = sps.pearsonr(g, y)
        chi2 = rho ** 2 * (d - 2) / max(1e-12, 1 - rho ** 2)
        out.append({"term": nm, "rho": float(rho), "chi2": float(chi2), "p": float(p)})
    return pd.DataFrame(out)


def breslow_baseline(res: CoxResult) -> Dict[int, tuple]:
    """各层基线累积风险 {层: (事件时间, H0)}。"""
    X, t, e, w, strata = (res.extra[k] for k in ("X", "time", "event", "weights", "strata"))
    r = np.exp(X @ res.beta)
    out = {}
    for s in np.unique(strata):
        m = strata == s
        ts = np.unique(t[m & (e > 0)])
        H, acc = [], 0.0
        for u in ts:
            d = (w[m & (t == u) & (e > 0)]).sum()
            acc += d / (w[m & (t >= u)] * r[m & (t >= u)]).sum()
            H.append(acc)
        out[int(s)] = (ts, np.array(H))
    return out


def predict_survival(lp: np.ndarray, base: tuple, horizons: Sequence[float]) -> np.ndarray:
    """给定线性预测值与某层基线累积风险，返回各时间点的生存概率 [n, len(horizons)]。"""
    ts, H = base
    out = np.zeros((len(lp), len(horizons)))
    for j, h in enumerate(horizons):
        i = np.searchsorted(ts, h, side="right") - 1
        H0 = H[i] if i >= 0 else 0.0
        out[:, j] = np.exp(-H0 * np.exp(lp))
    return out


# ---------------- 时间依赖 AUC（累积 / 动态，IPCW） ----------------
def time_dependent_auc(time, event, risk, t: float) -> float:
    """t 时刻的累积 / 动态 AUC（Uno 等 2007，逆删失概率加权）：病例 = t 前发生事件者，对照 = t 时仍在随访者。"""
    T, E, R = _arr(time, event, risk)
    cens = kaplan_meier(T, 1 - E)

    def G(x):  # 删失生存函数在 x 之前（左极限）
        i = np.searchsorted(cens.time, x, side="left") - 1
        return float(cens.surv[max(i, 0)])

    case = (T <= t) & (E > 0)
    ctrl = T > t
    if case.sum() == 0 or ctrl.sum() == 0:
        return float("nan")
    wc = np.array([1.0 / max(G(x), 1e-12) for x in T[case]])
    rc, rk = R[case], R[ctrl]
    order = np.sort(rk)
    gt = np.searchsorted(order, rc, side="left")                  # 对照中风险 < 病例风险的个数
    eq = np.searchsorted(order, rc, side="right") - gt
    num = (wc * (gt + 0.5 * eq)).sum()
    return float(num / (wc.sum() * ctrl.sum()))
