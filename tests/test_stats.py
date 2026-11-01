"""v4 统计工具的数值核对：用教科书数据与可手算的例子检验自己实现的方法。

- Freireich（1963）6-MP 白血病试验（Gehan 数据，Kleinbaum《Survival Analysis》经典例子）：
  KM、log-rank χ² = 16.79、Cox 系数 Efron −1.572（SE 0.412）、Breslow −1.509（SE 0.410）。
"""

import numpy as np
import pandas as pd
import pytest

from gastric_mm.stats import classification as C
from gastric_mm.stats import survival as S

MP = [(6, 1), (6, 1), (6, 1), (6, 0), (7, 1), (9, 0), (10, 1), (10, 0), (11, 0), (13, 1), (16, 1), (17, 0), (19, 0), (20, 0),
      (22, 1), (23, 1), (25, 0), (32, 0), (32, 0), (34, 0), (35, 0)]
PL = [(t, 1) for t in [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]]


@pytest.fixture(scope="module")
def leuk():
    t = np.array([a for a, _ in MP + PL], float)
    e = np.array([b for _, b in MP + PL], float)
    g = np.r_[np.ones(21), np.zeros(21)]
    return t, e, g


def test_kaplan_meier_textbook(leuk):
    t, e, g = leuk
    km = S.kaplan_meier(t[g == 1], e[g == 1])
    assert [round(km.at(x), 3) for x in (6, 7, 10, 13, 16, 22, 23)] == [0.857, 0.807, 0.753, 0.690, 0.627, 0.538, 0.448]
    assert S.kaplan_meier(t[g == 0], e[g == 0]).median() == 8.0
    lo, hi = km.ci_at(10)
    assert lo < km.at(10) < hi
    assert S.n_at_risk(t[g == 1], [0, 10, 20]) == [21, 15, 8]


def test_logrank_textbook(leuk):
    t, e, g = leuk
    r = S.logrank(t, e, g)
    assert abs(r["chi2"] - 16.79) < 0.01 and r["df"] == 1 and r["p"] < 1e-4
    r3 = S.logrank(t, e, np.r_[np.zeros(14), np.ones(14), 2 * np.ones(14)])
    assert r3["df"] == 2 and np.isfinite(r3["chi2"])


def test_cox_textbook(leuk):
    t, e, g = leuk
    ef = S.cox_fit(g, t, e, ["6mp"], ties="efron")
    br = S.cox_fit(g, t, e, ["6mp"], ties="breslow")
    assert abs(ef.beta[0] + 1.572) < 1e-3 and abs(ef.se[0] - 0.412) < 1e-3
    assert abs(br.beta[0] + 1.509) < 1e-3 and abs(br.se[0] - 0.410) < 1e-3
    tab = ef.table()
    assert abs(tab.hr[0] - np.exp(-1.572)) < 1e-3 and tab.p[0] < 0.001
    assert ef.lrt()["p"] < 0.001
    # 无权重时稳健标准误与模型标准误同一量级
    rb = S.cox_fit(g, t, e, ["6mp"], weights=np.ones(len(t)))
    assert abs(rb.beta[0] - ef.beta[0]) < 1e-6 and 0.6 < rb.se[0] / ef.se[0] < 1.4
    ph = S.cox_ph_test(ef)
    assert ph.p[0] > 0.05                                   # 该数据满足比例风险假设


def test_cox_recovers_known_effects_and_strata():
    rng = np.random.RandomState(0)
    n = 3000
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n)
    stra = rng.randint(0, 3, n)
    base = np.array([0.05, 0.1, 0.2])[stra]
    T = rng.exponential(1 / (base * np.exp(0.7 * x1 - 0.5 * x2)))
    Cn = rng.uniform(0, 30, n)
    t, e = np.minimum(T, Cn), (T <= Cn).astype(float)
    r = S.cox_fit(np.c_[x1, x2], t, e, ["x1", "x2"], strata=stra)
    assert abs(r.beta[0] - 0.7) < 0.08 and abs(r.beta[1] + 0.5) < 0.12
    assert r.wald([0, 1])["p"] < 1e-10
    base_h = S.breslow_baseline(r)
    surv = S.predict_survival(np.zeros(1), base_h[0], [5.0])[0, 0]
    assert abs(surv - np.exp(-0.05 * 5)) < 0.05             # 第 0 层基线生存与真实值接近
    # 中位随访（反向 KM）大于 0 且不超过最长随访
    mf = S.median_followup(t, e)
    assert 0 < mf <= t.max()


def test_time_dependent_auc():
    rng = np.random.RandomState(1)
    n = 800
    risk = rng.normal(size=n)
    T = rng.exponential(1 / (0.1 * np.exp(risk)))
    # 无删失时，t 时刻的累积/动态 AUC 等于"t 前死亡 vs t 后存活"的普通 AUC
    from gastric_mm.metrics import auc_binary
    t0 = float(np.median(T))
    a = S.time_dependent_auc(T, np.ones(n), risk, t0)
    assert abs(a - auc_binary((T <= t0).astype(float), risk)) < 1e-9
    Cn = rng.uniform(0, 30, n)
    a2 = S.time_dependent_auc(np.minimum(T, Cn), (T <= Cn).astype(float), risk, t0)
    assert abs(a2 - a) < 0.06                               # 有删失时 IPCW 估计仍接近


def test_delong():
    rng = np.random.RandomState(2)
    n = 400
    y = rng.binomial(1, 0.3, n).astype(float)
    a = y * 1.0 + rng.normal(size=n)
    b = y * 0.5 + rng.normal(size=n)
    from gastric_mm.metrics import auc_binary
    r = C.delong_test(y, a, b)
    assert abs(r["auc_a"] - auc_binary(y, a)) < 1e-12 and abs(r["auc_b"] - auc_binary(y, b)) < 1e-12
    # 与配对 bootstrap 标准误一致
    diffs = []
    for _ in range(600):
        i = rng.randint(0, n, n)
        diffs.append(auc_binary(y[i], a[i]) - auc_binary(y[i], b[i]))
    assert abs(r["se"] / np.std(diffs) - 1) < 0.15
    assert r["p"] < 0.05
    one = C.delong_auc(y, a)
    assert one["lo"] < one["auc"] < one["hi"]
    # 两个相同模型：差为 0
    same = C.delong_test(y, a, a)
    assert abs(same["diff"]) < 1e-12
    # 已发表的小例子（Hanley & McNeil 风格，平局用中秩）：AUC 手算
    yy = np.array([1, 1, 1, 0, 0, 0, 0], float)
    ss = np.array([0.9, 0.8, 0.4, 0.4, 0.3, 0.2, 0.1])
    assert abs(C.delong_auc(yy, ss)["auc"] - (11.5 / 12)) < 1e-12


def test_calibration_and_recalibration():
    rng = np.random.RandomState(3)
    n = 5000
    lp = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-lp))
    y = (rng.rand(n) < p_true).astype(float)
    good = C.calibration(y, p_true)
    assert abs(good["slope"] - 1) < 0.1 and abs(good["intercept"]) < 0.1
    over = 1 / (1 + np.exp(-(lp + 1.0)))                    # 整体偏高
    bad = C.calibration(y, over)
    assert bad["intercept"] < -0.8
    fix = C.recalibrate_intercept(y, over)
    assert abs(fix["shift"] + 1.0) < 0.1 and abs(fix["after"]["intercept"]) < 0.05
    assert fix["after"]["brier"] < bad["brier"]
    assert len(good["curve"]) == 10


def test_decision_curve():
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0], float)
    p = np.array([0.9, 0.7, 0.6, 0.2, 0.1, 0.3, 0.2, 0.1, 0.05, 0.4])
    d = C.decision_curve(y, p, [0.25, 0.5])
    # pt = 0.5：阳性 = {0.9, 0.7, 0.6}，TP = 2，FP = 1 → 2/10 − 1/10 × 1 = 0.1
    assert abs(float(d.loc[d.threshold == 0.5, "model"].iloc[0]) - 0.1) < 1e-12
    # 全部处理：prev − (1 − prev) × odds；pt = 0.25 → 0.3 − 0.7/3
    assert abs(float(d.loc[d.threshold == 0.25, "treat_all"].iloc[0]) - (0.3 - 0.7 / 3)) < 1e-12
    rng = np.random.RandomState(4)
    T = rng.exponential(10, 300)
    ds = C.decision_curve_survival(T, np.ones(300), rng.rand(300), 10.0, [0.2, 0.5])
    assert list(ds.columns) == ["threshold", "model", "treat_all", "treat_none"]


def test_plots_smoke(tmp_path, leuk):
    from gastric_mm.stats import plots as P
    t, e, g = leuk
    out = P.km_plot({"6-MP": (t[g == 1], e[g == 1]), "安慰剂": (t[g == 0], e[g == 0])}, tmp_path / "km", "KM",
                    risk_times=[0, 10, 20, 30], annotation="log-rank p<0.001")
    assert all(p.exists() for p in out)
    rng = np.random.RandomState(5)
    y = rng.binomial(1, 0.3, 200).astype(float)
    s = y + rng.normal(size=200)
    P.roc_plot([("A", y, s), ("B", y, rng.normal(size=200))], tmp_path / "roc")
    P.forest_plot([{"label": "中心 1", "hr": 1.8, "lo": 1.2, "hi": 2.7, "n": 300},
                   {"label": "中心 2", "hr": float("nan"), "lo": float("nan"), "hi": float("nan"), "note": "事件不足"}],
                  tmp_path / "forest")
    P.calibration_plot([("A", C.calibration(y, 1 / (1 + np.exp(-s)))["curve"])], tmp_path / "cal")
    P.dca_plot({"A": C.decision_curve(y, 1 / (1 + np.exp(-s)))}, tmp_path / "dca")
    P.nomogram_plot([{"name": "风险评分", "coef": 0.8, "values": [("-2", -2), ("0", 0), ("2", 2)]},
                     {"name": "分期", "coef": 0.6, "values": [("I", 0), ("II", 1), ("III", 2)]}],
                    [12, 36], {12: 0.9, 36: 0.7}, tmp_path / "nomo")
    assert len(list(tmp_path.glob("*.png"))) == 6
