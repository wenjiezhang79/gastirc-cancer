"""v4 探索：免疫治疗亚组（docs/decisions.md 第十五节第 7 小节；结论只写"提示"）。

按 ``immuno_setting`` 分场景分析，不混合：
| 场景 | 免疫组 | 对照组 | 结局 |
|---|---|---|---|
| neoadj（新辅助 / 围手术期） | 新辅助方案 = 化疗 + 免疫 | 新辅助方案 = 化疗 | OS；另看 pCR |
| adjuvant（术后辅助） | 非新辅助的手术患者中接受辅助免疫者 | 其余非新辅助手术患者 | OS |
| first_line（晚期一线） | 一线 = 化疗 + 免疫 | 一线 = 单纯化疗 | OS |
| later_line（后线） | 后线免疫者 | 无可比对照 | 只描述 |

每个有对照的场景：逻辑回归倾向性评分（协变量 v4.immuno.ps_covariates + 中心）→ 稳定化逆概率权重（按分位数截断）→
报告加权前后的标准化均数差；加权 Cox（稳健标准误）估计免疫治疗 HR，以及"模型预测的 MSI / EBV / PD-L1（每 10%）、风险评分 × 免疫"的交互作用。
全部队列合并（内部训练集折外 + 内部测试集 + 外部），场景内事件 < min_events 时只描述。

命令行：
    python -m gastric_mm.evaluation.immuno --config configs/server.yaml

输出：<output_dir>/v4/immuno/<时间>/report.md、results.json、figs/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..stats import plots as P
from ..stats import survival as S
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from ..stats.plots import tr as ftr
from .common import COHORT_NAMES, design, fmt, fmt_p, ft, setup_figures, surv_arrays, v4, write_json
from .table import build

SETTING_NAMES = {"neoadj": "新辅助 / 围手术期", "adjuvant": "术后辅助", "first_line": "晚期一线", "later_line": "晚期后线"}
MARKERS = {"msi_h_prob": "预测 MSI-H", "ebv_pos_prob": "预测 EBV", "pdl1_cps5_prob": "预测 PD-L1 CPS≥5", "risk_z": "风险评分"}
SETTING_EN = {"neoadj": "Neoadjuvant / perioperative", "adjuvant": "Adjuvant", "first_line": "Advanced first line",
              "later_line": "Advanced later line"}
MARKER_EN = {"msi_h_prob": "predicted MSI-H", "ebv_pos_prob": "predicted EBV", "pdl1_cps5_prob": "predicted PD-L1 CPS≥5",
             "risk_z": "risk score"}


def setting_groups(df: pd.DataFrame):
    """返回 {场景: (子表, 是否免疫 0/1 数组)}。"""
    out = {}
    io = pd.to_numeric(df.immuno, errors="coerce")
    neo = pd.to_numeric(df.neoadj, errors="coerce")
    s = df[(df.population == "resected") & (neo == 1) & df.neoadj_regimen.isin(["chemo", "chemo_io"])]
    out["neoadj"] = (s, (s.neoadj_regimen == "chemo_io").astype(float).to_numpy())
    s = df[(df.population == "resected") & (neo == 0)]
    out["adjuvant"] = (s, ((df.loc[s.index, "immuno_setting"] == "adjuvant") & (io.loc[s.index] == 1)).astype(float).to_numpy())
    s = df[(df.population == "unresectable") & df.tx_firstline.isin(["chemo", "chemo_io"])]
    out["first_line"] = (s, (s.tx_firstline == "chemo_io").astype(float).to_numpy())
    s = df[df.immuno_setting == "later_line"]
    out["later_line"] = (s, None)
    return out


def iptw(g: pd.DataFrame, treat: np.ndarray, covars, trim):
    X, names, keep, _ = design(g, covars, missing="category", min_level_n=3)
    if g.center.nunique() > 1:
        C = pd.get_dummies(g.center, drop_first=True).astype(float)
        X = np.c_[X, C.to_numpy()]
        names += [f"中心[{c}]" for c in C.columns]
    m = keep & np.isfinite(treat)
    Xs = X[m]
    mu, sd = Xs.mean(0), Xs.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.c_[np.ones(m.sum()), (Xs - mu) / sd]
    from ..stats.classification import _logistic_fit, _sigmoid
    b = _logistic_fit(Z + 0.0, treat[m])
    ps = np.clip(_sigmoid(Z @ b), 1e-4, 1 - 1e-4)
    pt = treat[m].mean()
    w = np.where(treat[m] == 1, pt / ps, (1 - pt) / (1 - ps))
    lo, hi = np.quantile(w, trim)
    w = np.clip(w, lo, hi)

    def smd(weights):
        out = {}
        for j, nm in enumerate(names):
            x = Xs[:, j]
            a, c = treat[m] == 1, treat[m] == 0
            ma = np.average(x[a], weights=weights[a])
            mc = np.average(x[c], weights=weights[c])
            va = np.average((x[a] - ma) ** 2, weights=weights[a])
            vc = np.average((x[c] - mc) ** 2, weights=weights[c])
            out[nm] = float((ma - mc) / np.sqrt(max((va + vc) / 2, 1e-12)))
        return out
    return m, w, {"before": smd(np.ones(len(w))), "after": smd(w)}


def weighted_cox(g, ep, m, w, treat, marker: np.ndarray = None):
    t, e, ok = surv_arrays(g, ep)
    mm = ok[m]
    tt, ee, ww, tr = t[m][mm], e[m][mm], w[mm], treat[m][mm]
    strata = g.center.to_numpy()[m][mm] if g.center.nunique() > 1 else None
    if marker is None:
        X, nm = tr[:, None], ["免疫"]
    else:
        mk = marker[m][mm]
        ok2 = np.isfinite(mk)
        tt, ee, ww, tr, mk = tt[ok2], ee[ok2], ww[ok2], tr[ok2], mk[ok2]
        strata = strata[ok2] if strata is not None else None
        X, nm = np.c_[tr, mk, tr * mk], ["免疫", "修饰变量", "交互项"]
    if ee.sum() < 5:
        return None
    r = S.cox_fit(X, tt, ee, nm, strata=strata, weights=ww)
    tab = r.table().set_index("term")
    key = "交互项" if marker is not None else "免疫"
    return {"hr": float(tab.loc[key, "hr"]), "lo": float(tab.loc[key, "hr_lo"]), "hi": float(tab.loc[key, "hr_hi"]),
            "p": float(tab.loc[key, "p"]), "n": int(len(tt)), "events": int(ee.sum())}


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    ic = sec.get("immuno", {}) or {}
    min_ev = int(ic.get("min_events", 30))
    run = Run(cfg, "v4/immuno", args.run_name)
    log = run.logger
    figs = run.dir / "figs"
    figs.mkdir(exist_ok=True)
    setup_figures(cfg, log)
    df, models, frozen, _ = build(cfg, log)
    covs = list(ic.get("ps_covariates", ["stage_group", "age10", "ecog", "sex"]))
    trim = list(ic.get("ps_trim", [0.01, 0.99]))
    L = ["# v4 探索：免疫治疗亚组", "", f"- 合并队列：{', '.join(COHORT_NAMES.get(c, c) for c in sorted(df.cohort.unique()))}（按中心分层）",
         f"- 倾向性评分协变量：{', '.join(covs)} + 中心；稳定化逆概率权重，按 {trim} 分位数截断；加权 Cox 用稳健标准误。",
         f"- 场景内事件 < {min_ev} 时只描述。结论只写“提示”。", ""]
    results = {}
    for st, (g, treat) in setting_groups(df).items():
        L += [f"## {SETTING_NAMES[st]}", ""]
        t, e, ok = surv_arrays(g, "os")
        if treat is None:
            L += [f"免疫治疗 {len(g)} 例，无可比对照组，只描述。", ""]
            for mk, nm in MARKERS.items():
                if mk in g and g[mk].notna().sum() >= 10 and e[ok].sum() >= 5:
                    v = g[mk].to_numpy(float)
                    hi = (v > np.nanmedian(v)).astype(float)
                    lo_l, hi_l = ftr(f"{nm}低", f"Low {MARKER_EN[mk]}"), ftr(f"{nm}高", f"High {MARKER_EN[mk]}")
                    grp = np.where(hi == 1, hi_l, lo_l)
                    P.km_plot({lab: (t[ok & (grp == lab)], e[ok & (grp == lab)]) for lab in (lo_l, hi_l)},
                              figs / f"km_{st}_{mk}", ftr(f"{SETTING_NAMES[st]} · 按{nm}（中位数）", f"{SETTING_EN[st]} · by {MARKER_EN[mk]} (median)"))
            results[st] = {"n": int(len(g)), "note": "无对照，只描述"}
            continue
        n_io, n_ctrl = int((treat == 1).sum()), int((treat == 0).sum())
        ev = int(e[ok].sum())
        item = {"n_io": n_io, "n_ctrl": n_ctrl, "events": ev}
        L.append(f"免疫 {n_io} 例、对照 {n_ctrl} 例，OS 事件 {ev}。")
        if n_io < 5 or n_ctrl < 5 or ev < min_ev:
            L += ["例数或事件不足，只描述。", ""]
            results[st] = item
            continue
        m, w, bal = iptw(g, treat, covs, trim)
        item["balance"] = bal
        item["io_hr"] = weighted_cox(g, "os", m, w, treat)
        P.km_plot({ft("对照"): (t[ok & (treat == 0)], e[ok & (treat == 0)]), ft("免疫"): (t[ok & (treat == 1)], e[ok & (treat == 1)])},
                  figs / f"km_{st}_io", ftr(f"{SETTING_NAMES[st]} · 免疫 vs 对照（未加权）", f"{SETTING_EN[st]} · IO vs control (unweighted)"))
        worst_b = max(abs(v) for v in bal["before"].values()) if bal["before"] else float("nan")
        worst_a = max(abs(v) for v in bal["after"].values()) if bal["after"] else float("nan")
        h = item["io_hr"] or {}
        L += [f"加权后最大标准化均数差 {fmt(worst_a, 2)}（加权前 {fmt(worst_b, 2)}；< 0.1 视为均衡）。",
              f"免疫治疗 HR（加权）：{fmt(h.get('hr'), 2)}（{fmt(h.get('lo'), 2)}–{fmt(h.get('hi'), 2)}），p={fmt_p(h.get('p'))}", "",
              "| 修饰变量 × 免疫 | 交互 HR（95% CI） | p |", "|---|---|---|"]
        inter = {}
        for mk, nm in MARKERS.items():
            if mk not in g:
                continue
            v = g[mk].to_numpy(float) * (1.0 if mk == "risk_z" else 10.0)
            r = weighted_cox(g, "os", m, w, treat, v)
            inter[mk] = r
            if r:
                L.append(f"| {nm}{'（每 1 SD）' if mk == 'risk_z' else '（每 10%）'} | {fmt(r['hr'], 2)}（{fmt(r['lo'], 2)}–{fmt(r['hi'], 2)}） | {fmt_p(r['p'])} |")
        item["interactions"] = inter
        if st == "neoadj" and "pcr" in g:
            y = pd.to_numeric(g.pcr, errors="coerce")
            item["pcr_rate"] = {"io": float(y[treat == 1].mean()), "ctrl": float(y[treat == 0].mean())}
            L.append(f"\npCR 率：免疫 {fmt(item['pcr_rate']['io'], 2)}，对照 {fmt(item['pcr_rate']['ctrl'], 2)}（未加权，描述）。")
        L += ["", "交互 HR < 1 表示该预测值越高，免疫治疗的相对获益越大。", ""]
        results[st] = item
    write_json(run.path("results.json"), results)
    run.write_text("\n".join(L), "report.md")
    log.info("免疫亚组报告：%s", run.path("report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 探索：免疫治疗亚组（分场景，逆概率加权，预测标志物 × 免疫交互）。", cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
