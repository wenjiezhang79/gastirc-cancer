"""v4 列线图（docs/decisions.md 第十五节第 5 小节）。

每类人群（手术 / 不可切除）一个列线图：风险评分（每 1 SD）+ cTNM 分期（+ 年龄）。在**内部训练集**上用**折外**风险评分拟合 Cox，
系数与基线生存（Breslow）冻结后，在内部测试集与外部队列验证：C-index（与单独 cTNM、单独风险评分比较，配对 bootstrap 差值）、
1/3/5 年校准曲线、决策曲线。内部训练集上的结果为表观值，仅供参考。

命令行：
    python -m gastric_mm.evaluation.nomogram --config configs/server.yaml

输出：<output_dir>/v4/nomogram/<时间>/report.md、nomogram.json（冻结的系数）、results.json、figs/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import load_config
from ..stats import classification as SC
from ..stats import plots as P
from ..stats import survival as S
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from ..stats.plots import tr as ftr
from .common import COHORT_NAMES, POP_NAMES, cohort_groups, design, fc, fmt, fp, ft, setup_figures, surv_arrays, v4, write_json
from .table import build


def fit_frozen(tr: pd.DataFrame, covars, horizons):
    t, e, ok = surv_arrays(tr, "os")
    X, names, keep, notes = design(tr, covars)
    m = ok & keep
    r = S.cox_fit(X[m], t[m], e[m], names)
    base = S.breslow_baseline(r)[0]
    s0 = S.predict_survival(np.zeros(1), base, horizons)[0]
    return {"covariates": list(covars), "terms": names, "coef": r.beta.tolist(), "se": r.se.tolist(),
            "baseline_survival": {str(int(h)): float(v) for h, v in zip(horizons, s0)}, "n": int(m.sum()),
            "events": int(e[m].sum()), "notes": notes, "_base": base}


def apply(model: dict, g: pd.DataFrame):
    X, nms, keep, _ = design(g, model["covariates"], min_level_n=0)
    Xd = pd.DataFrame(X, columns=nms).reindex(columns=model["terms"], fill_value=0.0).to_numpy()
    return Xd @ np.asarray(model["coef"]), keep


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    run = Run(cfg, "v4/nomogram", args.run_name)
    log = run.logger
    figs = run.dir / "figs"
    figs.mkdir(exist_ok=True)
    setup_figures(cfg, log)
    df, models, frozen, _ = build(cfg, log)
    hz = [float(x) for x in sec.get("horizons_months", [12, 36, 60])]
    n_boot = int(sec.get("n_boot", 1000))
    extra = list((sec.get("nomogram", {}) or {}).get("covariates", ["stage_group", "age10"]))
    frozen_out, results = {}, {}
    L = ["# v4 列线图", "", "- 风险评分 + " + ", ".join(extra) + "；内部训练集（折外风险评分）拟合，系数冻结后验证。",
         "- 比较：列线图 vs 单独 cTNM 分期（stage_group）vs 单独风险评分；C-index 差值为配对 bootstrap 95% CI。内部训练集为表观值。", ""]
    for pop in ("resected", "unresectable"):
        d = df[df.population == pop]
        tr = d[d.cohort == "internal_train"]
        if surv_arrays(tr, "os")[1].sum() < int(sec.get("min_events", 30)):
            L += [f"## {POP_NAMES[pop]}", "", "内部训练集事件数不足，不拟合列线图。", ""]
            continue
        specs = {"nomogram": ["risk_z"] + extra, "stage_only": ["stage_group"], "risk_only": ["risk_z"]}
        fits = {}
        for k, cv in specs.items():
            try:
                fits[k] = fit_frozen(tr, cv, hz)
            except Exception as ex:
                log.warning("%s 的 %s 模型拟合失败：%s", POP_NAMES[pop], k, ex)
        if "nomogram" not in fits:
            continue
        nomo = fits["nomogram"]
        frozen_out[pop] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fits.items()}
        # 列线图
        terms = []
        coef = dict(zip(nomo["terms"], nomo["coef"]))
        if "risk_z" in coef:
            terms.append({"name": ftr("风险评分（SD）", "Risk score (SD)"), "coef": coef["risk_z"], "values": [(str(v), v) for v in (-2, -1, 0, 1, 2)]})
        st = [(k.split("[")[1].split(" vs ")[0], v) for k, v in coef.items() if k.startswith("stage_group[")]
        if st:
            ref = [k for k in nomo["terms"] if k.startswith("stage_group[")][0].split(" vs ")[1].rstrip("]")
            terms.append({"name": ftr("cTNM 分期", "cTNM stage"), "coef": 1.0, "values": [(ref, 0.0)] + [(lv, c) for lv, c in st]})
        if "age10" in coef:
            terms.append({"name": ftr("年龄", "Age"), "coef": coef["age10"], "values": [(str(a), a / 10) for a in (30, 40, 50, 60, 70, 80)]})
        if terms:
            P.nomogram_plot(terms, hz, {h: nomo["baseline_survival"][str(int(h))] for h in hz}, figs / f"nomogram_{pop}",
                            f"{fp(pop)} " + ftr("列线图", "nomogram"))
        rp = {}
        L += [f"## {POP_NAMES[pop]}", "", f"拟合：内部训练集 {nomo['n']} 例、事件 {nomo['events']}；"
              + "；".join(f"{t} HR {np.exp(c):.2f}" for t, c in zip(nomo["terms"], nomo["coef"])), "",
              "| 队列 | 例数 | 列线图 C-index | 单独分期 | 单独风险评分 | 列线图 − 单独分期（95% CI） |", "|---|---|---|---|---|---|"]
        for coh, g in cohort_groups(d):
            t, e, ok = surv_arrays(g, "os")
            lps, keep = {}, ok.copy()
            for k, f in fits.items():
                lp, kk = apply(f, g)
                lps[k] = lp
                keep &= kk & np.isfinite(lp)
            if keep.sum() < 10 or e[keep].sum() < 5:
                continue
            c = {k: M.c_index(t[keep], e[keep], v[keep]) for k, v in lps.items()}
            dd = M.paired_bootstrap_diff(M.c_index, [t[keep], e[keep], lps["nomogram"][keep]],
                                         [t[keep], e[keep], lps["stage_only"][keep]], n_boot, 0) if "stage_only" in lps else {}
            surv = S.predict_survival(lps["nomogram"][keep], nomo["_base"], hz)
            cal = {int(h): SC.survival_calibration(t[keep], e[keep], surv[:, j], h) for j, h in enumerate(hz)}
            dca = {}
            for j, h in enumerate(hz[1:], 1):
                curves = {ft("列线图"): SC.decision_curve_survival(t[keep], e[keep], 1 - surv[:, j], h)}
                if "stage_only" in fits:
                    ss = S.predict_survival(lps["stage_only"][keep], fits["stage_only"]["_base"], [h])[:, 0]
                    curves[ft("单独分期")] = SC.decision_curve_survival(t[keep], e[keep], 1 - ss, h)
                P.dca_plot(curves, figs / f"dca_nomo_{pop}_{coh}_{int(h)}", f"{fp(pop)} · {fc(coh)} · " + ftr(f"{int(h)} 个月", f"{int(h)} mo"))
                dca[int(h)] = {k: v.to_dict("records") for k, v in curves.items()}
            P.calibration_plot([(ftr(f"{h} 个月", f"{h} mo"), pts) for h, pts in cal.items() if pts], figs / f"cal_nomo_{pop}_{coh}",
                               f"{fp(pop)} · {fc(coh)} · " + ftr("列线图校准", "nomogram calibration"), ftr("预测生存率", "Predicted survival"),
                               ftr("实际生存率（KM）", "Observed survival (KM)"))
            rp[coh] = {"n": int(keep.sum()), "cindex": c, "diff_vs_stage": dd, "calibration": cal}
            L.append(f"| {COHORT_NAMES[coh]} | {int(keep.sum())} | {fmt(c.get('nomogram'))} | {fmt(c.get('stage_only'))} | "
                     f"{fmt(c.get('risk_only'))} | {fmt(dd.get('diff'))}（{fmt(dd.get('lo'))}–{fmt(dd.get('hi'))}） |")
        L += ["", f"列线图见 figs/nomogram_{pop}.png；校准与决策曲线见 figs/cal_nomo_*、figs/dca_nomo_*。", ""]
        results[pop] = rp
    write_json(run.path("nomogram.json"), frozen_out)
    write_json(run.path("results.json"), results)
    run.write_text("\n".join(L), "report.md")
    log.info("列线图报告：%s", run.path("report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 列线图：风险评分 + cTNM（+ 年龄），内部训练集拟合后冻结，各队列验证 C-index、校准与决策曲线。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
