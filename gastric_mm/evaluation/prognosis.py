"""v4 预后分析（四层；统计分析计划见 docs/decisions.md 第十五节第 3 小节）。

手术患者与不可切除患者始终分开分析；所有分层都使用**同一个主模型**的风险评分（risk_z = 每 1 个标准差，按内部训练集冻结的
均值 / 标准差标准化；高 / 低风险 = 冻结的中位数截断）。

- 第 1 层（主要）：C-index（bootstrap 95% CI）、1/3/5 年时间依赖 AUC；高 / 低风险 KM + log-rank + HR；多因素 Cox
  （完整病例为主，"未记录"作单独一类为敏感性；合并多个中心时按中心分层；比例风险检验）；增量价值（临床模型 vs 临床 + 风险评分，
  系数在内部训练集拟合后冻结，C-index 差值的配对 bootstrap 95% CI）；**预先规定的成功标准**在外部合并队列判定。
- 第 2 层（关键次要）：按分期、按治疗组分层的 KM、log-rank、C-index、组内多因素 Cox；涉及"单纯手术""未接受系统治疗"时
  另报标志时间点分析（只纳入存活 ≥ landmark_months 者，时间从标志点算起）。
- 第 3 层（次要）：各中心 HR 森林图与异质性检验；1/3/5 年校准曲线与决策曲线（模型输出生存概率时）；DFS / PFS 只做 C-index 与 KM。
- 第 4 层（探索）：风险评分 × 治疗、预测标志物 × 对应治疗的交互作用（全部队列合并、按中心分层；结论只写"提示"）。
事件数 < min_events 的层只描述（KM 与生存率），不做检验。

命令行：
    python -m gastric_mm.evaluation.prognosis --config configs/server.yaml

输出：<output_dir>/v4/prognosis/<时间>/report.md、results.json、figs/*.png|pdf
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import metrics as M
from ..config import load_config
from ..stats import classification as SC
from ..stats import plots as P
from ..stats import survival as S
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from .clinical import FIRSTLINE_NAMES, TX_PATTERN_NAMES
from ..stats.plots import tr as ftr
from .common import (COHORT_NAMES, EXTERNAL, POP_NAMES, cohort_groups, design, fc, fmt, fmt_p, fp, ft, p_text, setup_figures,
                     surv_arrays, v4, write_json)
from .table import build

EP_NAMES = {"os": "OS", "dfs": "DFS", "pfs": "PFS"}


class Ctx:
    def __init__(self, cfg, run):
        sec = v4(cfg)
        self.sec = sec
        self.h = [float(x) for x in sec.get("horizons_months", [12, 36, 60])]
        self.n_boot = int(sec.get("n_boot", 1000))
        self.min_ev = int(sec.get("min_events", 30))
        self.landmark = float(sec.get("landmark_months", 3))
        self.figs = run.dir / "figs"
        self.figs.mkdir(exist_ok=True)


# ---------------- 基本量 ----------------
def cindex_ci(t, e, r, n_boot):
    return M.bootstrap_ci(M.c_index, [t, e, r], n_boot, 0)


def cox_simple(x, t, e, strata=None):
    try:
        r = S.cox_fit(np.asarray(x, float)[:, None], t, e, ["x"], strata=strata)
        tab = r.table().iloc[0]
        return {"hr": float(tab.hr), "lo": float(tab.hr_lo), "hi": float(tab.hr_hi), "p": float(tab.p), "coef": float(tab.coef),
                "se": float(tab.se)}
    except Exception:
        return {"hr": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "coef": float("nan"),
                "se": float("nan")}


def km_summary(g: pd.DataFrame, ep: str, group_col: str, ctx: Ctx, title: str, fig_name: str, order=None, names=None,
               test: bool = True) -> dict:
    t, e, ok = surv_arrays(g, ep)
    lv = order or [x for x in pd.unique(g[group_col].dropna())]
    groups, rates = {}, {}
    for v in lv:
        m = ok & (g[group_col] == v).to_numpy()
        if m.sum() == 0:
            continue
        nm = ft((names or {}).get(v, str(v)))
        groups[nm] = (t[m], e[m])
        km = S.kaplan_meier(t[m], e[m])
        rates[nm] = {f"S{int(h)}": km.at(h) for h in ctx.h}
        rates[nm].update(n=int(m.sum()), events=int(e[m].sum()), median=km.median())
    out = {"groups": rates}
    mask = ok & g[group_col].notna().to_numpy()
    if test and len(groups) >= 2 and e[mask].sum() >= ctx.min_ev:
        lr = S.logrank(t[mask], e[mask], g.loc[mask, group_col].to_numpy())
        out["logrank_p"] = lr["p"]
        if len(groups) == 2:
            hi = (g.loc[mask, group_col] == lv[-1]).astype(float).to_numpy()
            out["hr"] = cox_simple(hi, t[mask], e[mask])
    else:
        out["note"] = f"事件数 {int(e[mask].sum())} < {ctx.min_ev}，只描述" if len(groups) >= 2 else "分组不足"
    ann = f"log-rank {p_text(out.get('logrank_p'))}" if "logrank_p" in out else ftr(out.get("note", ""), "")
    if "hr" in out:
        ann += f"\nHR {out['hr']['hr']:.2f} ({out['hr']['lo']:.2f}–{out['hr']['hi']:.2f})"
    if groups:
        tmax = max(float(np.max(v[0])) for v in groups.values())
        step = 12 if tmax > 36 else 6
        P.km_plot(groups, ctx.figs / fig_name, title, risk_times=list(np.arange(0, tmax + 1e-9, step)), annotation=ann)
        out["figure"] = f"figs/{fig_name}.png"
    return out


def multivariable(g: pd.DataFrame, ep: str, covars: Sequence[str], ctx: Ctx, missing: str = "complete",
                  strata_col: Optional[str] = None) -> dict:
    t, e, ok = surv_arrays(g, ep)
    X, names, keep, notes = design(g, covars, missing=missing)
    m = ok & keep
    if m.sum() < 10 or e[m].sum() < ctx.min_ev or X.shape[1] == 0:
        return {"note": f"可分析病例 {int(m.sum())}、事件 {int(e[m].sum())}，不足以做多因素 Cox", "n": int(m.sum())}
    strata = g.loc[m, strata_col].to_numpy() if strata_col and g.loc[m, strata_col].nunique() > 1 else None
    try:
        r = S.cox_fit(X[m], t[m], e[m], names, strata=strata)
    except Exception as ex:  # 数值问题（如完全分离）
        return {"note": f"Cox 拟合失败：{type(ex).__name__}", "n": int(m.sum())}
    tab = r.table()
    ph = S.cox_ph_test(r)
    return {"n": int(m.sum()), "events": int(e[m].sum()), "table": tab.to_dict("records"), "ph": ph.to_dict("records"),
            "notes": notes, "stratified": strata is not None, "converged": r.converged}


def treatment_covars(pop: str, sec) -> List[str]:
    cc = sec.get("cox_covariates", {}) or {}
    tr = cc.get("resected_treatment" if pop == "resected" else "unresectable_treatment", []) or []
    return list(cc.get("common", ["risk_z", "stage_group", "age10", "sex", "ecog"])) + list(tr)


# ---------------- 增量价值（系数在内部训练集拟合后冻结） ----------------
def incremental(df_pop: pd.DataFrame, ep: str, ctx: Ctx, groups) -> dict:
    clin = list((ctx.sec.get("incremental", {}) or {}).get("clinical", ["stage_group", "age10", "sex", "ecog"]))
    tr = df_pop[df_pop.cohort == "internal_train"]
    t, e, ok = surv_arrays(tr, ep)
    fits = {}
    for nm, cv in (("clinical", clin), ("clinical_risk", clin + ["risk_z"])):
        X, names, keep, notes = design(tr, cv)
        m = ok & keep
        if e[m].sum() < ctx.min_ev:
            return {"note": "内部训练集事件不足，无法拟合增量价值模型"}
        r = S.cox_fit(X[m], t[m], e[m], names)
        fits[nm] = (cv, names, r.beta)
    out = {"clinical_terms": fits["clinical"][1], "full_terms": fits["clinical_risk"][1], "by_cohort": {}}
    for coh, g in groups:
        tg, eg, okg = surv_arrays(g, ep)
        lps = {}
        keep_all = okg.copy()
        for nm, (cv, names, beta) in fits.items():
            X, nms, keep, _ = design(g, cv, min_level_n=0)
            Xd = pd.DataFrame(X, columns=nms).reindex(columns=names, fill_value=0.0).to_numpy()
            lps[nm] = Xd @ beta
            keep_all &= keep
        if keep_all.sum() < 10 or eg[keep_all].sum() < 5:
            continue
        a, b = lps["clinical_risk"][keep_all], lps["clinical"][keep_all]
        d = M.paired_bootstrap_diff(M.c_index, [tg[keep_all], eg[keep_all], a], [tg[keep_all], eg[keep_all], b], ctx.n_boot, 0)
        out["by_cohort"][coh] = {"c_clinical": M.c_index(tg[keep_all], eg[keep_all], b),
                                 "c_full": M.c_index(tg[keep_all], eg[keep_all], a), "diff": d["diff"], "lo": d["lo"],
                                 "hi": d["hi"], "n": int(keep_all.sum()), "events": int(eg[keep_all].sum())}
    return out


# ---------------- 各层 ----------------
def layer1(df_pop, pop, ep, ctx: Ctx, groups) -> dict:
    res = {}
    for coh, g in groups:
        t, e, ok = surv_arrays(g, ep)
        r = g["risk_z"].to_numpy(float)
        m = ok & np.isfinite(r)
        if m.sum() < 10:
            continue
        item = {"n": int(m.sum()), "events": int(e[m].sum()), "median_followup": S.median_followup(t[m], e[m])}
        if e[m].sum() >= 5:
            item["cindex"] = cindex_ci(t[m], e[m], r[m], ctx.n_boot)
            item["td_auc"] = {int(h): S.time_dependent_auc(t[m], e[m], r[m], h) for h in ctx.h}
            item["hr_per_sd"] = cox_simple(r[m], t[m], e[m])
        g2 = g.assign(_grp=g["risk_high"].map({0.0: "低风险", 1.0: "高风险"}))
        item["km"] = km_summary(g2, ep, "_grp", ctx, f"{fp(pop)} · {fc(coh)} · {EP_NAMES[ep]}",
                                f"km_{ep}_{pop}_{coh}", order=["低风险", "高风险"])
        if ep == "os":
            g3 = g.assign(_grp=g["risk_tertile"])
            item["km_tertile"] = km_summary(g3, ep, "_grp", ctx, f"{fp(pop)} · {fc(coh)} · " + ftr("三分位", "tertiles"),
                                            f"km3_{ep}_{pop}_{coh}", order=["低", "中", "高"],
                                            names={"低": "低风险", "中": "中风险", "高": "高风险"})
            covs = treatment_covars(pop, ctx.sec)
            strata = ctx.sec.get("cox_strata", "center")
            item["cox"] = multivariable(g, ep, covs, ctx, "complete", strata)
            item["cox_missing_category"] = multivariable(g, ep, covs, ctx, "category", strata)
        res[coh] = item
    if ep == "os":
        res["_incremental"] = incremental(df_pop, ep, ctx, groups)
    return res


def success_criterion(l1: dict) -> dict:
    ext = l1.get("external")
    inc = l1.get("_incremental", {}).get("by_cohort", {}).get("external")
    if not ext or not inc or "table" not in ext.get("cox", {}):
        return {"evaluated": False, "note": "无外部合并队列（开发模式或外部队列尚未运行最终评估），暂不判定"}
    row = [r for r in ext["cox"]["table"] if r["term"] == "risk_z"]
    hr_ok = bool(row and row[0]["hr_lo"] > 1.0)
    inc_ok = bool(np.isfinite(inc["lo"]) and inc["lo"] > 0)
    return {"evaluated": True, "risk_hr_lo_gt_1": hr_ok, "incremental_lo_gt_0": inc_ok, "passed": hr_ok and inc_ok,
            "hr": row[0] if row else None, "incremental": inc}


def subgroup(g, pop, ep, col, levels, names, ctx: Ctx, coh, landmark_levels=()) -> dict:
    out = {}
    within = [c for c in ("stage_group", "age10", "ecog") if c != col]
    for lv in levels:
        s = g[g[col] == lv]
        if len(s) < 5:
            continue
        t, e, ok = surv_arrays(s, ep)
        r = s["risk_z"].to_numpy(float)
        m = ok & np.isfinite(r)
        item = {"n": int(m.sum()), "events": int(e[m].sum())}
        s2 = s.assign(_grp=s["risk_high"].map({0.0: "低风险", 1.0: "高风险"}))
        item["km"] = km_summary(s2, ep, "_grp", ctx, f"{fp(pop)} · {ft(names.get(lv, lv)) if names.get(lv) else ftr(lv + ' 期', 'Stage ' + lv)} · {fc(coh)}",
                                f"km_{ep}_{pop}_{col}_{lv}_{coh}", order=["低风险", "高风险"])
        if e[m].sum() >= ctx.min_ev:
            item["cindex"] = cindex_ci(t[m], e[m], r[m], ctx.n_boot)
            item["hr_per_sd"] = cox_simple(r[m], t[m], e[m])
            item["cox"] = multivariable(s, ep, ["risk_z"] + within, ctx, "complete", ctx.sec.get("cox_strata", "center"))
        if lv in landmark_levels:
            L = ctx.landmark
            sl = s2[(pd.to_numeric(s2[f"{ep}_time"], errors="coerce") >= L)].copy()
            sl[f"{ep}_time"] = pd.to_numeric(sl[f"{ep}_time"]) - L
            item["landmark"] = km_summary(sl, ep, "_grp", ctx, f"{ft(names.get(lv, lv))} · " + ftr(f"标志时间点 {L:g} 个月", f"landmark {L:g} mo") + f" · {fc(coh)}",
                                          f"kmL_{ep}_{pop}_{col}_{lv}_{coh}", order=["低风险", "高风险"])
            tl, el, okl = surv_arrays(sl, ep)
            rl = sl["risk_z"].to_numpy(float)
            ml = okl & np.isfinite(rl)
            if el[ml].sum() >= ctx.min_ev:
                item["landmark"]["cindex"] = cindex_ci(tl[ml], el[ml], rl[ml], ctx.n_boot)
        out[lv] = item
    return out


def layer2(pop, ep, ctx: Ctx, groups) -> dict:
    res = {}
    for coh, g in groups:
        item = {"stage": subgroup(g, pop, ep, "stage_group", ["I", "II", "III", "IV"], {}, ctx, coh)}
        if pop == "resected":
            item["treatment"] = subgroup(g, pop, ep, "tx_pattern", ["surgery_only", "surgery_adjuvant", "neoadj_surgery"],
                                         TX_PATTERN_NAMES, ctx, coh, landmark_levels=("surgery_only",))
        else:
            lv = [x for x in ("none", "chemo", "chemo_her2", "chemo_io", "chemo_cldn", "chemo_other", "other")
                  if (g.tx_firstline == x).any()]
            item["treatment"] = subgroup(g, pop, ep, "tx_firstline", lv, FIRSTLINE_NAMES, ctx, coh, landmark_levels=("none",))
        res[coh] = item
    return res


def layer3(df_pop, pop, ep, ctx: Ctx, groups) -> dict:
    res = {"centers": [], "calibration": {}, "dca": {}}
    centers = [("test", None)] + [(c, None) for c in EXTERNAL]
    rows = []
    for coh, lab in centers:
        g = df_pop[df_pop.cohort == coh]
        t, e, ok = surv_arrays(g, ep)
        r = g["risk_z"].to_numpy(float)
        m = ok & np.isfinite(r)
        if m.sum() < 10:
            continue
        if e[m].sum() >= ctx.min_ev:
            h = cox_simple(r[m], t[m], e[m])
            rows.append({"label": fc(coh), "coh": coh, "n": int(m.sum()), **h})
        else:
            rows.append({"label": fc(coh), "coh": coh, "n": int(m.sum()), "hr": float("nan"), "lo": float("nan"), "hi": float("nan"),
                         "note": f"事件 {int(e[m].sum())} < {ctx.min_ev}"})
    res["centers"] = rows
    est = [r for r in rows if np.isfinite(r["hr"])]
    if len(est) >= 2:
        b = np.array([r["coef"] for r in est])
        w = 1 / np.array([r["se"] for r in est]) ** 2
        bbar = (w * b).sum() / w.sum()
        Q = float((w * (b - bbar) ** 2).sum())
        res["heterogeneity"] = {"Q": Q, "df": len(est) - 1, "p": float(sps.chi2.sf(Q, len(est) - 1)),
                                "I2": float(max(0.0, (Q - (len(est) - 1)) / Q)) if Q > 0 else 0.0}
    if rows:
        P.forest_plot(rows, ctx.figs / f"forest_{ep}_{pop}", f"{fp(pop)} · " + ftr("各中心风险评分 HR（每 1 SD）", "HR per SD of risk score by center"))
        res["forest_figure"] = f"figs/forest_{ep}_{pop}.png"
    for coh, g in groups:
        cal, dca = {}, {}
        t, e, ok = surv_arrays(g, ep)
        for h in ctx.h:
            col = f"{ep}_S{int(h)}m"
            if col not in g or g[col].isna().all():
                continue
            sp = g[col].to_numpy(float)
            m = ok & np.isfinite(sp)
            pts = SC.survival_calibration(t[m], e[m], sp[m], h)
            if pts:
                cal[int(h)] = pts
                dca[int(h)] = SC.decision_curve_survival(t[m], e[m], 1 - sp[m], h).to_dict("records")
        if cal:
            P.calibration_plot([(ftr(f"{int(h)} 个月", f"{int(h)} mo"), pts) for h, pts in cal.items()], ctx.figs / f"cal_{ep}_{pop}_{coh}",
                               f"{fp(pop)} · {fc(coh)} · " + ftr("生存校准", "calibration"), ftr("预测生存率", "Predicted survival"),
                               ftr("实际生存率（KM）", "Observed survival (KM)"))
            P.dca_plot({ftr(f"{h} 个月", f"{h} mo"): pd.DataFrame(v) for h, v in dca.items()}, ctx.figs / f"dca_{ep}_{pop}_{coh}",
                       f"{fp(pop)} · {fc(coh)} · " + ftr("决策曲线", "decision curve"))
            res["calibration"][coh] = cal
            res["dca"][coh] = {h: "见图" for h in dca}
    return res


def interaction_cox(g, ep, treat: np.ndarray, effect_mod: np.ndarray, label: str, ctx: Ctx) -> dict:
    t, e, ok = surv_arrays(g, ep)
    Xc, names, keep, _ = design(g, ["stage_group", "age10", "ecog"])
    m = ok & keep & np.isfinite(treat) & np.isfinite(effect_mod)
    if len(np.unique(treat[m])) < 2:
        return {"term": label, "n": int(m.sum()), "events": int(e[m].sum()),
                "note": "没有接受该治疗的病例（可能因例数少已并入“化疗 + 其他”）"}
    if m.sum() < 20 or e[m].sum() < ctx.min_ev:
        return {"term": label, "n": int(m.sum()), "events": int(e[m].sum()), "note": "例数或事件不足，只描述"}
    X = np.column_stack([effect_mod[m], treat[m], effect_mod[m] * treat[m], Xc[m]])
    nm = ["修饰变量", "治疗", "交互项"] + names
    strata = g.loc[m, "center"].to_numpy() if g.loc[m, "center"].nunique() > 1 else None
    try:
        r = S.cox_fit(X, t[m], e[m], nm, strata=strata)
    except Exception as ex:
        return {"term": label, "note": f"拟合失败：{type(ex).__name__}"}
    tab = r.table().set_index("term")
    return {"term": label, "n": int(m.sum()), "events": int(e[m].sum()),
            "interaction_hr": float(tab.loc["交互项", "hr"]), "lo": float(tab.loc["交互项", "hr_lo"]),
            "hi": float(tab.loc["交互项", "hr_hi"]), "p": float(tab.loc["交互项", "p"]),
            "treat_hr_at_mean": float(tab.loc["治疗", "hr"])}


def layer4(df_pop, pop, ep, ctx: Ctx) -> dict:
    """全部队列合并（按中心分层）。只作探索。"""
    ic = ctx.sec.get("interactions", {}) or {}
    g = df_pop
    res = []
    rz = g["risk_z"].to_numpy(float)
    if pop == "resected":
        for tr in ic.get("treatments_resected", ["adj_chemo", "neoadj"]) or []:
            if tr == "adj_chemo":
                s = g[g.tx_pattern.isin(["surgery_only", "surgery_adjuvant"])]
                treat = (s.tx_pattern == "surgery_adjuvant").astype(float).to_numpy()
                res.append(interaction_cox(s, ep, treat, s.risk_z.to_numpy(float), "风险评分 × 辅助化疗（非新辅助者）", ctx))
                L = ctx.landmark
                sl = s[pd.to_numeric(s[f"{ep}_time"], errors="coerce") >= L].copy()
                sl[f"{ep}_time"] = pd.to_numeric(sl[f"{ep}_time"]) - L
                tl = (sl.tx_pattern == "surgery_adjuvant").astype(float).to_numpy()
                res.append(interaction_cox(sl, ep, tl, sl.risk_z.to_numpy(float),
                                           f"风险评分 × 辅助化疗（标志时间点 {L:g} 个月）", ctx))
                _interaction_km(s, ep, treat, ctx, pop, "adj_chemo", ("未辅助化疗", "辅助化疗"))
            elif tr == "neoadj":
                treat = pd.to_numeric(g.neoadj, errors="coerce").to_numpy(float)
                res.append(interaction_cox(g, ep, treat, rz, "风险评分 × 新辅助", ctx))
    else:
        fl = g.tx_firstline
        for tr in ic.get("treatments_unresectable", []) or []:
            s = g[fl.isin(["chemo", tr])]
            treat = (s.tx_firstline == tr).astype(float).to_numpy()
            res.append(interaction_cox(s, ep, treat, s.risk_z.to_numpy(float),
                                       f"风险评分 × {FIRSTLINE_NAMES.get(tr, tr)}（vs 单纯化疗）", ctx))
        for marker, tr in (ic.get("marker_by_treatment", {}) or {}).items():
            col = f"{marker}_prob"
            if col not in g:
                continue
            s = g[fl.isin(["chemo", tr])]
            treat = (s.tx_firstline == tr).astype(float).to_numpy()
            res.append(interaction_cox(s, ep, treat, s[col].to_numpy(float) * 10,
                                       f"预测 {marker}（每 10%）× {FIRSTLINE_NAMES.get(tr, tr)}（vs 单纯化疗）", ctx))
    return {"pooled_cohorts": sorted(g.cohort.unique().tolist()), "rows": res}


def _interaction_km(s, ep, treat, ctx, pop, tag, names):
    lab = np.where(s.risk_high.to_numpy() == 1, ft("高风险"), ft("低风险"))
    grp = [f"{a}·{ft(names[int(b)])}" if np.isfinite(b) else None for a, b in zip(lab, treat)]
    s2 = s.assign(_grp=grp)
    order = [f"{ft(a)}·{ft(b)}" for a in ("低风险", "高风险") for b in names]
    km_summary(s2, ep, "_grp", ctx, f"{fp(pop)} · " + ftr(f"风险 × {names[1]}（探索）", f"risk × {ft(names[1])} (exploratory)"), f"kmX_{ep}_{pop}_{tag}", order=order,
               test=False)


# ---------------- 报告 ----------------
def _ci(d):
    return f"{fmt(d[0])}（{fmt(d[1])}–{fmt(d[2])}）" if d else "—"


def _hr(d):
    return f"{fmt(d['hr'], 2)}（{fmt(d['lo'], 2)}–{fmt(d['hi'], 2)}），p={fmt_p(d['p'])}" if d and np.isfinite(d.get("hr", np.nan)) \
        else (d or {}).get("note", "—")


def report(results: dict, ctx: Ctx, dev: bool) -> str:
    L = ["# v4 预后分析报告", "",
         "- 统计分析计划：docs/decisions.md 第十五节第 3 小节（预先规定）；风险评分来自主模型，按内部训练集冻结的均值 / 标准差标准化（每 1 SD）。",
         "- 手术患者与不可切除患者分开分析；所有分层都用同一个模型的风险评分。",
         f"- 事件数 < {ctx.min_ev} 的层只描述，不做检验；第 2–4 层为次要 / 探索性分析，不做多重比较校正。"]
    if dev:
        L.append("- **开发模式**：尚无最终评估结果，只有内部训练集折外预测；成功标准暂不判定。")
    L.append("")
    for (pop, ep), r in results.items():
        L += [f"## {POP_NAMES[pop]} · {EP_NAMES[ep]}", ""]
        l1 = r["layer1"]
        L += ["### 第 1 层（主要）", "", "| 队列 | 例数 | 事件 | 中位随访（月） | C-index（95% CI） | "
              + " | ".join(f"{int(h)} 个月 AUC" for h in ctx.h) + " | 高 vs 低风险 HR | log-rank p |",
              "|" + "---|" * (7 + len(ctx.h))]
        for coh, it in l1.items():
            if coh.startswith("_"):
                continue
            km = it.get("km", {})
            L.append(f"| {COHORT_NAMES[coh]} | {it['n']} | {it['events']} | {fmt(it['median_followup'], 1)} | {_ci(it.get('cindex'))} | "
                     + " | ".join(fmt(it.get('td_auc', {}).get(int(h))) for h in ctx.h)
                     + f" | {_hr(km.get('hr'))} | {fmt_p(km.get('logrank_p'))} |")
        L.append("")
        if ep == "os":
            for coh, it in l1.items():
                if coh.startswith("_") or "cox" not in it:
                    continue
                cx = it["cox"]
                L.append(f"**多因素 Cox · {COHORT_NAMES[coh]}**（完整病例 n={cx.get('n')}，事件 {cx.get('events', '—')}"
                         f"{'，按中心分层' if cx.get('stratified') else ''}）")
                L.append("")
                if "table" in cx:
                    phd = {p["term"]: p["p"] for p in cx.get("ph", [])}
                    L += ["| 变量 | HR（95% CI） | p | 比例风险检验 p |", "|---|---|---|---|"]
                    for row in cx["table"]:
                        L.append(f"| {row['term']} | {fmt(row['hr'], 2)}（{fmt(row['hr_lo'], 2)}–{fmt(row['hr_hi'], 2)}） | "
                                 f"{fmt_p(row['p'])} | {fmt_p(phd.get(row['term']))} |")
                    sens = it.get("cox_missing_category", {})
                    rz = [x for x in sens.get("table", []) if x["term"] == "risk_z"]
                    if rz:
                        L.append(f"| 敏感性（“未记录”作单独一类，n={sens['n']}）risk_z | {fmt(rz[0]['hr'], 2)}"
                                 f"（{fmt(rz[0]['hr_lo'], 2)}–{fmt(rz[0]['hr_hi'], 2)}） | {fmt_p(rz[0]['p'])} | |")
                    if cx.get("notes"):
                        L.append("")
                        L.append("说明：" + "；".join(cx["notes"]))
                else:
                    L.append(cx.get("note", ""))
                L.append("")
            inc = l1.get("_incremental", {})
            if inc.get("by_cohort"):
                L += ["**增量价值**（临床模型：" + ", ".join(inc["clinical_terms"]) + "；系数在内部训练集拟合后冻结；内部训练集为表观值）", "",
                      "| 队列 | 临床模型 C-index | + 风险评分 | 差值（95% CI） |", "|---|---|---|---|"]
                for coh, d in inc["by_cohort"].items():
                    L.append(f"| {COHORT_NAMES[coh]} | {fmt(d['c_clinical'])} | {fmt(d['c_full'])} | "
                             f"{fmt(d['diff'])}（{fmt(d['lo'])}–{fmt(d['hi'])}） |")
                L.append("")
            sc = r.get("success", {})
            if sc.get("evaluated"):
                L.append(f"**预先规定的成功标准（外部合并队列）：{'达到' if sc['passed'] else '未达到'}**——风险评分 HR 95% CI 下限 > 1："
                         f"{'是' if sc['risk_hr_lo_gt_1'] else '否'}；增量 C-index 95% CI 下限 > 0：{'是' if sc['incremental_lo_gt_0'] else '否'}。")
            else:
                L.append(f"预先规定的成功标准：{sc.get('note', '')}")
            L.append("")
        if "layer2" in r:
            L += ["### 第 2 层（关键次要：按分期、按治疗分层）", ""]
            for coh, it in r["layer2"].items():
                for kind, title in (("stage", "分期"), ("treatment", "治疗组")):
                    sub = it.get(kind, {})
                    if not sub:
                        continue
                    L += [f"**{COHORT_NAMES[coh]} · 按{title}**", "",
                          "| 分组 | 例数 | 事件 | C-index（95% CI） | 风险评分 HR（每 1 SD） | 高 vs 低 log-rank p | 标志时间点 log-rank p |",
                          "|---|---|---|---|---|---|---|"]
                    for lv, d in sub.items():
                        nm = TX_PATTERN_NAMES.get(lv) or FIRSTLINE_NAMES.get(lv) or (f"{lv} 期" if kind == "stage" else lv)
                        lm = d.get("landmark", {})
                        L.append(f"| {nm} | {d['n']} | {d['events']} | {_ci(d.get('cindex'))} | {_hr(d.get('hr_per_sd'))} | "
                                 f"{fmt_p(d['km'].get('logrank_p')) if 'logrank_p' in d['km'] else d['km'].get('note', '—')} | "
                                 f"{fmt_p(lm.get('logrank_p')) if lm else ''} |")
                    L.append("")
        if "layer3" in r:
            l3 = r["layer3"]
            L += ["### 第 3 层（次要）", ""]
            if l3.get("centers"):
                L += ["| 中心 | 例数 | 风险评分 HR（每 1 SD） |", "|---|---|---|"]
                for row in l3["centers"]:
                    L.append(f"| {COHORT_NAMES.get(row.get('coh'), row['label'])} | {row['n']} | {_hr(row)} |")
                het = l3.get("heterogeneity")
                if het:
                    L.append(f"\n异质性检验：Q={het['Q']:.2f}（df={het['df']}），p={fmt_p(het['p'])}，I²={100 * het['I2']:.0f}%")
                L.append("")
            if l3.get("calibration"):
                L.append("生存校准与决策曲线见 figs/cal_*、figs/dca_*。")
                L.append("")
        if "layer4" in r:
            l4 = r["layer4"]
            L += [f"### 第 4 层（探索；合并队列：{', '.join(COHORT_NAMES.get(c, c) for c in l4['pooled_cohorts'])}，按中心分层；结论只写“提示”）", "",
                  "| 交互项 | 例数 | 事件 | 交互 HR（95% CI） | p |", "|---|---|---|---|---|"]
            for row in l4["rows"]:
                if "interaction_hr" in row:
                    L.append(f"| {row['term']} | {row['n']} | {row['events']} | {fmt(row['interaction_hr'], 2)}"
                             f"（{fmt(row['lo'], 2)}–{fmt(row['hi'], 2)}） | {fmt_p(row['p'])} |")
                else:
                    L.append(f"| {row['term']} | {row.get('n', '—')} | {row.get('events', '—')} | {row.get('note', '')} | |")
            L += ["", "交互 HR < 1 表示修饰变量越高（如风险越高），该治疗的相对获益越大。", ""]
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "v4/prognosis", args.run_name)
    log = run.logger
    df, models, frozen, _ = build(cfg, log)
    ctx = Ctx(cfg, run)
    setup_figures(cfg, log)
    dev = not models[v4(cfg).models.main].final_cohorts
    results = {}
    for pop in ("resected", "unresectable"):
        d = df[df.population == pop]
        if len(d) < 10:
            continue
        groups = cohort_groups(d)
        for ep in ("os", "dfs" if pop == "resected" else "pfs"):
            if f"{ep}_time" not in d or d[f"{ep}_time"].notna().sum() < 10:
                continue
            log.info("预后分析：%s · %s", POP_NAMES[pop], EP_NAMES[ep])
            r = {"layer1": layer1(d, pop, ep, ctx, groups)}
            if ep == "os":
                r["success"] = success_criterion(r["layer1"])
                r["layer2"] = layer2(pop, ep, ctx, groups)
                r["layer3"] = layer3(d, pop, ep, ctx, groups)
                r["layer4"] = layer4(d, pop, ep, ctx)
            results[(pop, ep)] = r
    run.write_text(report(results, ctx, dev), "report.md")
    write_json(run.path("results.json"), {f"{p}_{e}": v for (p, e), v in results.items()})
    log.info("预后分析报告：%s", run.path("report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 预后分析：四层（整体、分期 / 治疗分层、中心 / 校准 / 决策曲线、交互作用），手术与不可切除分开。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
