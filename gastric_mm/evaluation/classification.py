"""v4 分类任务评价（Lauren 分型与标志物；docs/decisions.md 第十五节第 2 小节）。

每个队列（内部训练集折外、内部测试集、外部各中心、外部合并）× 每个模型 × 每个任务：
- AUC 与 DeLong 95% CI（Lauren 三分类为宏平均 AUC + bootstrap CI，另报肠型 vs 非肠型）、PR 曲线下面积、
  按冻结的约登阈值计算的敏感度 / 特异度；
- 主模型与各对照模型的 DeLong 配对检验（在两者都有预测的病例上）；
- 校准：校准曲线、Brier 分数、校准截距与斜率。外部队列另做敏感性分析：只用该队列数据调整整体偏移（截距），报告调整量与调整后的校准。
MSI、EBV 阳性少，以外部合并结果为主要参考。

命令行：
    python -m gastric_mm.evaluation.classification --config configs/server.yaml

输出：<output_dir>/v4/classification/<时间>/report.md、results.json、figs/roc_*、figs/cal_*
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import load_config
from ..stats import classification as SC
from ..stats import plots as P
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from ..stats.plots import tr as ftr
from .common import (COHORT_NAMES, EXTERNAL, ModelPreds, cohort_groups, fc, fmt, fmt_p, load_covariates, merged_table,
                     setup_figures, v4, write_json)

TASK_NAMES = {"lauren": "Lauren 分型", "lauren_intestinal": "Lauren 肠型 vs 非肠型", "her2_pos": "HER2", "msi_h": "MSI-H",
              "ebv_pos": "EBV", "pdl1_cps1": "PD-L1 CPS≥1", "pdl1_cps5": "PD-L1 CPS≥5", "pdl1_cps10": "PD-L1 CPS≥10",
              "cldn182_pos": "Claudin18.2"}
TASK_EN = {"lauren": "Lauren type", "lauren_intestinal": "Lauren intestinal vs non-intestinal", "her2_pos": "HER2",
           "msi_h": "MSI-H", "ebv_pos": "EBV", "pdl1_cps1": "PD-L1 CPS≥1", "pdl1_cps5": "PD-L1 CPS≥5", "pdl1_cps10": "PD-L1 CPS≥10",
           "cldn182_pos": "Claudin18.2"}


def tn(task):
    return ftr(TASK_NAMES.get(task, task), TASK_EN.get(task, task))


def scores(df: pd.DataFrame, task: str):
    """返回 (标签, 分数)；lauren_intestinal 用三分类概率中的肠型概率。"""
    if task == "lauren_intestinal":
        return df["lauren_intestinal"].to_numpy(float), df["lauren_p0"].to_numpy(float) if "lauren_p0" in df else None
    col = f"{task}_prob"
    return df[task].to_numpy(float), (df[col].to_numpy(float) if col in df else None)


def eval_binary(y, s, thr, n_boot) -> dict:
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    out = {"n": int(len(y)), "n_pos": int((y == 1).sum())}
    if out["n_pos"] < 2 or (y == 0).sum() < 2:
        out["note"] = "阳性或阴性不足 2 例"
        return out
    out.update(SC.delong_auc(y, s))
    out["pr_auc"] = M.pr_auc(y, s)
    if np.isfinite(thr):
        out["sens"], out["spec"] = M.sens_spec(y, s, thr)
        out["threshold"] = thr
    return out


def eval_lauren(df, n_boot) -> dict:
    cols = [f"lauren_p{k}" for k in range(3)]
    if not all(c in df for c in cols):
        return {"note": "该模型没有 Lauren 预测"}
    y = df["lauren"].to_numpy(float)
    Pm = df[cols].to_numpy(float)
    ok = np.isfinite(y) & np.all(np.isfinite(Pm), 1)
    v, lo, hi = M.bootstrap_ci(lambda a, b: M.multiclass_auc(a, b)[0], [y[ok], Pm[ok]], n_boot, 0, strata=y[ok])
    return {"n": int(ok.sum()), "auc": v, "lo": lo, "hi": hi, "balanced_acc": M.balanced_accuracy(y[ok], Pm[ok])}


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    run = Run(cfg, "v4/classification", args.run_name)
    log = run.logger
    figs = run.dir / "figs"
    figs.mkdir(exist_ok=True)
    setup_figures(cfg, log)
    n_boot = int(sec.get("n_boot", 1000))
    bins = int(sec.get("calibration_bins", 10))
    cov = load_covariates(cfg, log)
    refs = [sec.models.main] + list(sec.models.get("compare", []) or [])
    models = {r: ModelPreds(cfg, r) for r in refs}
    tables = {r: merged_table(cov, mp) for r, mp in models.items()}
    main = sec.models.main
    tasks = []
    for t in sec.get("classification_tasks", []):
        tasks += [t, "lauren_intestinal"] if t == "lauren" else [t]
    results: Dict[str, dict] = {}
    for coh, gmain in cohort_groups(tables[main]):
        rc = {}
        for task in tasks:
            rt = {"models": {}, "delong": {}, "calibration": {}}
            curves = []
            for ref, tab in tables.items():
                g = tab[tab.cohort.isin(EXTERNAL)] if coh == "external" else tab[tab.cohort == coh]
                if task == "lauren":
                    rt["models"][ref] = eval_lauren(g, n_boot)
                    continue
                y, s = scores(g, task)
                if s is None:
                    continue
                rt["models"][ref] = eval_binary(y, s, models[ref].threshold(task), n_boot)
                curves.append((ref.split("/")[-1], y, s))
                cal = SC.calibration(y, s, bins)
                if coh in EXTERNAL or coh == "external":
                    cal["recalibration"] = SC.recalibrate_intercept(y, s)
                rt["calibration"][ref] = cal
            if task != "lauren":
                gm = gmain
                ym, sm = scores(gm, task)
                for ref in refs[1:]:
                    tab = tables[ref]
                    g = tab[tab.cohort.isin(EXTERNAL)] if coh == "external" else tab[tab.cohort == coh]
                    yo, so = scores(g, task)
                    if so is None or sm is None:
                        continue
                    a = pd.DataFrame({"study_id": gm.study_id, "y": ym, "a": sm})
                    b = pd.DataFrame({"study_id": g.study_id, "b": so})
                    j = a.merge(b, on="study_id")
                    rt["delong"][ref] = SC.delong_test(j.y.to_numpy(), j.a.to_numpy(), j.b.to_numpy())
                if curves:
                    P.roc_plot(curves, figs / f"roc_{task}_{coh}", f"{tn(task)} · {fc(coh)}")
            rc[task] = rt
        results[coh] = rc
    # 主模型各队列的校准图（每个任务一张）
    for task in tasks:
        if task == "lauren":
            continue
        series = [(fc(c), results[c][task]["calibration"].get(main, {}).get("curve", [])) for c in results]
        if any(s for _, s in series):
            P.calibration_plot(series, figs / f"cal_{task}", f"{tn(task)} · " + ftr("主模型校准", "calibration (main model)"))
    write_json(run.path("results.json"), results)
    run.write_text(report(results, refs, tasks, models), "report.md")
    log.info("分类评价报告：%s", run.path("report.md"))
    return 0


def report(results, refs, tasks, models) -> str:
    main = refs[0]
    dev = not models[main].final_cohorts
    L = ["# v4 分类任务评价", "", f"- 主模型：{main}；对照：{', '.join(refs[1:]) or '无'}",
         "- AUC 的 95% CI：二分类为 DeLong 法，Lauren 三分类（宏平均 AUC）为 bootstrap；敏感度 / 特异度用内部训练集折外预测确定的约登阈值（冻结）。",
         "- MSI、EBV 阳性少，以外部合并结果为主要参考。"]
    if dev:
        L.append("- **开发模式**：尚无最终评估结果，只有内部训练集折外预测。")
    L.append("")
    for coh, rc in results.items():
        L += [f"## {COHORT_NAMES[coh]}", "", "| 任务 | 模型 | n（阳性） | AUC（95% CI） | PR-AUC | 敏感度 / 特异度 | 校准截距 / 斜率 | Brier | 与主模型比较（DeLong） |",
              "|---|---|---|---|---|---|---|---|---|"]
        for task in tasks:
            rt = rc.get(task, {})
            for ref, m in rt.get("models", {}).items():
                if "auc" not in m:
                    L.append(f"| {TASK_NAMES.get(task, task)} | {ref} | {m.get('n', '—')} | {m.get('note', '—')} | | | | | |")
                    continue
                cal = rt.get("calibration", {}).get(ref, {})
                dl = rt.get("delong", {}).get(ref)
                comp = f"差 {fmt(dl['diff'])}（{fmt(dl['lo'])}–{fmt(dl['hi'])}），p={fmt_p(dl['p'])}" if dl and ref != main else ""
                ss = f"{fmt(m.get('sens'))} / {fmt(m.get('spec'))}" if "sens" in m else "—"
                L.append(f"| {TASK_NAMES.get(task, task)} | {ref}{'（主）' if ref == main else ''} | {m['n']}"
                         f"{'（' + str(m['n_pos']) + '）' if 'n_pos' in m else ''} | {fmt(m['auc'])}（{fmt(m['lo'])}–{fmt(m['hi'])}） | "
                         f"{fmt(m.get('pr_auc'))} | {ss} | {fmt(cal.get('intercept'), 2)} / {fmt(cal.get('slope'), 2)} | "
                         f"{fmt(cal.get('brier'))} | {comp} |")
        L.append("")
        rec = [(task, ref, c["recalibration"]) for task in tasks for ref, c in rc.get(task, {}).get("calibration", {}).items()
               if "recalibration" in c]
        if rec:
            L += ["**校准敏感性分析**（只用本队列数据调整整体偏移 / 截距）", "",
                  "| 任务 | 模型 | 截距调整量 | 调整后 Brier | 调整后校准斜率 |", "|---|---|---|---|---|"]
            for task, ref, r in rec:
                L.append(f"| {TASK_NAMES.get(task, task)} | {ref} | {fmt(r['shift'], 2)} | {fmt(r['after'].get('brier'))} | "
                         f"{fmt(r['after'].get('slope'), 2)} |")
            L.append("")
    L.append("ROC 与校准图见 figs/。")
    return "\n".join(L)


def main(argv=None) -> int:
    p = base_parser("v4 分类任务评价：AUC（DeLong）、PR、敏感度 / 特异度、DeLong 比较、校准（含外部截距重新校准）。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
