"""v4 第 1 步：分析用数据表（供 v4 各项分析与研究者用 R / SPSS 复核）。

命令行：
    python -m gastric_mm.evaluation.table --config configs/server.yaml

输出（<output_dir>/v4/table/<时间>/，只在服务器上生成，**不要放进代码仓库**）：
    analysis_table.csv      每例一行：study_id、队列、中心、人群、协变量、派生治疗分类、各终点（月数与事件）、
                            主模型的风险评分 / risk_z / 高低风险 / 三分位、各任务预测概率与按冻结阈值的预测类别
    predictions_<模型>.csv  各对照模型的预测（同样只含 study_id 与预测值）
    frozen.json             冻结量：风险评分标准化与三分位（内部训练集）、风险截断与阈值（postprocess.json）、一线方案合并规则
    sources.md              每个队列的预测来自哪次最终评估；开发模式（无最终评估）时注明
"""

from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from .common import (COHORT_NAMES, ModelPreds, add_firstline, add_risk_vars, frozen_risk_stats, load_covariates,
                     merged_table, v4, write_json)

KEEP_COV = ["study_id", "cohort", "center", "population", "age", "sex", "ecog", "stage_group", "ct_n_pos", "neoadj",
            "adj_chemo", "immuno", "immuno_setting", "tx_pattern", "first_line", "tx_firstline", "neoadj_regimen", "trg",
            "pcr", "mpr", "surgery_type", "d2", "adj_rt", "n_lines", "diag_year",
            "os_time", "os_event", "dfs_time", "dfs_event", "pfs_time", "pfs_event"]


def build(cfg, logger=None):
    """返回 (主模型合并表, {模型: ModelPreds}, 冻结量)。各分析模块共用。"""
    sec = v4(cfg)
    cov = load_covariates(cfg, logger)
    refs = [sec.models.main] + list(sec.models.get("compare", []) or [])
    models = {r: ModelPreds(cfg, r) for r in refs}
    main = models[sec.models.main]
    df = merged_table(cov, main)
    stats = frozen_risk_stats(df)
    df = add_risk_vars(df, main, stats)
    df, fl_map = add_firstline(df, int(sec.get("min_group_n", 30)))
    frozen = {"risk_standardization_and_tertiles": stats, "first_line_mapping": fl_map,
              "risk_cutoffs_median": main.post.get("survival_risk_cutoffs_median", {}),
              "binary_thresholds": main.post.get("binary_thresholds_youden", {}),
              "source": "均由内部训练集折外预测确定"}
    return df, models, frozen, cov


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "v4/table", args.run_name)
    log = run.logger
    df, models, frozen, cov = build(cfg, log)
    main_ref = v4(cfg).models.main
    pred_cols = [c for c in df.columns if c.endswith(("_prob", "_risk")) or "_p" in c and c.split("_p")[-1].isdigit()
                 or c.endswith("_S12m") or c.endswith("_S36m") or c.endswith("_S60m")]
    out = df[[c for c in KEEP_COV if c in df.columns] + [c for c in ("risk_z", "risk_high", "risk_tertile") if c in df]
             + pred_cols].copy()
    for t, thr in frozen["binary_thresholds"].items():
        if f"{t}_prob" in out:
            out[f"{t}_pred"] = (out[f"{t}_prob"] >= thr).astype(float).where(out[f"{t}_prob"].notna())
    out.to_csv(run.path("analysis_table.csv"), index=False, encoding="utf-8-sig")
    for ref, mp in models.items():
        if ref == main_ref:
            continue
        mp.pred.to_csv(run.path(f"predictions_{ref.replace('/', '__')}.csv"), index=False, encoding="utf-8-sig")
    write_json(run.path("frozen.json"), frozen)
    L = ["# v4 分析用数据表", "", f"- 主模型：{main_ref}；对照：{', '.join(r for r in models if r != main_ref) or '无'}", ""]
    dev = all(not mp.final_cohorts for mp in models.values())
    if dev:
        L += ["**开发模式**：尚无任何最终评估结果，只包含内部训练集折外预测。正式分析须在所有模型定型、运行 `--final` 之后进行。", ""]
    L += ["| 模型 | 队列 | 预测来源 |", "|---|---|---|"]
    for ref, mp in models.items():
        for coh, src in mp.sources.items():
            L.append(f"| {ref} | {COHORT_NAMES.get(coh, coh)} | {src} |")
    L += ["", "| 队列 | 例数 | 手术 | 不可切除 |", "|---|---|---|---|"]
    for coh, g in out.groupby("cohort", sort=False):
        L.append(f"| {COHORT_NAMES.get(coh, coh)} | {len(g)} | {int((g.population == 'resected').sum())} | "
                 f"{int((g.population == 'unresectable').sum())} |")
    L += ["", "analysis_table.csv 只含 study_id 与分析变量，供用 R / SPSS 复核；只在服务器上保存，不要放进代码仓库。", ""]
    run.write_text("\n".join(L), "sources.md")
    log.info("分析用数据表：%s（%d 例）", run.path("analysis_table.csv"), len(out))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 分析用数据表：合并各模型预测、协变量与冻结量（供 v4 分析与 R / SPSS 复核）。", cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
