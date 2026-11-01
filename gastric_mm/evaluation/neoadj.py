"""v4 探索：新辅助疗效预测（pCR / 主要病理缓解；docs/decisions.md 第十五节第 6 小节）。

**不改主模型。** 只在新辅助病例中，用主模型输出的预测值作特征拟合 L1 逻辑回归：
- 模型特征：风险评分（risk_z）与各任务预测概率（logit 变换：Lauren 肠型 / 弥漫型、HER2、MSI、EBV、PD-L1 CPS≥5、Claudin18.2）；
- 临床特征：cT3–4（ct_t34）、cN+；
- 三个模型都校正新辅助方案（化疗 / 放化疗 / 化疗 + 免疫）：临床、模型、临床 + 模型。

说明（实现时的调整，已在报告中注明）：方案原写"用融合模型的 CLS 嵌入"。但每例的折外嵌入来自不同的折模型，各折模型的嵌入空间不可比，
直接拼在一起拟合会失真；因此改用各折模型的**输出**（概率与风险评分），它们在各折、各队列之间尺度一致。

内部训练集：5 折交叉验证（与划分文件同折）给出折外预测与 AUC；再用全部内部训练集拟合一次，冻结后用于内部测试集与外部队列。
主要终点 = 专家判定的 pCR（定义见 v4.neoadj.pcr_definition）；内部训练集 pCR 阳性 < min_pos 时自动改用主要病理缓解（TRG 0–1）。

命令行：
    python -m gastric_mm.evaluation.neoadj --config configs/server.yaml

输出：<output_dir>/v4/neoadj/<时间>/report.md、results.json、neoadj_models.json（冻结的系数）、figs/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..ct_mtl.linear import fit_with_cv
from ..stats import classification as SC
from ..stats import plots as P
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from ..stats.plots import tr as ftr
from .common import COHORT_NAMES, EXTERNAL, cohort_groups, fc, fmt, fmt_p, setup_figures, v4, write_json
from .table import build

MODEL_FEATS = ["risk_z", "lauren_p0", "lauren_p1", "her2_pos_prob", "msi_h_prob", "ebv_pos_prob", "pdl1_cps5_prob",
               "cldn182_pos_prob"]
CLIN_FEATS = ["ct_t34", "ct_n_pos"]
REGIMENS = ["chemo", "chemoradio", "chemo_io"]
REGIMEN_NAMES = {"chemo": "化疗", "chemoradio": "放化疗", "chemo_io": "化疗 + 免疫"}


def features(g: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols = {}
    if kind in ("model", "combined"):
        for c in MODEL_FEATS:
            if c not in g:
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            cols[c] = v if c == "risk_z" else np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    if kind in ("clinical", "combined"):
        for c in CLIN_FEATS:
            if c in g:
                cols[c] = pd.to_numeric(g[c], errors="coerce")
    for r in REGIMENS[1:]:
        cols[f"regimen_{r}"] = (g["neoadj_regimen"] == r).astype(float)
    return pd.DataFrame(cols, index=g.index)


def fit(X: np.ndarray, y: np.ndarray, sec_lin, seed=0):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z = (X - mu) / sd
    r = fit_with_cv("binary", Z, y, {"w": np.ones(len(y))}, 1, n_lambdas=int(sec_lin.get("n_lambdas", 20)),
                    min_ratio=float(sec_lin.get("lambda_min_ratio", 0.01)), inner_folds=5, seed=seed)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "W": np.asarray(r["W"])[:, 0].tolist(), "b": float(np.asarray(r["b"])[0]),
            "lambda": r["lambda"], "n_nonzero": r["n_nonzero"]}


def predict(m: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(m["mu"])) / np.array(m["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(m["W"]) + m["b"])))


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    nc = sec.get("neoadj", {}) or {}
    run = Run(cfg, "v4/neoadj", args.run_name)
    log = run.logger
    figs = run.dir / "figs"
    figs.mkdir(exist_ok=True)
    setup_figures(cfg, log)
    df, models, frozen, cov = build(cfg, log)
    lab = cov[["study_id", "ct_t34"]] if "ct_t34" in cov else None
    if lab is not None and "ct_t34" not in df:
        df = df.merge(lab, on="study_id", how="left")
    neo = df[(pd.to_numeric(df.neoadj, errors="coerce") == 1)].copy()
    tr = neo[neo.cohort == "internal_train"]
    L = ["# v4 探索：新辅助疗效预测", "", f"- pCR 定义：{nc.get('pcr_definition', '专家判定')}",
         "- 特征：主模型输出的风险评分与各任务预测概率（不用嵌入：各折模型的嵌入空间不可比）；临床：cT3–4、cN+；均校正新辅助方案。",
         "- 内部训练集为 5 折交叉验证的折外 AUC；其他队列用全部内部训练集拟合后冻结的模型。结论只写“提示”。", ""]
    endpoint = str(nc.get("endpoint", "pcr"))
    n_pos = int((pd.to_numeric(tr[endpoint], errors="coerce") == 1).sum())
    if n_pos < int(nc.get("min_pos", 30)) and endpoint == "pcr":
        L.append(f"内部训练集 pCR 阳性只有 {n_pos} 例（< {nc.get('min_pos', 30)}），按预先规定改用主要病理缓解（TRG 0–1）为主要终点。")
        endpoint = "mpr"
    L += [f"- 主要终点：{'pCR' if endpoint == 'pcr' else '主要病理缓解（TRG 0–1）'}；新辅助病例："
          + "，".join(f"{COHORT_NAMES[c]} {len(g)} 例（阳性 {int((pd.to_numeric(g[endpoint], errors='coerce') == 1).sum())}）"
                     for c, g in cohort_groups(neo)), ""]
    lin = (cfg.get_path("v2.radiomics.model") or {})
    results, frozen_models = {"endpoint": endpoint}, {}
    kinds = ("clinical", "model", "combined")
    trv = tr[pd.to_numeric(tr[endpoint], errors="coerce").notna()]
    ytr = pd.to_numeric(trv[endpoint], errors="coerce").to_numpy(float)
    if (ytr == 1).sum() < 5 or (ytr == 0).sum() < 5:
        L.append("内部训练集阳性或阴性过少，无法拟合。")
        run.write_text("\n".join(L), "report.md")
        write_json(run.path("results.json"), results)
        return 0
    oof = {}
    for k in kinds:
        F = features(trv, k)
        keep = F.notna().all(1).to_numpy()
        pr = np.full(len(trv), np.nan)
        for f in sorted(trv.fold.dropna().unique()):
            tr_m = keep & (trv.fold != f).to_numpy()
            te_m = keep & (trv.fold == f).to_numpy()
            if te_m.sum() == 0 or len(np.unique(ytr[tr_m])) < 2:
                continue
            m = fit(F.to_numpy(float)[tr_m], ytr[tr_m], lin)
            pr[te_m] = predict(m, F.to_numpy(float)[te_m])
        oof[k] = pr
        frozen_models[k] = {"features": list(F.columns), **fit(F.to_numpy(float)[keep], ytr[keep], lin)}
    write_json(run.path("neoadj_models.json"), frozen_models)
    L += ["| 队列 | 模型 | n（阳性） | AUC（95% CI） | 与临床模型比较（DeLong） |", "|---|---|---|---|---|"]
    for coh, g in [("internal_train", trv)] + [(c, x) for c, x in cohort_groups(neo) if c != "internal_train"]:
        g = g[pd.to_numeric(g[endpoint], errors="coerce").notna()]
        y = pd.to_numeric(g[endpoint], errors="coerce").to_numpy(float)
        preds = {}
        for k in kinds:
            if coh == "internal_train":
                preds[k] = oof[k]
            else:
                F = features(g, k).reindex(columns=frozen_models[k]["features"])
                preds[k] = np.where(F.notna().all(1), predict(frozen_models[k], F.fillna(0).to_numpy(float)), np.nan)
        rc = {}
        for k in kinds:
            a = SC.delong_auc(y, preds[k])
            cmp = SC.delong_test(y, preds[k], preds["clinical"]) if k != "clinical" else None
            rc[k] = {"auc": a, "vs_clinical": cmp}
            L.append(f"| {COHORT_NAMES[coh]} | {k} | {int(np.isfinite(preds[k]).sum())}（{int((y == 1).sum())}） | "
                     f"{fmt(a['auc'])}（{fmt(a['lo'])}–{fmt(a['hi'])}） | "
                     + (f"差 {fmt(cmp['diff'])}，p={fmt_p(cmp['p'])}" if cmp else "") + " |")
        P.roc_plot([(k, y, preds[k]) for k in kinds], figs / f"roc_{endpoint}_{coh}", ftr("新辅助疗效", "Neoadjuvant response") + f" · {fc(coh)}")
        by_reg = {}
        for reg in REGIMENS:
            m = (g.neoadj_regimen == reg).to_numpy()
            if m.sum() >= 10:
                by_reg[reg] = SC.delong_auc(y[m], preds["combined"][m])
        rc["by_regimen_combined"] = by_reg
        results[coh] = rc
    L += ["", "**按新辅助方案分组（临床 + 模型）**", "", "| 队列 | 方案 | AUC（95% CI） |", "|---|---|---|"]
    for coh, rc in results.items():
        if coh == "endpoint":
            continue
        for reg, a in rc.get("by_regimen_combined", {}).items():
            L.append(f"| {COHORT_NAMES[coh]} | {REGIMEN_NAMES[reg]} | {fmt(a['auc'])}（{fmt(a['lo'])}–{fmt(a['hi'])}） |")
    L += ["", "ROC 图见 figs/。", ""]
    write_json(run.path("results.json"), results)
    run.write_text("\n".join(L), "report.md")
    log.info("新辅助疗效报告：%s", run.path("report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 探索：新辅助疗效预测（pCR / TRG 0–1），用主模型输出 + 临床变量的 L1 逻辑回归，不改主模型。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
