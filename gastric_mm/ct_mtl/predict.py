"""v2 最终预测：5 折（× 种子）模型集成，评估内部测试集或外部队列（只运行一次）。

规则与 v1 相同（CLAUDE.md 6.5 / 15a）：必须加 ``--final``；每次运行在实验目录 final_evaluations.json 登记模型指纹，
同一模型对同一队列再次运行被拒绝（确需重跑加 --allow-rerun，报告注明）；阈值、风险截断一律使用交叉验证阶段
由折外预测确定的 postprocess.json。外部队列按中心分别报告，并另报外部合并结果。
深度模型（kind=token）与影像组学基线（kind=radiomics）都用本命令。

命令行：
    python -m gastric_mm.ct_mtl.predict --config configs/server.yaml --exp ct_radimagenet --cohort test --final
    python -m gastric_mm.ct_mtl.predict --config configs/server.yaml --exp ct_radimagenet --cohort external --final

输出：<实验目录>/final_<队列>_<时间>/  predictions.csv、report.md、metrics.json（深度模型另有 attention.h5）
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..mtl.data import CTFeatureStore
from ..mtl.experiment import (check_final_registry, exp_root, files_fingerprint, read_meta, register_final,
                              safe_evaluate, save_attention)
from ..mtl.model import load_token_model
from ..mtl.trainer import predict
from ..path_mtl.common import load_labels_and_split
from ..path_mtl.evaluate import merged, metrics_table_md
from ..path_mtl.predict import COHORTS, select_cohort
from ..training.task_spec import TaskSpec
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from ..utils.log import setup_logging
from .radiomics_model import feature_columns, load_radiomics_table, predict_linear

NAMESPACE = "ct_mtl"


def _ensemble_token(exp: Path, files, cfg, coh: pd.DataFrame, args, log):
    models = [load_token_model(f, map_location="cpu") for f in files]
    spec: TaskSpec = models[0][0].spec
    ex = models[0][1]
    store = CTFeatureStore(cfg, ex["encoder"], list(ex["rois"]), cache=True)
    have = set(store.available(list(coh.study_id)))
    n_miss = len(coh) - len(have)
    coh = coh[coh.study_id.isin(have)].reset_index(drop=True)
    if len(coh) == 0:
        raise UserError("该队列没有可用的 CT 特征", "请确认该队列的 CT 特征已提取（24_ct_features.sh）。")
    device = resolve_device(args.gpu)
    horizons = list(cfg.v2.get("horizons_months", [12, 36, 60]))
    preds, attn_sum = [], {}
    for m, extra in models:
        p, attn, _ = predict(m, extra, [store], coh, device, int(cfg.v2.train.batch_size), horizons)
        preds.append(p.set_index("study_id"))
        for sid, d in attn.items():
            acc = attn_sum.setdefault(sid, {})
            for role, a in d.items():
                acc[role] = acc.get(role, 0) + a / len(models)
    ens = (sum(preds) / len(preds)).reset_index()
    return ens, spec, coh, n_miss, attn_sum


def _ensemble_radiomics(exp: Path, files, cfg, coh: pd.DataFrame, meta: dict):
    tab = load_radiomics_table(cfg, meta["radiomics_name"])
    names = feature_columns(tab)
    have = set(tab.study_id)
    n_miss = int((~coh.study_id.isin(have)).sum())
    coh = coh[coh.study_id.isin(have)].reset_index(drop=True)
    if len(coh) == 0:
        raise UserError("该队列没有影像组学特征", "请确认该队列已完成分割审核并提取了影像组学特征。")
    X = tab.set_index("study_id").loc[coh.study_id, names].to_numpy(float)
    horizons = list(cfg.v2.get("horizons_months", [12, 36, 60]))
    models = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    preds = [predict_linear(m, X, names, coh, horizons).set_index("study_id") for m in models]
    ens = (sum(preds) / len(preds)).reset_index()
    return ens, TaskSpec.from_dict(models[0]["spec"]), coh, n_miss, {}


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    exp = exp_root(cfg, NAMESPACE) / args.exp
    if not (exp / "experiment.json").exists():
        raise UserError(f"找不到实验 {args.exp}", "请确认 --exp 与训练时一致，且使用同一个配置文件。")
    if not args.final:
        raise UserError("内部测试集和外部队列只能在模型定型后使用一次，本命令必须加 --final 才会运行",
                        "确认该模型已经定型（不会再根据测试结果修改）后，再加 --final 运行。开发期间请只看交叉验证报告。")
    if not (exp / "postprocess.json").exists():
        raise UserError("该实验的交叉验证尚未汇总完成（缺少 postprocess.json）", "请先跑完全部折，直到出现 cv_report.md。")
    meta = read_meta(exp)
    kind = meta.get("kind", "token")
    files = sorted(exp.glob("seed*/fold*/model.pt" if kind == "token" else "seed*/fold*/model.json"))
    if not files:
        raise UserError("实验目录中没有模型文件", "请先完成交叉验证训练。")
    fp = files_fingerprint(files, exp)
    reg, prev = check_final_registry(exp, args.cohort, fp, args.allow_rerun)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = exp / f"final_{args.cohort}_{ts}"
    i = 1
    while out.exists():
        i += 1
        out = exp / f"final_{args.cohort}_{ts}-{i}"
    out.mkdir(parents=True)
    log = setup_logging(out / "run.log")
    post = json.loads((exp / "postprocess.json").read_text(encoding="utf-8"))
    log.warning("【最终评估】实验 %s（%s），队列 %s，模型 %d 个（指纹 %s）%s", args.exp, kind, args.cohort, len(files), fp,
                "（重跑）" if prev else "")
    df = load_labels_and_split(cfg, log)
    coh = select_cohort(df, args.cohort, cfg.manifest.internal_center).reset_index(drop=True)
    if kind == "token":
        ens, spec, coh, n_miss, attn = _ensemble_token(exp, files, cfg, coh, args, log)
    else:
        ens, spec, coh, n_miss, attn = _ensemble_radiomics(exp, files, cfg, coh, meta)
    log.info("队列 %s：%d 例有特征%s", args.cohort, len(coh), f"（{n_miss} 例缺少特征）" if n_miss else "")
    thr = post.get("binary_thresholds_youden", {})
    for t in spec.binary:
        if t in thr and np.isfinite(thr[t]):
            ens[f"{t}_pred"] = (ens[f"{t}_prob"] >= thr[t]).astype(int)
    cuts = post.get("survival_risk_cutoffs_median", {})
    for ep in spec.survival:
        c = cuts.get(ep, {}).get("all")
        if c is not None:
            ens[f"{ep}_high_risk"] = (ens[f"{ep}_risk"] > c).astype(int)
    ens.to_csv(out / "predictions.csv", index=False)
    if attn:
        save_attention(out / "attention.h5", attn)
    n_boot = int(cfg.v2.get("eval", {}).get("n_boot", 1000))
    full = merged(ens, coh.drop(columns=["fold"]))
    groups = [("全部", full)]
    if args.cohort == "external":
        groups = [(c, full[full["center"] == c]) for c in sorted(full["center"].unique())] + [("外部合并", full)]
    L = [f"# 最终评估：{args.exp} → {args.cohort}", "",
         f"- 时间：{ts}；模型：{len(files)} 个（{meta.get('encoder')}，{kind}），指纹 {fp}",
         f"- 病例：{len(full)}（缺少特征 {n_miss} 例未纳入）" + ("；**本次为重跑**" if prev else ""),
         "- 阈值与风险分组截断值来自交叉验证折外预测（postprocess.json），未使用本队列调整。", ""]
    all_metrics = {}
    for name, g in groups:
        rows = safe_evaluate(g, spec, cfg, n_boot=n_boot, thresholds=thr)
        all_metrics[name] = rows
        L.append(metrics_table_md(rows, f"{name}（n={len(g)}）"))
        for ep in spec.survival:
            col = f"{ep}_high_risk"
            if col in g:
                hi, lo = g[g[col] == 1], g[g[col] == 0]
                L.append(f"- {ep}：高风险组 {len(hi)} 例（事件 {int(np.nansum(hi[f'{ep}_event']))}），"
                         f"低风险组 {len(lo)} 例（事件 {int(np.nansum(lo[f'{ep}_event']))}）；KM 曲线与 log-rank 见 v4")
        L.append("")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    register_final(exp, reg, {"cohort": args.cohort, "models_fingerprint": fp, "time": ts, "dir": out.name,
                              "n": int(len(full)), "rerun": bool(prev)})
    log.info("最终评估完成：%s", out / "report.md")
    return 0


def main(argv=None) -> int:
    p = base_parser("v2 CT 模型最终预测：5 折模型集成，评估内部测试集或外部队列（只运行一次）。",
                    cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="实验名")
    p.add_argument("--cohort", required=True, choices=COHORTS,
                   help="test=内部测试集；ext1/ext2/ext3=单个外部中心；external=全部外部中心（分别报告并合并）")
    p.add_argument("--final", action="store_true", help="确认模型已定型，进行最终评估（必填）")
    p.add_argument("--allow-rerun", action="store_true", help="允许对同一队列重跑（报告中会注明）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
