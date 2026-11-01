"""v3 最终预测：融合模型（5 折 × 种子集成）或后期融合（冻结的组合参数）评估内部测试集 / 外部队列（只运行一次）。

规则与 v1 / v2 相同（CLAUDE.md 6.5 / 15a）：必须加 ``--final``；每次运行在实验目录 final_evaluations.json 登记模型指纹，
同一模型对同一队列再次运行被拒绝（确需重跑加 --allow-rerun，报告注明）；阈值、风险截断一律使用交叉验证阶段由折外预测
确定的 postprocess.json。外部队列按中心分别报告并另报合并结果（MSI、EBV 阳性少，以合并结果为主要参考）；
所有结果另按模态可用性（两种都有 / 只有病理 / 只有 CT）分层报告。可以一起评估最终模型与各对照模型。

命令行：
    python -m gastric_mm.fusion.predict --config configs/server.yaml --exp fusion_main --cohort test --final
    python -m gastric_mm.fusion.predict --config configs/server.yaml --exp late_stack --cohort external --final

输出：<实验目录>/final_<队列>_<时间>/  predictions.csv、report.md、metrics.json（融合模型另有 attention.h5）
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..ct_mtl.predict import _ensemble_token
from ..models.path_model import load_model
from ..mtl.data import CTFeatureStore, PathFeatureStore
from ..mtl.experiment import (check_final_registry, exp_root, files_fingerprint, read_meta, register_final,
                              safe_evaluate, save_attention)
from ..mtl.fusion_model import load_fusion_model
from ..mtl.trainer import predict
from ..path_mtl.common import load_labels_and_split
from ..path_mtl.evaluate import merged, metrics_table_md
from ..path_mtl.predict import COHORTS, select_cohort
from ..training.bag_dataset import FeatureStore
from ..training.task_spec import TaskSpec
from ..training.trainer import predict as predict_v1
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from ..utils.log import setup_logging
from .common import GROUP_NAMES, NAMESPACE, SECTION, SourceExp, availability, group_of, wrap_stores
from .late import apply_task, pred_frame, side_by_side, task_features


def _ensemble_fusion(files, cfg, coh: pd.DataFrame, device, log):
    models = [load_fusion_model(f, map_location="cpu") for f in files]
    spec: TaskSpec = models[0][0].spec
    ex = models[0][1]
    pstore = PathFeatureStore(cfg, ex["path_encoders"], cache=True)
    cstore = CTFeatureStore(cfg, ex["ct_encoder"], list(ex["ct_rois"]), cache=True)
    av = availability(list(coh.study_id), pstore, cstore)
    n_miss = int((av.group == "none").sum())
    av = av[av.group != "none"]
    coh = coh[coh.study_id.isin(av.study_id)].reset_index(drop=True)
    if len(coh) == 0:
        raise UserError("该队列没有任何病例有病理或 CT 特征", "请先运行 30_v3_check.sh 查看该队列的模态可用性。")
    stores = wrap_stores(pstore, cstore, av)
    sec = cfg[SECTION]
    horizons = list(sec.get("horizons_months", [12, 36, 60]))
    preds, attn_sum = [], {}
    for m, extra in models:
        p, attn, _ = predict(m, extra, stores, coh, device, int(sec.train.batch_size), horizons)
        preds.append(p.set_index("study_id"))
        for sid, d in attn.items():
            acc = attn_sum.setdefault(sid, {})
            for role, a in d.items():
                acc[role] = acc.get(role, 0) + a / len(models)
    ens = (sum(preds) / len(preds)).reset_index()
    return ens, spec, coh, n_miss, attn_sum


def _ensemble_v1(src: SourceExp, cfg, coh: pd.DataFrame, device) -> pd.DataFrame:
    files = sorted(src.dir.glob("seed*/fold*/model.pt"))
    models = [load_model(f, map_location="cpu") for f in files]
    store = FeatureStore(cfg, models[0][1]["encoders"], cache=True)
    have = set(store.available(list(coh.study_id)))
    sub = coh[coh.study_id.isin(have)].reset_index(drop=True)
    if len(sub) == 0:
        return pd.DataFrame({"study_id": []})
    horizons = list(cfg.v1.get("horizons_months", [12, 36, 60]))
    preds = [predict_v1(m, ex, store, sub, device, int(cfg.v1.train.batch_size), horizons)[0].set_index("study_id")
             for m, ex in models]
    return (sum(preds) / len(preds)).reset_index()


def _late_predictions(exp: Path, meta: dict, cfg, coh: pd.DataFrame, device, args, log):
    lm = json.loads((exp / "late_model.json").read_text(encoding="utf-8"))
    spec = TaskSpec.from_dict(lm["spec"])
    srcs = {"path": SourceExp(cfg, "path", meta["sources"]["path"]), "ct": SourceExp(cfg, "ct", meta["sources"]["ct"])}
    for k, s in srcs.items():
        if s.fingerprint != meta["source_fingerprints"][k]:
            raise UserError(f"后期融合实验依赖的{s.label}实验 {s.name} 已被重新训练（指纹不同）",
                            "后期融合的组合参数是用原来的折外预测拟合的；请重新运行 32_v3_late_fusion.sh（新实验名）。")
    a = _ensemble_v1(srcs["path"], cfg, coh, device)
    files_c = sorted(srcs["ct"].dir.glob("seed*/fold*/model.pt"))
    try:
        b, _, _, _, _ = _ensemble_token(srcs["ct"].dir, files_c, cfg, coh, args, log)
    except UserError:                      # 该队列没有任何 CT 特征
        b = pd.DataFrame({"study_id": pd.Series([], dtype=str)})
    if len(a) == 0 and len(b) == 0:
        raise UserError("该队列没有任何病例有病理或 CT 特征", "请先运行 30_v3_check.sh 查看该队列的模态可用性。")
    both = side_by_side(a, b)
    preds = {t: apply_task(lm["method"], t, spec, task_features(t, spec, both, "_p"), task_features(t, spec, both, "_c"),
                           lm["models"][t]) for t in spec.all_tasks}
    ens = pred_frame(spec, both.study_id, preds)
    ens = ens.merge(both[["study_id", "has_path", "has_ct"]], on="study_id")
    n_miss = int((~coh.study_id.isin(ens.study_id)).sum())
    coh = coh[coh.study_id.isin(ens.study_id)].reset_index(drop=True)
    return ens, spec, coh, n_miss, {}


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    exp = exp_root(cfg, NAMESPACE) / args.exp
    if not (exp / "experiment.json").exists():
        raise UserError(f"找不到融合实验 {args.exp}", "请确认 --exp 与训练时一致，且使用同一个配置文件。")
    if not args.final:
        raise UserError("内部测试集和外部队列只能在模型定型后使用一次，本命令必须加 --final 才会运行",
                        "确认该模型已经定型（不会再根据测试结果修改）后，再加 --final 运行。开发期间请只看交叉验证报告。")
    if not (exp / "postprocess.json").exists():
        raise UserError("该实验的交叉验证尚未汇总完成（缺少 postprocess.json）", "请先跑完全部折，直到出现 cv_report.md。")
    meta = read_meta(exp)
    kind = meta.get("kind", "fusion")
    if kind == "late":
        files = [exp / "late_model.json"]
        srcs = [SourceExp(cfg, k, meta["sources"][k]) for k in ("path", "ct")]
        dep = [f for s in srcs for f in sorted(s.dir.glob("seed*/fold*/model.pt"))]
        fp = files_fingerprint(files, exp) + "-" + files_fingerprint(dep, exp.parent.parent)
    else:
        files = sorted(exp.glob("seed*/fold*/model.pt"))
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
    log.warning("【最终评估】融合实验 %s（%s），队列 %s，指纹 %s%s", args.exp, kind, args.cohort, fp, "（重跑）" if prev else "")
    df = load_labels_and_split(cfg, log)
    coh = select_cohort(df, args.cohort, cfg.manifest.internal_center).reset_index(drop=True)
    device = resolve_device(args.gpu)
    if kind == "late":
        ens, spec, coh, n_miss, attn = _late_predictions(exp, meta, cfg, coh, device, args, log)
        n_models = "后期融合（冻结参数）"
    else:
        ens, spec, coh, n_miss, attn = _ensemble_fusion(files, cfg, coh, device, log)
        n_models = f"{len(files)} 个融合模型集成"
    log.info("队列 %s：%d 例至少有一种模态%s", args.cohort, len(coh), f"（{n_miss} 例两种都没有）" if n_miss else "")
    thr = post.get("binary_thresholds_youden", {})
    for t in spec.binary:
        if t in thr and np.isfinite(thr[t]):
            ens[f"{t}_pred"] = (ens[f"{t}_prob"] >= thr[t]).astype(int)
    cuts = post.get("survival_risk_cutoffs_median", {})
    for ep in spec.survival:
        c = cuts.get(ep, {}).get("all")
        if c is not None:
            ens[f"{ep}_high_risk"] = (ens[f"{ep}_risk"] > c).astype(int)
    ens["modality_group"] = group_of(ens).to_numpy()
    ens.to_csv(out / "predictions.csv", index=False)
    if attn:
        save_attention(out / "attention.h5", attn)
    n_boot = int(cfg[SECTION].get("eval", {}).get("n_boot", 1000))
    full = merged(ens, coh.drop(columns=["fold"]))
    groups = [("全部", full)]
    if args.cohort == "external":
        groups = [(c, full[full["center"] == c]) for c in sorted(full["center"].unique())] + [("外部合并", full)]
    L = [f"# 最终评估：{args.exp} → {args.cohort}", "",
         f"- 时间：{ts}；模型：{n_models}（{kind}），指纹 {fp}",
         f"- 病例：{len(full)}（两种模态都没有的 {n_miss} 例未纳入）" + ("；**本次为重跑**" if prev else ""),
         "- 阈值与风险分组截断值来自交叉验证折外预测（postprocess.json），未使用本队列调整。",
         "- 外部队列中 MSI、EBV 阳性少，以“外部合并”结果为主要参考。", ""]
    all_metrics = {}
    for name, g in groups:
        rows = safe_evaluate(g, spec, cfg, n_boot=n_boot, thresholds=thr)
        all_metrics[name] = {"all": rows}
        cnt = g["modality_group"].value_counts().to_dict()
        L.append(metrics_table_md(rows, f"{name}（n={len(g)}；" + "、".join(
            f"{GROUP_NAMES[k]} {cnt.get(k, 0)}" for k in ("both", "path_only", "ct_only")) + "）"))
        for ep in spec.survival:
            col = f"{ep}_high_risk"
            if col in g:
                hi, lo = g[g[col] == 1], g[g[col] == 0]
                L.append(f"- {ep}：高风险组 {len(hi)} 例（事件 {int(np.nansum(hi[f'{ep}_event']))}），"
                         f"低风险组 {len(lo)} 例（事件 {int(np.nansum(lo[f'{ep}_event']))}）；KM 曲线与 log-rank 见 v4")
        L.append("")
        for gk in ("both", "path_only", "ct_only"):
            sub = g[g["modality_group"] == gk]
            if len(sub) >= 5:
                r2 = safe_evaluate(sub, spec, cfg, n_boot=n_boot, thresholds=thr)
                all_metrics[name][gk] = r2
                L.append(metrics_table_md([r for r in r2 if r["main"] or r["metric"] in ("宏平均AUC", "AUC", "C-index")],
                                          f"{name} · {GROUP_NAMES[gk]}（n={len(sub)}）"))
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    register_final(exp, reg, {"cohort": args.cohort, "models_fingerprint": fp, "time": ts, "dir": out.name,
                              "n": int(len(full)), "rerun": bool(prev)})
    log.info("最终评估完成：%s", out / "report.md")
    return 0


def main(argv=None) -> int:
    p = base_parser("v3 融合模型 / 后期融合最终预测：评估内部测试集或外部队列（只运行一次）。",
                    cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="融合实验名（fusion 下；后期融合实验也用本命令）")
    p.add_argument("--cohort", required=True, choices=COHORTS,
                   help="test=内部测试集；ext1/ext2/ext3=单个外部中心；external=全部外部中心（分别报告并合并）")
    p.add_argument("--final", action="store_true", help="确认模型已定型，进行最终评估（必填）")
    p.add_argument("--allow-rerun", action="store_true", help="允许对同一队列重跑（报告中会注明）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
