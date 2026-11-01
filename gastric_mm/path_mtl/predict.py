"""v1 最终预测：用 5 折（× 种子）模型集成预测内部测试集或外部队列。

规则（CLAUDE.md 5.2 / 6.5）：
- 内部测试集与外部队列只在模型定型后使用一次。必须加 ``--final`` 才会运行；
- 每次运行在实验目录的 final_evaluations.json 中登记（模型指纹、队列、时间）；
  同一模型对同一队列再次运行会被拒绝（防止"看了测试结果再回去改模型"），确需重跑须加 --allow-rerun，
  并会在报告中注明；
- 阈值、风险分组截断值一律使用交叉验证阶段由折外预测确定的 postprocess.json（冻结）。

命令行：
    python -m gastric_mm.path_mtl.predict --config configs/server.yaml --exp uni2h_abmil --cohort test --final
    python -m gastric_mm.path_mtl.predict --config configs/server.yaml --exp uni2h_abmil --cohort external --final

输出：<实验目录>/final_<队列>_<时间>/  predictions.csv、report.md、metrics.json、attention.h5
外部队列按中心分别报告，并另报外部合并结果。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import load_config
from ..models.path_model import load_model
from ..training.bag_dataset import FeatureStore
from ..training.task_spec import TaskSpec
from ..training.trainer import predict
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from .common import exp_root, load_labels_and_split
from .evaluate import evaluate, merged, metrics_table_md
from .train import save_attention

COHORTS = ("test", "ext1", "ext2", "ext3", "external")


def model_files(exp: Path) -> List[Path]:
    return sorted(exp.glob("seed*/fold*/model.pt"))


def models_fingerprint(files: List[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(f.parents[2]).as_posix().encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def select_cohort(df: pd.DataFrame, cohort: str, internal: str) -> pd.DataFrame:
    if cohort == "test":
        return df[(df["set"] == "test") & (df["center"] == internal)]
    if cohort == "external":
        return df[df["set"] == "external"]
    return df[(df["set"] == "external") & (df["center"] == cohort)]


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    exp = exp_root(cfg) / args.exp
    if not (exp / "experiment.json").exists():
        raise UserError(f"找不到实验 {args.exp}", "请确认 --exp 与训练时一致，且使用同一个配置文件。")
    if not args.final:
        raise UserError(
            "内部测试集和外部队列只能在模型定型后使用一次，本命令必须加 --final 才会运行",
            "确认该模型已经定型（不会再根据测试结果修改）后，再加 --final 运行。开发期间请只看交叉验证报告。",
        )
    if not (exp / "postprocess.json").exists():
        raise UserError("该实验的交叉验证尚未汇总完成（缺少 postprocess.json）",
                        "请先跑完全部折：python -m gastric_mm.path_mtl.train ... 直到出现 cv_report.md。")
    from ..utils.log import setup_logging
    files = model_files(exp)
    if not files:
        raise UserError("实验目录中没有模型文件", "请先完成交叉验证训练。")
    meta = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
    fp = models_fingerprint(files)
    reg_p = exp / "final_evaluations.json"
    reg = json.loads(reg_p.read_text(encoding="utf-8")) if reg_p.exists() else []
    prev = [r for r in reg if r["cohort"] == args.cohort and r["models_fingerprint"] == fp]
    if prev and not args.allow_rerun:
        raise UserError(
            f"该模型已于 {prev[0]['time']} 对队列 {args.cohort} 做过最终评估（结果在 {prev[0]['dir']}）",
            "最终评估只做一次。如确有必要重跑（例如程序出错），加 --allow-rerun，报告中会注明这是重跑。",
        )
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = exp / f"final_{args.cohort}_{ts}"
    i = 1
    while out.exists():
        i += 1
        out = exp / f"final_{args.cohort}_{ts}-{i}"
    out.mkdir(parents=True)
    log = setup_logging(out / "run.log")
    post = json.loads((exp / "postprocess.json").read_text(encoding="utf-8"))
    log.warning("【最终评估】实验 %s，队列 %s，模型 %d 个（指纹 %s）%s", args.exp, args.cohort, len(files), fp,
                "（重跑）" if prev else "")

    df = load_labels_and_split(cfg, log)
    coh = select_cohort(df, args.cohort, cfg.manifest.internal_center).reset_index(drop=True)
    device = resolve_device(args.gpu)
    models = [load_model(f, map_location="cpu") for f in files]
    spec: TaskSpec = models[0][0].spec
    encoders = models[0][1]["encoders"]
    store = FeatureStore(cfg, encoders, cache=True)
    have = set(store.available(list(coh.study_id)))
    n_miss = len(coh) - len(have)
    coh = coh[coh.study_id.isin(have)].reset_index(drop=True)
    log.info("队列 %s：%d 例有特征%s", args.cohort, len(coh), f"（{n_miss} 例缺少特征）" if n_miss else "")
    if len(coh) == 0:
        raise UserError("该队列没有可用的病例特征", "请确认外部中心的特征已导入（05b_import_trident.sh）。")
    horizons = list(cfg.v1.get("horizons_months", [12, 36, 60]))
    preds, attn_sum = [], {}
    for i, (m, extra) in enumerate(models):
        p, attn, _ = predict(m, extra, store, coh, device, int(cfg.v1.train.batch_size), horizons)
        preds.append(p.set_index("study_id"))
        for sid, a in attn.items():
            attn_sum[sid] = attn_sum.get(sid, 0) + a.mean(0, keepdims=True) / len(models)
    ens = sum(preds) / len(preds)
    ens = ens.reset_index()
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
    save_attention(out / "attention.h5", attn_sum)

    n_boot = int(cfg.v1.get("eval", {}).get("n_boot", 1000))
    full = merged(ens, coh.drop(columns=["fold"]))
    groups = [("全部", full)]
    if args.cohort == "external":
        groups = [(c, full[full["center"] == c]) for c in sorted(full["center"].unique())] + [("外部合并", full)]
    L = [f"# 最终评估：{args.exp} → {args.cohort}", "",
         f"- 时间：{ts}；模型：{len(files)} 个（{meta['encoders']}，{meta['aggregator']}），指纹 {fp}",
         f"- 病例：{len(full)}（缺少特征 {n_miss} 例未纳入）" + ("；**本次为重跑**" if prev else ""),
         "- 阈值与风险分组截断值来自交叉验证折外预测（postprocess.json），未使用本队列调整。", ""]
    all_metrics = {}
    for name, g in groups:
        rows = evaluate(g, spec, cfg, n_boot=n_boot, thresholds=thr)
        all_metrics[name] = rows
        L.append(metrics_table_md(rows, f"{name}（n={len(g)}）"))
        for ep in spec.survival:
            col = f"{ep}_high_risk"
            if col in g:
                hi = g[g[col] == 1]
                lo = g[g[col] == 0]
                L.append(f"- {ep}：高风险组 {len(hi)} 例（事件 {int(np.nansum(hi[f'{ep}_event']))}），"
                         f"低风险组 {len(lo)} 例（事件 {int(np.nansum(lo[f'{ep}_event']))}）；KM 曲线与 log-rank 见 v4")
        L.append("")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    reg.append({"cohort": args.cohort, "models_fingerprint": fp, "time": ts, "dir": out.name, "n": int(len(full)),
                "rerun": bool(prev)})
    reg_p.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("最终评估完成：%s", out / "report.md")
    return 0


def main(argv=None) -> int:
    p = base_parser("v1 最终预测：5 折模型集成，评估内部测试集或外部队列（只运行一次）。",
                    cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="实验名")
    p.add_argument("--cohort", required=True, choices=COHORTS,
                   help="test=内部测试集；ext1/ext2/ext3=单个外部中心；external=全部外部中心（分别报告并合并）")
    p.add_argument("--final", action="store_true", help="确认模型已定型，进行最终评估（必填）")
    p.add_argument("--allow-rerun", action="store_true", help="允许对同一队列重跑（报告中会注明）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
