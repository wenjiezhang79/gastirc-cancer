"""v1 病理多任务模型：交叉验证训练。

只使用划分文件中 set=train 的病例（内部测试集、外部队列在这里根本不会被读取）。
每个（随机种子, 折）：
  训练部分 = 其余各折；从中再按分层抽 v1.train.inner_val_fraction（默认 15%）作内层验证用于早停；
  被留出的那一折只用于折外预测，不参与任何训练决策。

命令行：
    python -m gastric_mm.path_mtl.train --config configs/server.yaml --exp uni2h_abmil --gpu 0
    # 两张卡并行：同一个 --exp，分别指定 --folds 0,1,2 --gpu 0 与 --folds 3,4 --gpu 1
    # 编码器初筛：--profile screening（只训练 Lauren + MSI，3 个随机种子）

输出（实验目录 <output_dir>/path_mtl/<实验名>/）：
    seed<s>/fold<k>/  model.pt、history.csv、case_ids.json、oof.csv、attention.h5、tokens.npz
    全部折完成后自动汇总：oof_predictions.csv、cv_metrics.json、cv_report.md、postprocess.json、
                          oof_attention.h5、oof_embeddings.h5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import Config, load_config
from ..split import _stratum_keys, greedy_assign
from ..training.bag_dataset import FeatureStore
from ..training.task_spec import TaskSpec
from ..training.trainer import predict, train_one
from ..models.path_model import save_model
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from ..utils.runio import write_csv
from .common import apply_profile, load_labels_and_split, open_experiment, parse_int_list
from .evaluate import evaluate, merged, metrics_table_md


def inner_split(lab: pd.DataFrame, cfg: Config, seed: int) -> np.ndarray:
    """训练部分内再分出内层验证集（分层）。返回布尔数组：True = 内层验证。"""
    frac = float(cfg.v1.train.get("inner_val_fraction", 0.15))
    keys = _stratum_keys(lab.reset_index(drop=True), list(cfg.split.stratify_by))
    grp = greedy_assign(keys, [1 - frac, frac], np.random.RandomState(seed))
    return grp == 1


def save_attention(path: Path, attn: Dict[str, np.ndarray]) -> None:
    import h5py
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for sid, a in attn.items():
            f.create_dataset(sid, data=a.astype(np.float16), compression="gzip")
    tmp.replace(path)


def run_fold(cfg: Config, spec: TaskSpec, store: FeatureStore, pool: pd.DataFrame, fold: int, seed: int,
             fold_dir: Path, device, logger) -> None:
    held = pool[pool["fold"] == fold].reset_index(drop=True)
    rest = pool[pool["fold"] != fold].reset_index(drop=True)
    is_val = inner_split(rest, cfg, seed * 100 + fold)
    lab_tr, lab_va = rest[~is_val].reset_index(drop=True), rest[is_val].reset_index(drop=True)
    prefix = f"[seed{seed} fold{fold}]"
    logger.info("%s 训练 %d 例，内层验证 %d 例，留出折 %d 例", prefix, len(lab_tr), len(lab_va), len(held))
    model, extra, hist = train_one(cfg, spec, store, lab_tr, lab_va, device, seed, logger, prefix)
    extra.update(fold=fold, n_train=len(lab_tr), n_inner_val=len(lab_va))
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, fold_dir / "model.pt", extra)
    write_csv(hist, fold_dir / "history.csv")
    (fold_dir / "case_ids.json").write_text(json.dumps(
        {"train": list(lab_tr.study_id), "inner_val": list(lab_va.study_id), "heldout": list(held.study_id)},
        ensure_ascii=False), encoding="utf-8")
    horizons = list(cfg.v1.get("horizons_months", [12, 36, 60]))
    preds, attn, tokens = predict(model, extra, store, held, device, int(cfg.v1.train.batch_size), horizons,
                                  keep_attention=bool(cfg.v1.get("save_attention", True)))
    preds["fold"] = fold
    preds["seed"] = seed
    preds.to_csv(fold_dir / "oof.csv", index=False)
    if attn:
        save_attention(fold_dir / "attention.h5", attn)
    np.savez_compressed(fold_dir / "tokens.npz", study_id=np.array(list(held.study_id)), tokens=tokens)
    (fold_dir / "done.json").write_text(json.dumps({"best_epoch": extra["best_epoch"],
                                                    "best_val_monitor": extra["best_val_monitor"]}), encoding="utf-8")
    logger.info("%s 完成：最佳 epoch %d", prefix, extra["best_epoch"])


def aggregate(exp: Path, cfg: Config, spec: TaskSpec, pool: pd.DataFrame, seeds: List[int], n_folds: int,
              logger) -> bool:
    """所有（种子, 折）都完成后汇总；否则返回 False。"""
    missing = [(s, k) for s in seeds for k in range(n_folds) if not (exp / f"seed{s}" / f"fold{k}" / "done.json").exists()]
    if missing:
        logger.info("尚有 %d 个（种子, 折）未完成：%s；全部完成后会自动汇总", len(missing),
                    ", ".join(f"seed{s}/fold{k}" for s, k in missing[:10]))
        return False
    n_boot = int(cfg.v1.get("eval", {}).get("n_boot", 1000))
    oof = pd.concat([pd.read_csv(exp / f"seed{s}" / f"fold{k}" / "oof.csv", dtype={"study_id": str})
                     for s in seeds for k in range(n_folds)], ignore_index=True)
    num = [c for c in oof.columns if c not in ("study_id", "fold", "seed")]
    avg = oof.groupby("study_id", sort=False).agg({**{c: "mean" for c in num}, "fold": "first"}).reset_index()
    avg.to_csv(exp / "oof_predictions.csv", index=False)
    df = merged(avg, pool.drop(columns=["fold"]))

    rows = evaluate(df, spec, cfg, n_boot=n_boot)
    # 各（种子, 折）分别计算，给出均值 ± 标准差
    per = {}
    from .evaluate import primary_value
    for (s, k), g in oof.groupby(["seed", "fold"]):
        d = merged(g, pool.drop(columns=["fold"]))
        for t in spec.all_tasks:
            per.setdefault(t, []).append(primary_value(d, spec, t))
    fold_stats = {t: (float(np.nanmean(v)), float(np.nanstd(v))) for t, v in per.items()}
    # 各种子分别的折外指标（种子间波动）
    seed_stats = {}
    for s, g in oof.groupby("seed"):
        d = merged(g, pool.drop(columns=["fold"]))
        seed_stats[int(s)] = {t: primary_value(d, spec, t) for t in spec.all_tasks}

    thresholds = {t: M.youden_threshold(df[t], df[f"{t}_prob"]) for t in spec.binary}
    cutoffs = {}
    for ep in spec.survival:
        cutoffs[ep] = {"all": float(np.nanmedian(df[f"{ep}_risk"]))}
        for pop in ("resected", "unresectable"):
            m = df["population"] == pop
            if m.sum():
                cutoffs[ep][pop] = float(np.nanmedian(df.loc[m, f"{ep}_risk"]))
    post = {"source": "内部训练集折外预测（种子平均）", "binary_thresholds_youden": thresholds,
            "survival_risk_cutoffs_median": cutoffs, "n_cases": int(len(df))}
    (exp / "postprocess.json").write_text(json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"oof": rows, "per_fold_mean_sd": fold_stats, "per_seed": seed_stats, "seeds": seeds,
               "n_folds": n_folds, "n_cases": int(len(df))}
    (exp / "cv_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    _merge_h5(exp, seeds[0], n_folds)
    (exp / "cv_report.md").write_text(_report(exp, cfg, spec, rows, fold_stats, seed_stats, post, len(df)),
                                      encoding="utf-8")
    logger.info("交叉验证汇总完成：%s", exp / "cv_report.md")
    for r in rows:
        if r["main"]:
            logger.info("  %s %s = %.3f（95%% CI %.3f–%.3f，n=%d）", r["task"], r["metric"], r["value"], r["lo"],
                        r["hi"], r["n"])
    return True


def _merge_h5(exp: Path, seed: int, n_folds: int) -> None:
    import h5py
    att = [exp / f"seed{seed}" / f"fold{k}" / "attention.h5" for k in range(n_folds)]
    if all(p.exists() for p in att):
        with h5py.File(exp / "oof_attention.tmp", "w") as out:
            for p in att:
                with h5py.File(p, "r") as f:
                    for sid in f:
                        out.create_dataset(sid, data=f[sid][()], compression="gzip")
        (exp / "oof_attention.tmp").replace(exp / "oof_attention.h5")
    with h5py.File(exp / "oof_embeddings.tmp", "w") as out:
        for k in range(n_folds):
            z = np.load(exp / f"seed{seed}" / f"fold{k}" / "tokens.npz")
            for sid, t in zip(z["study_id"], z["tokens"]):
                out.create_dataset(str(sid), data=t.astype(np.float32))
    (exp / "oof_embeddings.tmp").replace(exp / "oof_embeddings.h5")


def _report(exp, cfg, spec, rows, fold_stats, seed_stats, post, n) -> str:
    meta = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
    L = [f"# 交叉验证报告：{exp.name}", "",
         f"- 编码器：{', '.join(meta['encoders'])}；聚合器：{meta['aggregator']}；预设：{meta.get('profile') or '无'}",
         f"- 病例：内部训练集 {n} 例（有特征者）；随机种子：{list(seed_stats)}",
         f"- 生存损失：{cfg.v1.loss.survival_loss}；多任务加权：{cfg.v1.loss.weighting}",
         "- 折外预测：每例由未见过它的模型预测；多个种子时取平均。置信区间为 bootstrap 95% CI。",
         "- 注意：没有信号的任务，合并各折得到的折外 AUC 可能明显偏离 0.5（各折阳性率不同所致），"
         "此时请同时参考下方“各折均值”。", "",
         "## 折外指标（主结果）", "", metrics_table_md(rows),
         "## 各折主要指标（均值 ± 标准差，所有种子 × 折）", "", "| 任务 | 均值 | 标准差 |", "|---|---|---|"]
    for t, (m, s) in fold_stats.items():
        L.append(f"| {t} | {m:.3f} | {s:.3f} |")
    L += ["", "## 各随机种子的折外主要指标", "", "| 种子 | " + " | ".join(spec.all_tasks) + " |",
          "|---|" + "---|" * len(spec.all_tasks)]
    for s, d in seed_stats.items():
        L.append(f"| {s} | " + " | ".join(f"{d[t]:.3f}" for t in spec.all_tasks) + " |")
    L += ["", "## 由折外预测确定的阈值（之后冻结，用于测试集）", ""]
    for t, v in post["binary_thresholds_youden"].items():
        L.append(f"- {t}：约登阈值 {v:.3f}")
    for ep, d in post["survival_risk_cutoffs_median"].items():
        L.append(f"- {ep} 风险分组截断（中位数）：" + "，".join(f"{k} {v:.3f}" for k, v in d.items()))
    L.append("")
    return "\n".join(L)


def _run(args) -> int:
    cfg = apply_profile(load_config(args.config, args.set), args.profile)
    if args.encoders:
        cfg.v1["encoders"] = [e.strip() for e in args.encoders.split(",") if e.strip()]
    spec = TaskSpec.from_config(cfg)
    tag = f"train_folds-{args.folds or 'all'}"
    exp, log = open_experiment(cfg, args.exp, tag.replace(",", "-"))
    df = load_labels_and_split(cfg, log)
    store = FeatureStore(cfg, list(cfg.v1.encoders), bool(cfg.v1.get("feature_cache", True)))
    train_ids = df[df["set"] == "train"]
    have = set(store.available(list(train_ids.study_id)))
    pool = train_ids[train_ids.study_id.isin(have)].reset_index(drop=True)
    n_miss = len(train_ids) - len(pool)
    log.info("编码器 %s；任务：%s", ",".join(store.encoders), ", ".join(spec.all_tasks))
    log.info("内部训练集 %d 例，其中有特征 %d 例%s", len(train_ids), len(pool),
             f"（{n_miss} 例缺少特征，不参与病理模型）" if n_miss else "")
    if len(pool) < 20:
        raise UserError(f"有特征的训练病例只有 {len(pool)} 例，无法训练",
                        "请确认已运行 05b_import_trident.sh，且 v1.encoders 与导入的编码器名一致。")
    n_folds = int(pool["fold"].max()) + 1
    folds = parse_int_list(args.folds) or list(range(n_folds))
    seeds = parse_int_list(args.seeds) or [int(s) for s in cfg.v1.get("seeds", [0])]
    bad = [k for k in folds if k < 0 or k >= n_folds]
    if bad:
        raise UserError(f"折号 {bad} 无效（划分文件共 {n_folds} 折，编号 0–{n_folds - 1}）")
    device = resolve_device(args.gpu)
    log.info("设备：%s；折：%s；种子：%s", device, folds, seeds)
    for s in seeds:
        for k in folds:
            fd = exp / f"seed{s}" / f"fold{k}"
            if (fd / "done.json").exists() and not args.overwrite:
                log.info("[seed%d fold%d] 已完成，跳过（--overwrite 可重跑）", s, k)
                continue
            run_fold(cfg, spec, store, pool, k, s, fd, device, log)
    all_seeds = sorted({int(p.name[4:]) for p in exp.glob("seed*") if p.is_dir()} | set(seeds))
    if not args.no_aggregate:
        aggregate(exp, cfg, spec, pool, all_seeds, n_folds, log)
    return 0


def main(argv=None) -> int:
    p = base_parser("v1 病理多任务模型：交叉验证训练（只用内部训练集）。", cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="实验名（如 uni2h_abmil）；结果在 <output_dir>/path_mtl/<实验名>/")
    p.add_argument("--profile", default=None, help="使用配置中的预设（如 screening：编码器初筛）")
    p.add_argument("--encoders", default=None, help="覆盖 v1.encoders（逗号分隔；多个表示特征拼接）")
    p.add_argument("--folds", default=None, help="只跑这些折（如 0,1,2），默认全部")
    p.add_argument("--seeds", default=None, help="随机种子（如 0,1,2），默认取配置 v1.seeds")
    p.add_argument("--overwrite", action="store_true", help="已完成的折也重新训练")
    p.add_argument("--no-aggregate", action="store_true", help="跑完不汇总")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
