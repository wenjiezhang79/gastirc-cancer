"""v3 融合模型：交叉验证训练（只用内部训练集）。

每个（随机种子 s, 折 k）：
- 病例池 = 内部训练集中至少有一种模态特征的病例；训练部分 = 其余各折，其中属于**三方共用内层验证名单**
  （gastric_mm/inner_val.py）的病例作内层验证用于早停；留出折只用于折外预测；
- 单模态初始化：病理分支加载 v1 实验（v3.init.path_exp）同种子、同折模型的聚合器，CT 分支加载 v2 实验
  （v3.init.ct_exp）同种子、同折模型的 ROI 分支；分支以 lr × branch_lr_mult（默认 0.1）微调；
- 防泄漏检查：来源模型训练过本折留出病例、折号或内层验证名单不一致时直接报错。
内部测试集与外部队列在这里根本不会被读取。

命令行：
    python -m gastric_mm.fusion.train --config configs/server.yaml --exp fusion_main --gpu 0
    # 两张卡并行：同一个 --exp，分别 --folds 0,1,2 --gpu 0 与 --folds 3,4 --gpu 1
    # 单任务融合模型（必做消融）：--single-task msi_h
    # 消融：--set v3.model.fusion=concat 等（见 scripts/33_v3_ablation.sh），每个设置用新的实验名

输出（<output_dir>/fusion/<实验名>/）：
    seed<s>/fold<k>/ model.pt、history.csv、case_ids.json、oof.csv、attention.h5、tokens.npz
                     （缺失鲁棒性：oof_drop_ct.csv、oof_drop_path.csv）
    全部完成后：oof_predictions.csv、cv_metrics.json、cv_report.md（另含按模态可用性分层、模态贡献、缺失鲁棒性）、
               postprocess.json、oof_embeddings.h5、cv_fusion_extra.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import _to_config, load_config
from ..inner_val import SharedInnerVal
from ..mtl.experiment import aggregate, apply_profile, open_experiment, read_meta, safe_evaluate, save_attention
from ..mtl.fusion_model import FusionModel, save_fusion_model
from ..mtl.spec import spec_from_section
from ..mtl.trainer import predict, train_one
from ..path_mtl.common import load_labels_and_split, parse_int_list
from ..path_mtl.evaluate import merged, metrics_table_md, primary_value
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from ..utils.runio import write_csv
from .common import (GROUP_NAMES, NAMESPACE, SECTION, availability, branch_config, check_source_encoders,
                     check_source_fold, group_of, load_branch_weights, make_stores, sources_from_cfg, wrap_stores)


def describe(meta: dict) -> List[str]:
    init = meta.get("init_sources", {}) or {}
    return [f"- 病理编码器：{', '.join(meta.get('path_encoders', []))}；CT 编码器：{meta.get('ct_encoder')}；"
            f"CT ROI：{', '.join(meta.get('ct_rois', []))}",
            f"- 融合方式：{meta.get('fusion')}；缺失模态：{meta.get('missing')}；"
            f"初始化：病理 ← {init.get('path') or '从头训练'}，CT ← {init.get('ct') or '从头训练'}；"
            f"单任务：{meta.get('single_task') or '否（多任务）'}；预设：{meta.get('profile') or '无'}"]


def apply_single_task(cfg, task: str):
    """只保留一个任务（结构与其余超参数不变）。"""
    cfg = copy.deepcopy(cfg)
    tc = cfg[SECTION].tasks
    found = {k: [task] if task in (tc.get(k) or []) else [] for k in ("multiclass", "binary", "regression", "survival")}
    if not any(found.values()):
        raise UserError(f"--single-task {task} 不在 v3.tasks 中", "请从 v3.tasks 已启用的任务中选择，如 lauren、msi_h、os。")
    cfg[SECTION]["tasks"] = _to_config(found)
    cfg[SECTION]["_single_task"] = task
    return cfg


def run_fold(cfg, spec, stores, pool: pd.DataFrame, fold: int, seed: int, fold_dir: Path, device, logger,
             inner_val: SharedInnerVal, sources: Dict, model_cfg: dict, dims: Dict[str, int], fold_ids: set,
             pstore, cstore) -> None:
    sec = cfg[SECTION]
    held = pool[pool["fold"] == fold].reset_index(drop=True)
    rest = pool[pool["fold"] != fold].reset_index(drop=True)
    is_val = inner_val.mask(rest, seed, fold)
    lab_tr, lab_va = rest[~is_val].reset_index(drop=True), rest[is_val].reset_index(drop=True)
    prefix = f"[seed{seed} fold{fold}]"
    shared = inner_val.ids(seed, fold)
    for src in sources.values():
        if src is not None:
            check_source_fold(src, seed, fold, set(held.study_id), fold_ids, shared)
            check_source_encoders(src, seed, fold, cfg, dims)
    logger.info("%s 训练 %d 例，内层验证 %d 例，留出折 %d 例（两种都有 %d / 只有病理 %d / 只有 CT %d）", prefix,
                len(lab_tr), len(lab_va), len(held), int((held.group == "both").sum()),
                int((held.group == "path_only").sum()), int((held.group == "ct_only").sum()))

    def factory(n_bins):
        m = FusionModel(dims, model_cfg, spec, n_bins)
        for kind, src in sources.items():
            if src is not None:
                n = load_branch_weights(m, src, seed, fold)
                logger.info("%s %s分支已从 %s（seed%d fold%d）初始化：%d 个参数张量", prefix,
                            "病理" if kind == "path" else "CT ", src.name, seed, fold, n)
        return m

    model, extra, hist = train_one(sec, spec, stores, factory, lab_tr, lab_va, device, seed, logger, prefix)
    extra.update(fold=fold, n_train=len(lab_tr), n_inner_val=len(lab_va), path_encoders=list(sec.path_encoders),
                 ct_encoder=str(sec.ct_encoder), ct_rois=list(sec.ct_rois),
                 init_sources={k: (s.name if s else None) for k, s in sources.items()})
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_fusion_model(model, fold_dir / "model.pt", extra)
    write_csv(hist, fold_dir / "history.csv")
    (fold_dir / "case_ids.json").write_text(json.dumps(
        {"train": list(lab_tr.study_id), "inner_val": list(lab_va.study_id), "heldout": list(held.study_id)},
        ensure_ascii=False), encoding="utf-8")
    horizons = list(sec.get("horizons_months", [12, 36, 60]))
    bs = int(sec.train.batch_size)
    preds, attn, tokens = predict(model, extra, stores, held, device, bs, horizons,
                                  keep_attention=bool(sec.get("save_attention", True)))
    preds["fold"] = fold
    preds["seed"] = seed
    preds.to_csv(fold_dir / "oof.csv", index=False)
    if attn:
        save_attention(fold_dir / "attention.h5", attn)
    np.savez_compressed(fold_dir / "tokens.npz", study_id=np.array(list(held.study_id)), tokens=tokens)
    if bool(sec.get("eval", {}).get("missing_robustness", True)):
        both = held[held.group == "both"].reset_index(drop=True)
        if len(both):
            av = pool[["study_id", "has_path", "has_ct"]]
            for tag, hide in (("drop_ct", list(cstore.roles)), ("drop_path", list(pstore.roles))):
                p, _, _ = predict(model, extra, wrap_stores(pstore, cstore, av, hide=hide), both, device, bs, horizons,
                                  keep_attention=False)
                p["fold"] = fold
                p["seed"] = seed
                p.to_csv(fold_dir / f"oof_{tag}.csv", index=False)
    (fold_dir / "done.json").write_text(json.dumps({"best_epoch": extra["best_epoch"],
                                                    "best_val_monitor": extra["best_val_monitor"]}), encoding="utf-8")
    logger.info("%s 完成：最佳 epoch %d", prefix, extra["best_epoch"])


def _seed_avg(frames: List[pd.DataFrame]) -> pd.DataFrame:
    oof = pd.concat(frames, ignore_index=True)
    num = [c for c in oof.columns if c not in ("study_id", "fold", "seed")]
    return oof.groupby("study_id", sort=False).agg({**{c: "mean" for c in num}, "fold": "first"}).reset_index()


def fusion_extra_report(exp: Path, cfg, spec, pool: pd.DataFrame, seeds: List[int], n_folds: int, logger) -> None:
    """在 cv_report.md 后追加：各折按模态可用性的例数、按模态可用性分层的指标、模态贡献、缺失鲁棒性。"""
    sec = cfg[SECTION]
    n_boot = int(sec.get("eval", {}).get("n_boot", 1000))
    lab = pool.drop(columns=["fold"])
    oof = pd.read_csv(exp / "oof_predictions.csv", dtype={"study_id": str})
    df = merged(oof, lab.drop(columns=[c for c in lab.columns if c.startswith("has_") or c == "group"]))
    df["group"] = group_of(oof).to_numpy()
    L = ["", "## 各折病例数（按模态可用性）", "", "| 折 | 两种都有 | 只有病理 | 只有 CT |", "|---|---|---|---|"]
    for k in range(n_folds):
        g = pool[pool.fold == k]
        L.append(f"| {k} | " + " | ".join(str(int((g.group == x).sum())) for x in ("both", "path_only", "ct_only")) + " |")
    L += ["", "## 按模态可用性分层的折外指标", ""]
    strat = {}
    for gname in ("both", "path_only", "ct_only"):
        g = df[df.group == gname]
        if len(g) < 5:
            L.append(f"- {GROUP_NAMES[gname]}：{len(g)} 例，过少，不单独报告")
            continue
        rows = safe_evaluate(g, spec, cfg, n_boot=n_boot)
        strat[gname] = rows
        L.append(metrics_table_md([r for r in rows if r["main"] or r["metric"] in ("宏平均AUC", "AUC", "C-index")],
                                  f"{GROUP_NAMES[gname]}（n={len(g)}）"))
    extra = {"stratified": strat}
    if "attn_mod_path" in oof:
        L += ["", "## 模态贡献（CLS 对病理 / CT token 的注意力占比，种子平均）", "",
              "注意力占比只说明模型在融合时「看」哪种模态更多，不等于因果贡献；任务层面的贡献见下方缺失鲁棒性。", "",
              "| 分组 | 例数 | 病理占比 均值 | CT 占比 均值 |", "|---|---|---|---|"]
        contrib = {}
        for gname in ("both", "path_only", "ct_only"):
            g = df[df.group == gname]
            if len(g):
                mp, mc = float(g.attn_mod_path.mean()), float(g.attn_mod_ct.mean())
                contrib[gname] = {"path": mp, "ct": mc}
                L.append(f"| {GROUP_NAMES[gname]} | {len(g)} | {mp:.3f} | {mc:.3f} |")
        extra["attention_share"] = contrib
    drop = {}
    for tag in ("drop_ct", "drop_path"):
        files = [exp / f"seed{s}" / f"fold{k}" / f"oof_{tag}.csv" for s in seeds for k in range(n_folds)]
        if all(f.exists() for f in files):
            drop[tag] = merged(_seed_avg([pd.read_csv(f, dtype={"study_id": str}) for f in files]),
                               lab.drop(columns=[c for c in lab.columns if c.startswith("has_") or c == "group"]))
    if len(drop) == 2:
        full = df[df.study_id.isin(drop["drop_ct"].study_id)]
        L += ["", f"## 缺失鲁棒性（两种都有的 {len(full)} 例：预测时人为去掉一种模态）", "",
              "| 任务 | 两种都用 | 去掉 CT | 去掉病理 |", "|---|---|---|---|"]
        rob = {}
        for t in spec.all_tasks:
            v = [primary_value(d, spec, t) for d in (full, drop["drop_ct"], drop["drop_path"])]
            rob[t] = dict(zip(("full", "drop_ct", "drop_path"), v))
            L.append(f"| {t} | " + " | ".join("—" if not np.isfinite(x) else f"{x:.3f}" for x in v) + " |")
        extra["missing_robustness"] = rob
    L.append("")
    with open(exp / "cv_report.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    (exp / "cv_fusion_extra.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=float),
                                              encoding="utf-8")
    logger.info("已追加按模态可用性分层、模态贡献与缺失鲁棒性：%s", exp / "cv_report.md")


def _run(args) -> int:
    cfg = apply_profile(load_config(args.config, args.set), SECTION, args.profile)
    if args.single_task:
        cfg = apply_single_task(cfg, args.single_task)
    sec = cfg[SECTION]
    spec = spec_from_section(cfg, SECTION)
    sources = sources_from_cfg(cfg)
    if bool(sec.train.get("freeze_branches", False)) and not all(sources.values()):
        raise UserError("v3.train.freeze_branches = true 但没有同时设置 v3.init.path_exp 与 v3.init.ct_exp",
                        "冻结分支的消融需要从 v1 / v2 初始化（否则分支是随机的）。")
    seeds = parse_int_list(args.seeds) or [int(s) for s in sec.get("seeds", [0])]
    model_cfg, init_names = branch_config(cfg, sources, seeds[0])
    src_fp = {k: (s.fingerprint if s else None) for k, s in sources.items()}
    tag = f"train_folds-{args.folds or 'all'}".replace(",", "-")
    exp, log = open_experiment(cfg, SECTION, NAMESPACE, args.exp, tag, "fusion", {
        "path_encoders": list(sec.path_encoders), "ct_encoder": str(sec.ct_encoder), "ct_rois": list(sec.ct_rois),
        "fusion": sec.model.fusion, "missing": sec.model.missing, "init_sources": init_names,
        "init_fingerprints": src_fp, "branch_model_cfg": {"path": model_cfg.get("path"), "ct": model_cfg.get("ct")},
        "single_task": sec.get("_single_task")})
    meta = read_meta(exp)
    if meta.get("init_fingerprints") != src_fp:
        raise UserError(f"实验 {args.exp} 第一次运行时使用的 v1 / v2 来源实验与现在不同（同名实验被重新训练过？）",
                        "请换一个新的融合实验名（--exp）。")
    df = load_labels_and_split(cfg, log)
    pstore, cstore = make_stores(cfg)
    train_ids = df[df["set"] == "train"].reset_index(drop=True)
    av = availability(list(train_ids.study_id), pstore, cstore)
    pool = train_ids.merge(av, on="study_id", how="left")
    n_none = int((pool.group == "none").sum())
    pool = pool[pool.group != "none"].reset_index(drop=True)
    cnt = pool.group.value_counts().to_dict()
    log.info("任务：%s", ", ".join(spec.all_tasks))
    log.info("内部训练集 %d 例：两种都有 %d、只有病理 %d、只有 CT %d%s", len(train_ids), cnt.get("both", 0),
             cnt.get("path_only", 0), cnt.get("ct_only", 0), f"；两种都没有 {n_none} 例（不进模型）" if n_none else "")
    if len(pool) < 20:
        raise UserError(f"有特征的训练病例只有 {len(pool)} 例，无法训练",
                        "请先运行 30_v3_check.sh 查看模态可用性，确认 v3.path_encoders / v3.ct_encoder 与已有特征一致。")
    # 读取一例以确定维数
    first_p = pool.loc[pool.has_path == 1, "study_id"]
    if len(first_p):
        pstore.load(first_p.iloc[0])
    dims = {}
    if len(first_p):
        dims.update(pstore.dims)
    if (pool.has_ct == 1).any():
        dims.update(cstore.dims)
    if "path" not in dims or not any(r in dims for r in cstore.roles):
        raise UserError("病例池中缺少一种模态的全部特征，无法训练融合模型", "请先运行 30_v3_check.sh 查看模态可用性。")
    dims = {r: dims[r] for r in ["path"] + list(cstore.roles)}
    stores = wrap_stores(pstore, cstore, pool)
    n_folds = int(df.loc[df.set == "train", "fold"].max()) + 1
    folds = parse_int_list(args.folds) or list(range(n_folds))
    bad = [k for k in folds if k < 0 or k >= n_folds]
    if bad:
        raise UserError(f"折号 {bad} 无效（划分文件共 {n_folds} 折，编号 0–{n_folds - 1}）")
    device = resolve_device(args.gpu)
    log.info("设备：%s；折：%s；种子：%s；融合方式 %s（缺失 %s）", device, folds, seeds, sec.model.fusion, sec.model.missing)
    inner_val = SharedInnerVal(cfg, train_ids, log)
    for s in seeds:
        for k in folds:
            fd = exp / f"seed{s}" / f"fold{k}"
            if (fd / "done.json").exists() and not args.overwrite:
                log.info("[seed%d fold%d] 已完成，跳过（--overwrite 可重跑）", s, k)
                continue
            fold_ids = set(train_ids.loc[train_ids.fold == k, "study_id"])
            run_fold(cfg, spec, stores, pool, k, s, fd, device, log, inner_val, sources, model_cfg, dims, fold_ids,
                     pstore, cstore)
    all_seeds = sorted({int(p.name[4:]) for p in exp.glob("seed*") if p.is_dir()} | set(seeds))
    if not args.no_aggregate:
        if aggregate(exp, cfg, SECTION, spec, pool, all_seeds, n_folds, log, describe):
            fusion_extra_report(exp, cfg, spec, pool, all_seeds, n_folds, log)
    return 0


def main(argv=None) -> int:
    p = base_parser("v3 融合模型（病理 + CT）：交叉验证训练（只用内部训练集）。", cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="实验名（如 fusion_main）；结果在 <output_dir>/fusion/<实验名>/")
    p.add_argument("--profile", default=None, help="使用配置中的预设 v3.profiles.<名>")
    p.add_argument("--single-task", default=None, help="只训练这一个任务（单任务融合模型，结构与超参数不变），如 msi_h")
    p.add_argument("--folds", default=None, help="只跑这些折（如 0,1,2），默认全部")
    p.add_argument("--seeds", default=None, help="随机种子（如 0,1,2），默认取配置 v3.seeds")
    p.add_argument("--overwrite", action="store_true", help="已完成的折也重新训练")
    p.add_argument("--no-aggregate", action="store_true", help="跑完不汇总")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
