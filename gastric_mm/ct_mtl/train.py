"""v2 CT 单模态多任务模型：交叉验证训练（只用内部训练集）。

输入：④ 提取的 CT 特征 ``<work_dir>/ct/features/<编码器>/<study_id>.h5``（tumor / regional 两个 ROI token）。
每个（随机种子, 折）：训练部分 = 其余各折，其中属于共用内层验证名单（v1/v2/v3 三方共用，见 gastric_mm/inner_val.py）的病例作内层验证用于早停；
留出折只用于折外预测。内部测试集与外部队列在这里根本不会被读取。

命令行：
    python -m gastric_mm.ct_mtl.train --config configs/server.yaml --exp ct_radimagenet --encoder radimagenet_r50 --gpu 0
    # 两张卡并行：同一个 --exp，分别 --folds 0,1,2 --gpu 0 与 --folds 3,4 --gpu 1
    # 编码器初筛：--profile screening（Lauren + OS + cT 辅助头，3 个随机种子）

输出（<output_dir>/ct_mtl/<实验名>/）：
    seed<s>/fold<k>/ model.pt、history.csv、case_ids.json、oof.csv、attention.h5（2.5D 各层注意力）、tokens.npz
    全部完成后：oof_predictions.csv、cv_metrics.json、cv_report.md、postprocess.json、oof_embeddings.h5
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..mtl.data import CTFeatureStore
from ..inner_val import SharedInnerVal
from ..mtl.experiment import aggregate, apply_profile, open_experiment, save_attention
from ..mtl.model import TokenMTLModel, save_token_model
from ..mtl.spec import spec_from_section
from ..mtl.trainer import predict, train_one
from ..path_mtl.common import load_labels_and_split, parse_int_list
from ..utils.cli import base_parser, run_main
from ..utils.device import resolve_device
from ..utils.errors import UserError
from ..utils.runio import write_csv

SECTION, NAMESPACE = "v2", "ct_mtl"


def describe(meta: dict):
    return [f"- CT 编码器：{meta.get('encoder')}；ROI：{', '.join(meta.get('rois', []))}；"
            f"层聚合：{meta.get('aggregator')}；预设：{meta.get('profile') or '无'}"]


def run_fold(cfg, spec, store, pool, fold, seed, fold_dir: Path, device, logger, inner_val: SharedInnerVal) -> None:
    sec = cfg[SECTION]
    held = pool[pool["fold"] == fold].reset_index(drop=True)
    rest = pool[pool["fold"] != fold].reset_index(drop=True)
    is_val = inner_val.mask(rest, seed, fold)
    lab_tr, lab_va = rest[~is_val].reset_index(drop=True), rest[is_val].reset_index(drop=True)
    prefix = f"[seed{seed} fold{fold}]"
    logger.info("%s 训练 %d 例，内层验证 %d 例，留出折 %d 例", prefix, len(lab_tr), len(lab_va), len(held))
    dims = store.dims
    factory = lambda n_bins: TokenMTLModel(dims, dict(sec.model), spec, n_bins)  # noqa: E731
    model, extra, hist = train_one(sec, spec, [store], factory, lab_tr, lab_va, device, seed, logger, prefix)
    extra.update(fold=fold, n_train=len(lab_tr), n_inner_val=len(lab_va), encoder=store.encoder, rois=store.roles)
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_token_model(model, fold_dir / "model.pt", extra)
    write_csv(hist, fold_dir / "history.csv")
    (fold_dir / "case_ids.json").write_text(json.dumps(
        {"train": list(lab_tr.study_id), "inner_val": list(lab_va.study_id), "heldout": list(held.study_id)},
        ensure_ascii=False), encoding="utf-8")
    horizons = list(sec.get("horizons_months", [12, 36, 60]))
    preds, attn, tokens = predict(model, extra, [store], held, device, int(sec.train.batch_size), horizons,
                                  keep_attention=bool(sec.get("save_attention", True)))
    preds["fold"] = fold
    preds["seed"] = seed
    preds.to_csv(fold_dir / "oof.csv", index=False)
    if attn:
        save_attention(fold_dir / "attention.h5", attn)
    np.savez_compressed(fold_dir / "tokens.npz", study_id=np.array(list(held.study_id)), tokens=tokens)
    (fold_dir / "done.json").write_text(json.dumps({"best_epoch": extra["best_epoch"],
                                                    "best_val_monitor": extra["best_val_monitor"]}), encoding="utf-8")
    logger.info("%s 完成：最佳 epoch %d", prefix, extra["best_epoch"])


def _run(args) -> int:
    cfg = apply_profile(load_config(args.config, args.set), SECTION, args.profile)
    if args.encoder:
        cfg[SECTION]["encoder"] = args.encoder
    sec = cfg[SECTION]
    spec = spec_from_section(cfg, SECTION)
    tag = f"train_folds-{args.folds or 'all'}".replace(",", "-")
    exp, log = open_experiment(cfg, SECTION, NAMESPACE, args.exp, tag, "token",
                               {"encoder": sec.encoder, "rois": list(sec.rois), "aggregator": sec.model.aggregator,
                                "spec": spec.to_dict()})
    df = load_labels_and_split(cfg, log)
    store = CTFeatureStore(cfg, sec.encoder, list(sec.rois), bool(sec.get("feature_cache", True)))
    train_ids = df[df["set"] == "train"]
    have = set(store.available(list(train_ids.study_id)))
    pool = train_ids[train_ids.study_id.isin(have)].reset_index(drop=True)
    n_miss = len(train_ids) - len(pool)
    log.info("CT 编码器 %s；ROI %s；任务：%s", store.encoder, ",".join(store.roles), ", ".join(spec.all_tasks))
    log.info("内部训练集 %d 例，其中有 CT 特征 %d 例%s", len(train_ids), len(pool),
             f"（{n_miss} 例缺少 CT 特征，不参与 CT 模型）" if n_miss else "")
    if len(pool) < 20:
        raise UserError(f"有 CT 特征的训练病例只有 {len(pool)} 例，无法训练",
                        "请确认已运行 CT 特征提取（24_ct_features.sh），且 v2.encoder 与提取时的编码器名一致。")
    dims = store.dims
    log.info("各 ROI 特征维数：%s；缺少某 ROI 的病例：%s", dims,
             {r: int(sum(1 for s in pool.study_id if r not in store.load(s))) for r in store.roles})
    n_folds = int(pool["fold"].max()) + 1
    folds = parse_int_list(args.folds) or list(range(n_folds))
    seeds = parse_int_list(args.seeds) or [int(s) for s in sec.get("seeds", [0])]
    bad = [k for k in folds if k < 0 or k >= n_folds]
    if bad:
        raise UserError(f"折号 {bad} 无效（划分文件共 {n_folds} 折，编号 0–{n_folds - 1}）")
    device = resolve_device(args.gpu)
    inner_val = SharedInnerVal(cfg, train_ids, log)
    log.info("设备：%s；折：%s；种子：%s", device, folds, seeds)
    for s in seeds:
        for k in folds:
            fd = exp / f"seed{s}" / f"fold{k}"
            if (fd / "done.json").exists() and not args.overwrite:
                log.info("[seed%d fold%d] 已完成，跳过（--overwrite 可重跑）", s, k)
                continue
            run_fold(cfg, spec, store, pool, k, s, fd, device, log, inner_val)
    all_seeds = sorted({int(p.name[4:]) for p in exp.glob("seed*") if p.is_dir()} | set(seeds))
    if not args.no_aggregate:
        aggregate(exp, cfg, SECTION, spec, pool, all_seeds, n_folds, log, describe)
    return 0


def main(argv=None) -> int:
    p = base_parser("v2 CT 单模态多任务模型：交叉验证训练（只用内部训练集）。", cases=False, workers=False, gpu=True)
    p.add_argument("--exp", required=True, help="实验名（如 ct_radimagenet）；结果在 <output_dir>/ct_mtl/<实验名>/")
    p.add_argument("--profile", default=None, help="使用配置中的预设（如 screening：CT 编码器初筛）")
    p.add_argument("--encoder", default=None, help="覆盖 v2.encoder（CT 特征的编码器名）")
    p.add_argument("--folds", default=None, help="只跑这些折（如 0,1,2），默认全部")
    p.add_argument("--seeds", default=None, help="随机种子（如 0,1,2），默认取配置 v2.seeds")
    p.add_argument("--overwrite", action="store_true", help="已完成的折也重新训练")
    p.add_argument("--no-aggregate", action="store_true", help="跑完不汇总")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
