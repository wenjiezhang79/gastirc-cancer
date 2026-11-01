"""影像组学基线模型（v2 CT 分支的简单基线）：在与深度模型相同的交叉验证折上，每个任务一个 L1 线性模型。

输入：⑤ 提取的影像组学特征表 ``<work_dir>/ct/radiomics/<名称>/features.csv``（每例一行，只含研究编号）。
每折（只用该折的训练部分）：预处理（剔除缺失/常数 → 中位数填补 → z 分数 → 相关性去冗余）→ 每个任务
用内层 K 折交叉验证在 λ 路径上选 λ → 重拟合 → 预测留出折。生存任务为 L1 Cox，按人群分层（同深度模型：
手术与不可切除各自基线、共享系数），Breslow 基线给出各时间点生存概率。
输出格式与深度模型相同（oof_predictions.csv、cv_report.md、postprocess.json），可直接参与编码器比较。

命令行：
    python -m gastric_mm.ct_mtl.radiomics_model --config configs/server.yaml --exp radiomics_bw25

输出（<output_dir>/ct_mtl/<实验名>/）：seed0/fold<k>/model.json、oof.csv；汇总文件同深度模型；
另有 radiomics_report.md（各任务各折选中的特征）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..mtl.experiment import aggregate, open_experiment
from ..mtl.spec import spec_from_section
from ..path_mtl.common import load_labels_and_split, parse_int_list
from ..paths import WorkPaths
from ..training.bag_dataset import population_index
from ..training.task_spec import TaskSpec
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from . import linear as LIN

SECTION, NAMESPACE = "v2", "ct_mtl"
ID_COLS = ("study_id", "mask_source")


def load_radiomics_table(cfg: Config, name: str) -> pd.DataFrame:
    p = WorkPaths(cfg).ct_radiomics(name) / "features.csv"
    if not p.exists():
        raise UserError(f"找不到影像组学特征表（{name}/features.csv）",
                        "请先在 pyradiomics 环境中运行 25_radiomics.sh 提取影像组学特征。")
    df = pd.read_csv(p, dtype={"study_id": str})
    srcs = list(cfg.v2.radiomics.get("mask_sources", ["manual", "auto_reviewed"]) or [])
    if "mask_source" in df and srcs:
        df = df[df["mask_source"].isin(srcs) | df["mask_source"].isna()]
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _task_data(lab: pd.DataFrame, spec: TaskSpec, task: str, max_pos_w: float):
    """返回 (kind, y, aux, 有标签的行掩码, 输出维数)。"""
    if task in spec.multiclass:
        y = lab[task].to_numpy(float)
        return "multiclass", y, {}, np.isfinite(y), spec.multiclass[task]
    if task in spec.binary:
        y = lab[task].to_numpy(float)
        ok = np.isfinite(y)
        n1, n0 = float((y[ok] == 1).sum()), float((y[ok] == 0).sum())
        pw = float(np.clip(n0 / n1, 1.0, max_pos_w)) if n1 > 0 else 1.0
        return "binary", y, {"w": np.where(y == 1, pw, 1.0)}, ok, 1
    if task in spec.regression:
        y = lab[task].to_numpy(float)
        return "regression", y, {}, np.isfinite(y), 1
    tm, ev = lab[f"{task}_time"].to_numpy(float), lab[f"{task}_event"].to_numpy(float)
    pop = population_index(lab) if spec.n_pops(task) > 1 else np.zeros(len(lab), int)
    ok = np.isfinite(tm) & np.isfinite(ev) & (pop >= 0)
    return "survival", tm, {"time": tm, "event": np.nan_to_num(ev), "strata": pop}, ok, 1


def fit_fold(Xtr_raw: np.ndarray, names: List[str], lab_tr: pd.DataFrame, spec: TaskSpec, cfg: Config, seed: int) -> dict:
    mc = cfg.v2.radiomics.model
    pre = LIN.Preprocessor(float(mc.get("corr_threshold", 0.9))).fit(Xtr_raw, names)
    if not pre.names:
        raise UserError("预处理后没有剩下任何影像组学特征", "请检查特征表（是否大部分为空或常数）。")
    X = pre.transform(Xtr_raw)
    tasks: Dict[str, dict] = {}
    for t in spec.all_tasks:
        kind, y, aux, ok, n_out = _task_data(lab_tr, spec, t, float(cfg.v2.loss.get("max_pos_weight", 10)))
        if ok.sum() < 20 or (kind in ("binary", "multiclass") and len(np.unique(y[ok])) < 2) or \
                (kind == "survival" and aux["event"][ok].sum() < 5):
            tasks[t] = {"kind": kind, "skipped": "有标签的病例或事件太少"}
            continue
        r = LIN.fit_with_cv(kind, X[ok], y[ok] if kind != "survival" else y[ok], LIN.subset_aux(aux, ok), n_out,
                            int(mc.get("n_lambdas", 20)), float(mc.get("lambda_min_ratio", 0.01)),
                            int(mc.get("inner_folds", 5)), int(mc.get("max_iter", 500)), seed)
        d = {"kind": kind, "W": r["W"].tolist(), "b": r["b"].tolist(), "lambda": r["lambda"],
             "n_nonzero": r["n_nonzero"],
             "selected": [pre.names[i] for i in np.flatnonzero(np.abs(r["W"]).sum(1) > 0)]}
        if kind == "survival":
            lp = X[ok] @ r["W"][:, 0]
            base = LIN.breslow(lp, aux["time"][ok], aux["event"][ok], aux["strata"][ok])
            d["baseline"] = {str(s): [ts.tolist(), H.tolist()] for s, (ts, H) in base.items()}
        tasks[t] = d
    return {"format": "gastric_mm.radiomics_linear.v2", "pre": pre.to_dict(), "tasks": tasks, "spec": spec.to_dict()}


def predict_linear(model: dict, X_raw: np.ndarray, names: List[str], lab: pd.DataFrame, horizons) -> pd.DataFrame:
    spec = TaskSpec.from_dict(model["spec"])
    pre = LIN.Preprocessor.from_dict(model["pre"], names)
    X = pre.transform(X_raw)
    out = pd.DataFrame({"study_id": list(lab["study_id"])})
    pop = population_index(lab)
    for t in spec.all_tasks:
        d = model["tasks"][t]
        if "skipped" in d:
            if t in spec.multiclass:
                for c in range(spec.multiclass[t]):
                    out[f"{t}_p{c}"] = np.nan
            elif t in spec.binary:
                out[f"{t}_prob"] = np.nan
            elif t in spec.regression:
                out[f"{t}_pred"] = np.nan
            else:
                out[f"{t}_risk"] = np.nan
                for h in horizons:
                    out[f"{t}_S{int(h)}m"] = np.nan
            continue
        W, b = np.array(d["W"]), np.array(d["b"])
        z = X @ W + b
        if t in spec.multiclass:
            z = z - z.max(1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(1, keepdims=True)
            for c in range(P.shape[1]):
                out[f"{t}_p{c}"] = P[:, c]
        elif t in spec.binary:
            out[f"{t}_prob"] = LIN._sigmoid(z[:, 0])
        elif t in spec.regression:
            out[f"{t}_pred"] = z[:, 0]
        else:
            lp = X @ W[:, 0]
            out[f"{t}_risk"] = lp
            strata = pop if spec.n_pops(t) > 1 else np.zeros(len(lab), int)
            base = {int(s): (np.array(v[0]), np.array(v[1])) for s, v in d["baseline"].items()}
            S = LIN.survival_from_breslow(lp, strata, base, horizons)
            for j, h in enumerate(horizons):
                out[f"{t}_S{int(h)}m"] = S[:, j]
    return out


def describe(meta: dict):
    return [f"- 影像组学基线：特征表 {meta.get('radiomics_name')}（{meta.get('n_features')} 个特征）；"
            "每任务 L1 线性模型（λ 由内层交叉验证选择）"]


def _report(exp: Path, spec: TaskSpec, n_folds: int) -> str:
    L = ["# 影像组学基线：各任务选中的特征", "", "| 任务 | 各折非零特征数 | 选中次数最多的特征（次数/折数） |", "|---|---|---|"]
    models = [json.loads((exp / "seed0" / f"fold{k}" / "model.json").read_text(encoding="utf-8")) for k in range(n_folds)]
    for t in spec.all_tasks:
        ds = [m["tasks"][t] for m in models]
        if all("skipped" in d for d in ds):
            L.append(f"| {t} | 未拟合（{ds[0]['skipped']}） | |")
            continue
        cnt = Counter(f for d in ds for f in d.get("selected", []))
        top = "，".join(f"{f}（{c}/{n_folds}）" for f, c in cnt.most_common(8))
        L.append(f"| {t} | {', '.join(str(d.get('n_nonzero', '—')) for d in ds)} | {top} |")
    L.append("")
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.v2.radiomics["name"] = args.name
    spec = spec_from_section(cfg, SECTION)
    tab = load_radiomics_table(cfg, cfg.v2.radiomics.name)
    names = feature_columns(tab)
    exp, log = open_experiment(cfg, SECTION, NAMESPACE, args.exp, "radiomics", "radiomics",
                               {"radiomics_name": cfg.v2.radiomics.name, "n_features": len(names),
                                "encoder": f"radiomics:{cfg.v2.radiomics.name}", "rois": ["tumor_mask"],
                                "aggregator": "lasso", "spec": spec.to_dict()})
    df = load_labels_and_split(cfg, log)
    train_ids = df[df["set"] == "train"]
    pool = train_ids[train_ids.study_id.isin(set(tab.study_id))].reset_index(drop=True)
    log.info("影像组学特征表 %s：%d 例、%d 个特征；内部训练集中有特征 %d 例（共 %d 例）",
             cfg.v2.radiomics.name, len(tab), len(names), len(pool), len(train_ids))
    if len(pool) < 30:
        raise UserError(f"有影像组学特征的训练病例只有 {len(pool)} 例，无法建模",
                        "影像组学需要肿瘤分割（manual 或 auto_reviewed）；请先完成分割与审核后再提取。")
    feat = tab.set_index("study_id")[names]
    n_folds = int(pool["fold"].max()) + 1
    folds = parse_int_list(args.folds) or list(range(n_folds))
    horizons = list(cfg.v2.get("horizons_months", [12, 36, 60]))
    seed = 0
    for k in folds:
        fd = exp / f"seed{seed}" / f"fold{k}"
        if (fd / "done.json").exists() and not args.overwrite:
            log.info("[fold%d] 已完成，跳过", k)
            continue
        tr = pool[pool["fold"] != k].reset_index(drop=True)
        held = pool[pool["fold"] == k].reset_index(drop=True)
        log.info("[fold%d] 训练 %d 例，留出折 %d 例", k, len(tr), len(held))
        model = fit_fold(feat.loc[tr.study_id].to_numpy(float), names, tr, spec, cfg, seed * 100 + k)
        for t, d in model["tasks"].items():
            log.info("[fold%d]   %s：%s", k, t, d["skipped"] if "skipped" in d else
                     f"λ={d['lambda']:.4g}，非零特征 {d['n_nonzero']} 个")
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "model.json").write_text(json.dumps(model), encoding="utf-8")
        preds = predict_linear(model, feat.loc[held.study_id].to_numpy(float), names, held, horizons)
        preds["fold"] = k
        preds["seed"] = seed
        preds.to_csv(fd / "oof.csv", index=False)
        (fd / "done.json").write_text("{}", encoding="utf-8")
    if aggregate(exp, cfg, SECTION, spec, pool, [seed], n_folds, log, describe):
        (exp / "radiomics_report.md").write_text(_report(exp, spec, n_folds), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    p = base_parser("影像组学基线：每个任务一个 L1 线性模型（与深度模型相同的交叉验证折）。",
                    cases=False, workers=False)
    p.add_argument("--exp", required=True, help="实验名（如 radiomics_bw25）；结果在 <output_dir>/ct_mtl/<实验名>/")
    p.add_argument("--name", default=None, help="影像组学特征表名称（默认 v2.radiomics.name）")
    p.add_argument("--folds", default=None, help="只跑这些折，默认全部")
    p.add_argument("--overwrite", action="store_true", help="已完成的折也重新拟合")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
