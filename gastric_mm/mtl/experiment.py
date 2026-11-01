"""实验目录、配置指纹、内层验证划分、交叉验证汇总、最终评估登记（v2 / v3 共用）。

实验目录 ``<output_dir>/<命名空间>/<实验名>/``（v2 CT 单模态的命名空间为 ct_mtl）：结构与 v1 一致——
同一实验可以分多次、在两张 GPU 上分别运行不同的折；第一次运行时记录配置指纹，之后影响结果的配置被改动时拒绝继续。

汇总输出（全部折完成后）：oof_predictions.csv、cv_metrics.json、cv_report.md、postprocess.json（阈值与风险截断，
由内部训练集折外预测确定后冻结）、oof_embeddings.h5（token，供 v3 使用）。
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import Config, _to_config, deep_merge, resolve_path, save_config
from ..path_mtl.evaluate import evaluate, merged, metrics_table_md, primary_value
from ..split import _stratum_keys, greedy_assign
from ..training.task_spec import TaskSpec
from ..utils.errors import UserError
from ..utils.log import setup_logging
from ..utils.runio import git_commit, package_versions

NAME_RE_STR = r"^[A-Za-z0-9_.\-]+$"
# 不影响结果、允许在同一实验中修改的配置项（v2 节中的提取/分割/比较/评估等）
RUNTIME_KEYS = ("seeds", "feature_cache", "save_attention", "compare", "eval", "profiles", "external", "stomach",
                "seg_sample", "tumor_seg", "features")


def apply_profile(cfg: Config, section: str, profile: Optional[str]) -> Config:
    if not profile:
        return cfg
    profs = cfg[section].get("profiles", {}) or {}
    if profile not in profs:
        raise UserError(f"配置中没有预设 {section}.profiles.{profile}", f"可用：{', '.join(profs) or '无'}")
    cfg = copy.deepcopy(cfg)
    cfg[section] = _to_config(deep_merge(cfg[section].to_dict(), profs[profile]))
    cfg[section]["_profile"] = profile
    return cfg


def fingerprint(cfg: Config, section: str, kind: str) -> str:
    sec = {k: v for k, v in cfg[section].to_dict().items() if k not in RUNTIME_KEYS}
    sec.get("train", {}).pop("num_workers", None)
    if kind != "radiomics":
        sec.pop("radiomics", None)
    obj = {section: sec, "kind": kind, "labels": cfg.labels.to_dict(), "split": cfg.split.to_dict(),
           "internal_center": cfg.manifest.internal_center}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exp_root(cfg: Config, namespace: str) -> Path:
    return resolve_path(cfg, cfg.paths.output_dir) / namespace


def open_experiment(cfg: Config, section: str, namespace: str, name: str, log_tag: str, kind: str,
                    meta_extra: dict) -> Tuple[Path, object]:
    import re

    if not name or not re.match(NAME_RE_STR, name):
        raise UserError(f"实验名“{name}”不合法", "实验名只能包含英文字母、数字、下划线、点和短横线，如 ct_radimagenet。")
    d = exp_root(cfg, namespace) / name
    meta_p = d / "experiment.json"
    fp = fingerprint(cfg, section, kind)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fp:
            raise UserError(f"实验 {name} 已存在，但这次的配置与第一次运行时不同（影响结果的设置被改动）",
                            "请换一个新的实验名（--exp），不同设置的结果不能混在同一个实验里。")
    else:
        d.mkdir(parents=True, exist_ok=True)
        save_config(cfg, d / "config.yaml")
        meta = {"name": name, "kind": kind, "section": section, "fingerprint": fp,
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "profile": cfg[section].get("_profile"), "git_commit": git_commit(), "packages": package_versions()}
        meta.update(meta_extra)
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = setup_logging(d / "logs" / f"{ts}_{log_tag}.log")
    logger.info("实验目录：%s", d)
    return d, logger


def read_meta(exp: Path) -> dict:
    return json.loads((exp / "experiment.json").read_text(encoding="utf-8"))


def inner_split(lab: pd.DataFrame, cfg: Config, frac: float, seed: int) -> np.ndarray:
    """训练部分内分出内层验证集（按 split.stratify_by 分层）。True = 内层验证。"""
    keys = _stratum_keys(lab.reset_index(drop=True), list(cfg.split.stratify_by))
    return greedy_assign(keys, [1 - frac, frac], np.random.RandomState(seed)) == 1


def safe_evaluate(df: pd.DataFrame, spec: TaskSpec, cfg: Config, **kw) -> List[dict]:
    """同 path_mtl.evaluate.evaluate，但先去掉该组中没有任何标签的任务（如某外部中心没有不可切除患者时的 PFS），
    避免 v1 指标函数在空数组上报错；被去掉的任务在结果中记为"无标签"。"""
    def n_lab(t):
        if t in spec.survival:
            return int((df[f"{t}_time"].notna() & df[f"{t}_event"].notna()).sum())
        return int(df[t].notna().sum()) if t in df else 0

    empty = [t for t in spec.all_tasks if n_lab(t) == 0]
    if not empty:
        return evaluate(df, spec, cfg, **kw)
    sub = TaskSpec(multiclass={t: c for t, c in spec.multiclass.items() if t not in empty},
                   binary=[t for t in spec.binary if t not in empty],
                   regression=[t for t in spec.regression if t not in empty],
                   survival=[t for t in spec.survival if t not in empty],
                   population_baselines=list(spec.population_baselines))
    rows = evaluate(df, sub, cfg, **kw) if sub.all_tasks else []
    for t in empty:
        rows.append({"task": t, "metric": "—", "value": float("nan"), "lo": float("nan"), "hi": float("nan"),
                     "n": 0, "n_pos": None, "main": False, "note": "该组无标签"})
    return rows


# ---------------- 交叉验证汇总 ----------------
def postprocess_from_oof(df: pd.DataFrame, spec: TaskSpec) -> dict:
    thresholds = {t: M.youden_threshold(df[t], df[f"{t}_prob"]) for t in spec.binary}
    cutoffs = {}
    for ep in spec.survival:
        cutoffs[ep] = {"all": float(np.nanmedian(df[f"{ep}_risk"]))}
        for pop in ("resected", "unresectable"):
            m = df["population"] == pop
            if m.sum():
                cutoffs[ep][pop] = float(np.nanmedian(df.loc[m, f"{ep}_risk"]))
    return {"source": "内部训练集折外预测（种子平均）", "binary_thresholds_youden": thresholds,
            "survival_risk_cutoffs_median": cutoffs, "n_cases": int(len(df))}


def aggregate(exp: Path, cfg: Config, section: str, spec: TaskSpec, pool: pd.DataFrame, seeds: List[int],
              n_folds: int, logger, describe: Callable[[dict], List[str]]) -> bool:
    """所有（种子, 折）都完成后汇总；否则返回 False。describe(meta) 返回报告开头的说明行。"""
    missing = [(s, k) for s in seeds for k in range(n_folds)
               if not (exp / f"seed{s}" / f"fold{k}" / "done.json").exists()]
    if missing:
        logger.info("尚有 %d 个（种子, 折）未完成：%s；全部完成后会自动汇总", len(missing),
                    ", ".join(f"seed{s}/fold{k}" for s, k in missing[:10]))
        return False
    sec = cfg[section]
    n_boot = int(sec.get("eval", {}).get("n_boot", 1000))
    oof = pd.concat([pd.read_csv(exp / f"seed{s}" / f"fold{k}" / "oof.csv", dtype={"study_id": str})
                     for s in seeds for k in range(n_folds)], ignore_index=True)
    num = [c for c in oof.columns if c not in ("study_id", "fold", "seed")]
    avg = oof.groupby("study_id", sort=False).agg({**{c: "mean" for c in num}, "fold": "first"}).reset_index()
    avg.to_csv(exp / "oof_predictions.csv", index=False)
    lab = pool.drop(columns=["fold"])
    df = merged(avg, lab)
    rows = safe_evaluate(df, spec, cfg, n_boot=n_boot)
    per: Dict[str, list] = {}
    for _, g in oof.groupby(["seed", "fold"]):
        d = merged(g, lab)
        for t in spec.all_tasks:
            per.setdefault(t, []).append(primary_value(d, spec, t))
    fold_stats = {t: (float(np.nanmean(v)), float(np.nanstd(v))) for t, v in per.items()}
    seed_stats = {}
    for s, g in oof.groupby("seed"):
        d = merged(g, lab)
        seed_stats[int(s)] = {t: primary_value(d, spec, t) for t in spec.all_tasks}
    post = postprocess_from_oof(df, spec)
    (exp / "postprocess.json").write_text(json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"oof": rows, "per_fold_mean_sd": fold_stats, "per_seed": seed_stats, "seeds": seeds,
               "n_folds": n_folds, "n_cases": int(len(df))}
    (exp / "cv_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
    _merge_tokens(exp, seeds[0], n_folds)
    meta = read_meta(exp)
    L = [f"# 交叉验证报告：{exp.name}", ""] + describe(meta) + [
        f"- 病例：内部训练集 {len(df)} 例（有特征者）；随机种子：{list(seed_stats)}",
        "- 折外预测：每例由未见过它的模型预测；多个种子时取平均。置信区间为 bootstrap 95% CI。",
        "- 注意：没有信号的任务，合并各折得到的折外 AUC 可能偏离 0.5（各折阳性率不同所致），请同时参考“各折均值”。", "",
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
    (exp / "cv_report.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("交叉验证汇总完成：%s", exp / "cv_report.md")
    for r in rows:
        if r["main"]:
            logger.info("  %s %s = %.3f（95%% CI %.3f–%.3f，n=%d）", r["task"], r["metric"], r["value"], r["lo"],
                        r["hi"], r["n"])
    return True


def _merge_tokens(exp: Path, seed: int, n_folds: int) -> None:
    import h5py

    files = [exp / f"seed{seed}" / f"fold{k}" / "tokens.npz" for k in range(n_folds)]
    if not all(p.exists() for p in files):
        return
    with h5py.File(exp / "oof_embeddings.tmp", "w") as out:
        for p in files:
            z = np.load(p)
            for sid, t in zip(z["study_id"], z["tokens"]):
                out.create_dataset(str(sid), data=t.astype(np.float32))
    (exp / "oof_embeddings.tmp").replace(exp / "oof_embeddings.h5")


def save_attention(path: Path, attn: Dict[str, Dict[str, np.ndarray]]) -> None:
    import h5py

    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for sid, d in attn.items():
            g = f.create_group(sid)
            for role, a in d.items():
                g.create_dataset(role, data=np.asarray(a, np.float16))
    tmp.replace(path)


# ---------------- 最终评估登记（只运行一次） ----------------
def check_final_registry(exp: Path, cohort: str, fp: str, allow_rerun: bool) -> Tuple[list, list]:
    reg_p = exp / "final_evaluations.json"
    reg = json.loads(reg_p.read_text(encoding="utf-8")) if reg_p.exists() else []
    prev = [r for r in reg if r["cohort"] == cohort and r["models_fingerprint"] == fp]
    if prev and not allow_rerun:
        raise UserError(f"该模型已于 {prev[0]['time']} 对队列 {cohort} 做过最终评估（结果在 {prev[0]['dir']}）",
                        "最终评估只做一次。如确有必要重跑（例如程序出错），加 --allow-rerun，报告中会注明这是重跑。")
    return reg, prev


def register_final(exp: Path, reg: list, entry: dict) -> None:
    reg.append(entry)
    (exp / "final_evaluations.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def files_fingerprint(files: Sequence[Path], base: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.relative_to(base).as_posix().encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]
