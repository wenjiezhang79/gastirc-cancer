"""v3 后期融合对照：组合 v1 病理、v2 CT 两个单模态实验的折外预测（不训练神经网络）。

两种方法（docs/decisions.md 第十四节第 6 小节）：
- ``mean``（平均）：二分类取两模态 logit 的平均；多分类取对数概率的平均再 softmax；生存取两模态风险 z 分数的平均
  （z 分数的均值、标准差只用其他折估计）。不拟合任何参数。
- ``stack``（交叉拟合堆叠）：第 k 折的组合系数只用**其他折**的折外预测拟合——二分类 / 多分类用逻辑回归，
  生存用按人群分层的 Cox；特征为两模态的 logit / 对数概率 / 风险 z 分数（标准化后），缺一种模态的病例该模态特征
  取训练均值（标准化后为 0）。
只有一种模态的病例，其预测只来自该模态。最终评估时用全部折外预测拟合一次后冻结（late_model.json）。

命令行：
    python -m gastric_mm.fusion.late --config configs/server.yaml --exp late_mean --method mean
    python -m gastric_mm.fusion.late --config configs/server.yaml --exp late_stack --method stack
    # 来源实验默认取 v3.init.path_exp / ct_exp，可用 --path-exp / --ct-exp 指定

输出（<output_dir>/fusion/<实验名>/，experiment.json 中 kind=late）：
    oof_predictions.csv、cv_metrics.json、cv_report.md、postprocess.json、late_model.json
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config, load_config, save_config
from ..ct_mtl.linear import fit_l1
from ..mtl.experiment import NAME_RE_STR, exp_root, postprocess_from_oof, safe_evaluate
from ..path_mtl.common import load_labels_and_split
from ..path_mtl.evaluate import merged, metrics_table_md
from ..training.task_spec import POPULATIONS, TaskSpec
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.log import setup_logging
from ..utils.runio import git_commit, package_versions
from .common import NAMESPACE, SECTION, SourceExp

EPS = 1e-6
METHODS = ("mean", "stack")


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _softmax(z):
    z = z - np.nanmax(z, 1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def common_spec(spec: TaskSpec, a: pd.DataFrame, b: pd.DataFrame) -> Tuple[TaskSpec, List[str]]:
    """只保留两个来源实验都有预测的任务。"""
    def has(df, t, kind, c=None):
        if kind == "mc":
            return all(f"{t}_p{k}" in df for k in range(c))
        return (f"{t}_prob" if kind == "bin" else f"{t}_risk") in df

    mc = {t: c for t, c in spec.multiclass.items() if has(a, t, "mc", c) and has(b, t, "mc", c)}
    bi = [t for t in spec.binary if has(a, t, "bin") and has(b, t, "bin")]
    sv = [t for t in spec.survival if has(a, t, "sv") and has(b, t, "sv")]
    dropped = [t for t in spec.all_tasks if t not in list(mc) + bi + sv]
    return TaskSpec(multiclass=mc, binary=bi, regression=[], survival=sv,
                    population_baselines=[p for p in spec.population_baselines if p in sv]), dropped


def task_features(t: str, spec: TaskSpec, df: pd.DataFrame, suffix: str) -> np.ndarray:
    """某任务、某模态的特征矩阵 [n, q]（二分类 logit / 多分类对数概率 / 生存风险）。缺失为 NaN。"""
    def col(c):  # 某模态在该队列中完全没有预测时，整列为缺失
        return df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)

    if t in spec.multiclass:
        P = np.stack([col(f"{t}_p{k}{suffix}") for k in range(spec.multiclass[t])], 1)
        return np.log(np.clip(P, EPS, 1.0))
    if t in spec.binary:
        return _logit(col(f"{t}_prob{suffix}"))[:, None]
    return col(f"{t}_risk{suffix}")[:, None]


def _std_stats(X: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Z = X[rows]
    mu = np.nanmean(Z, 0) if np.isfinite(Z).any() else np.zeros(X.shape[1])
    sd = np.nanstd(Z, 0) if np.isfinite(Z).any() else np.ones(X.shape[1])
    return np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)


def _strata(lab: pd.DataFrame, spec: TaskSpec, ep: str) -> np.ndarray:
    if spec.n_pops(ep) > 1:
        return lab["population"].map(lambda v: POPULATIONS.index(v) if v in POPULATIONS else -1).to_numpy(int)
    return np.zeros(len(lab), dtype=int)


def fit_task(method: str, t: str, spec: TaskSpec, Xp: np.ndarray, Xc: np.ndarray, lab: pd.DataFrame,
             rows: np.ndarray, penalty: float) -> dict:
    """在 rows（训练部分）上拟合一个任务的组合模型；返回可序列化的参数。"""
    mu_p, sd_p = _std_stats(Xp, rows)
    mu_c, sd_c = _std_stats(Xc, rows)
    m = {"mu_p": mu_p.tolist(), "sd_p": sd_p.tolist(), "mu_c": mu_c.tolist(), "sd_c": sd_c.tolist()}
    if method == "mean":
        return m
    Zp = (Xp - mu_p) / sd_p
    Zc = (Xc - mu_c) / sd_c
    both = rows & np.isfinite(Zp).all(1) & np.isfinite(Zc).all(1)
    X = np.concatenate([Zp, Zc], 1)
    if t in spec.multiclass:
        y = lab[t].to_numpy(float)
        ok = both & np.isfinite(y)
        W, b = fit_l1("multiclass", X[ok], y[ok].astype(int), penalty, {}, n_out=spec.multiclass[t])
    elif t in spec.binary:
        y = lab[t].to_numpy(float)
        ok = both & np.isfinite(y)
        W, b = fit_l1("binary", X[ok], y[ok], penalty, {"w": np.ones(int(ok.sum()))}, n_out=1)
    else:
        tm, ev = lab[f"{t}_time"].to_numpy(float), lab[f"{t}_event"].to_numpy(float)
        st = _strata(lab, spec, t)
        ok = both & np.isfinite(tm) & np.isfinite(ev) & (st >= 0)
        W, b = fit_l1("survival", X[ok], None, penalty, {"time": tm[ok], "event": ev[ok], "strata": st[ok]}, n_out=1)
    if int(ok.sum()) < 10:
        raise UserError(f"后期融合堆叠：任务 {t} 在训练部分只有 {int(ok.sum())} 例两种模态都有且有标签，无法拟合",
                        "该任务的样本太少；可只用 --method mean。")
    m.update(W=np.asarray(W).tolist(), b=np.asarray(b).tolist(), n_fit=int(ok.sum()))
    return m


def apply_task(method: str, t: str, spec: TaskSpec, Xp: np.ndarray, Xc: np.ndarray, m: dict) -> np.ndarray:
    """返回：多分类 [n, C] 概率；二分类 [n] 概率；生存 [n] 风险。只有一种模态时只用该模态。"""
    Zp = (Xp - np.array(m["mu_p"])) / np.array(m["sd_p"])
    Zc = (Xc - np.array(m["mu_c"])) / np.array(m["sd_c"])
    hp, hc = np.isfinite(Zp).all(1), np.isfinite(Zc).all(1)
    if method == "mean":
        if t in spec.survival:              # 生存：z 分数平均
            A, B = Zp[:, 0], Zc[:, 0]
        else:                               # 原尺度（logit / 对数概率）平均
            A, B = Xp, Xc
            if t in spec.binary:
                A, B = A[:, 0], B[:, 0]
        out = np.where(hp & hc, (np.nan_to_num(A) + np.nan_to_num(B)) / 2, np.where(hp, A, B)) \
            if A.ndim == 1 else np.where((hp & hc)[:, None], (np.nan_to_num(A) + np.nan_to_num(B)) / 2,
                                         np.where(hp[:, None], A, B))
    else:
        X = np.concatenate([np.where(hp[:, None], Zp, 0.0), np.where(hc[:, None], Zc, 0.0)], 1)
        W, b = np.array(m["W"]), np.array(m["b"])
        out = X @ W + b
        if t not in spec.multiclass:
            out = out[:, 0]
        if out.ndim == 1:
            out = np.where(hp | hc, out, np.nan)
        else:
            out = np.where((hp | hc)[:, None], out, np.nan)
    if t in spec.multiclass:
        return _softmax(out)
    if t in spec.binary:
        return _sigmoid(out)
    return out


def combine(method: str, spec: TaskSpec, both_df: pd.DataFrame, lab: pd.DataFrame, fit_rows: np.ndarray,
            apply_rows: np.ndarray, penalty: float) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    preds, models = {}, {}
    for t in spec.all_tasks:
        Xp, Xc = task_features(t, spec, both_df, "_p"), task_features(t, spec, both_df, "_c")
        m = fit_task(method, t, spec, Xp, Xc, lab, fit_rows, penalty)
        models[t] = m
        preds[t] = apply_task(method, t, spec, Xp[apply_rows], Xc[apply_rows], m)
    return preds, models


def pred_frame(spec: TaskSpec, sids, preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame({"study_id": list(sids)})
    for t in spec.all_tasks:
        v = preds[t]
        if t in spec.multiclass:
            for k in range(v.shape[1]):
                out[f"{t}_p{k}"] = v[:, k]
        elif t in spec.binary:
            out[f"{t}_prob"] = v
        else:
            out[f"{t}_risk"] = v
    return out


def side_by_side(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """病理（后缀 _p）与 CT（后缀 _c）预测并排；has_path / has_ct 标记。"""
    keep = lambda d: d[[c for c in d.columns if c not in ("fold", "seed") and not c.startswith("has_")  # noqa: E731
                        and not c.startswith("attn_")]]
    pa, pb = keep(a).add_suffix("_p"), keep(b).add_suffix("_c")
    pa = pa.rename(columns={"study_id_p": "study_id"})
    pb = pb.rename(columns={"study_id_c": "study_id"})
    df = pa.merge(pb, on="study_id", how="outer")
    df["has_path"] = df["study_id"].isin(a.study_id).astype(int)
    df["has_ct"] = df["study_id"].isin(b.study_id).astype(int)
    return df


def late_fingerprint(cfg: Config, method: str, srcs: Dict[str, SourceExp]) -> str:
    obj = {"method": method, "penalty": float(cfg[SECTION].get("late", {}).get("penalty", 1e-3)),
           "sources": {k: [s.name, s.fingerprint] for k, s in srcs.items()},
           "tasks": cfg[SECTION].tasks.to_dict(), "labels": cfg.labels.to_dict(), "split": cfg.split.to_dict()}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _run(args) -> int:
    import re

    cfg = load_config(args.config, args.set)
    sec = cfg[SECTION]
    init = sec.get("init", {}) or {}
    pname, cname = args.path_exp or init.get("path_exp"), args.ct_exp or init.get("ct_exp")
    if not pname or not cname:
        raise UserError("后期融合需要 v1 病理实验与 v2 CT 实验", "请用 --path-exp / --ct-exp 指定，或在 v3.init 中填写。")
    if not re.match(NAME_RE_STR, args.exp or ""):
        raise UserError(f"实验名“{args.exp}”不合法", "实验名只能包含英文字母、数字、下划线、点和短横线。")
    srcs = {"path": SourceExp(cfg, "path", pname), "ct": SourceExp(cfg, "ct", cname)}
    exp = exp_root(cfg, NAMESPACE) / args.exp
    fp = late_fingerprint(cfg, args.method, srcs)
    if (exp / "experiment.json").exists():
        old = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
        if old.get("fingerprint") != fp:
            raise UserError(f"实验 {args.exp} 已存在，但这次的设置（方法 / 来源实验 / 任务）与第一次不同",
                            "请换一个新的实验名（--exp）。")
    exp.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = setup_logging(exp / "logs" / f"{ts}_late.log")
    for s in srcs.values():
        if not (s.dir / "oof_predictions.csv").exists():
            raise UserError(f"{s.label}实验 {s.name} 的交叉验证尚未汇总（缺少 oof_predictions.csv）", "请先跑完该实验的全部折。")
    a = pd.read_csv(srcs["path"].dir / "oof_predictions.csv", dtype={"study_id": str})
    b = pd.read_csv(srcs["ct"].dir / "oof_predictions.csv", dtype={"study_id": str})
    from ..mtl.spec import spec_from_section
    spec, dropped = common_spec(spec_from_section(cfg, SECTION), a, b)
    if dropped:
        log.info("以下任务不是两个来源实验都有，后期融合不包含：%s", ", ".join(dropped))
    df_lab = load_labels_and_split(cfg, log)
    train = df_lab[df_lab.set == "train"]
    both = side_by_side(a, b)
    lab = both[["study_id", "has_path", "has_ct"]].merge(train, on="study_id", how="left")
    if lab["set"].isna().any() or (lab["set"] != "train").any():
        raise UserError("来源实验的折外预测中有不属于当前内部训练集的病例", "请确认 v1 / v2 / v3 使用同一个划分文件与清单。")
    folds = lab["fold"].to_numpy(int)
    penalty = float(sec.get("late", {}).get("penalty", 1e-3))
    parts = []
    for k in sorted(np.unique(folds)):
        p, _ = combine(args.method, spec, both, lab, folds != k, folds == k, penalty)
        f = pred_frame(spec, both.study_id[folds == k], p)
        f["fold"] = k
        parts.append(f)
    oof = pd.concat(parts, ignore_index=True)
    oof = oof.merge(both[["study_id", "has_path", "has_ct"]], on="study_id", how="left")
    oof.to_csv(exp / "oof_predictions.csv", index=False)
    _, final_models = combine(args.method, spec, both, lab, np.ones(len(lab), bool), np.zeros(len(lab), bool), penalty)
    (exp / "late_model.json").write_text(json.dumps({"method": args.method, "spec": spec.to_dict(),
                                                     "models": final_models, "fitted_on": "内部训练集全部折外预测",
                                                     "n": int(len(lab))}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    meta = {"name": args.exp, "kind": "late", "method": args.method, "fingerprint": fp,
            "sources": {k: s.name for k, s in srcs.items()}, "source_fingerprints": {k: s.fingerprint for k, s in srcs.items()},
            "created": _dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "packages": package_versions(), "tasks": spec.all_tasks}
    if not (exp / "experiment.json").exists():
        save_config(cfg, exp / "config.yaml")
    (exp / "experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    n_boot = int(sec.get("eval", {}).get("n_boot", 1000))
    dfm = merged(oof, train.drop(columns=["fold"]))
    rows = safe_evaluate(dfm, spec, cfg, n_boot=n_boot)
    post = postprocess_from_oof(dfm, spec)
    (exp / "postprocess.json").write_text(json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp / "cv_metrics.json").write_text(json.dumps({"oof": rows, "n_cases": int(len(dfm))}, ensure_ascii=False,
                                                    indent=2, default=float), encoding="utf-8")
    nb, npo, nc = int(((oof.has_path == 1) & (oof.has_ct == 1)).sum()), int(((oof.has_path == 1) & (oof.has_ct == 0)).sum()), \
        int(((oof.has_path == 0) & (oof.has_ct == 1)).sum())
    L = [f"# 后期融合：{args.exp}（{'平均' if args.method == 'mean' else '交叉拟合堆叠'}）", "",
         f"- 来源：病理 {pname}、CT {cname}（两者的折外预测，种子平均）",
         f"- 病例：{len(oof)}（两种都有 {nb}，只有病理 {npo}，只有 CT {nc}；只有一种模态的病例只用该模态预测）",
         "- 交叉拟合：第 k 折的组合参数（含标准化）只用其他折的折外预测估计；最终评估用全部折外预测拟合一次后冻结（late_model.json）。",
         "- 生存终点只输出风险评分（不输出各时间点生存概率）。", "",
         "## 折外指标", "", metrics_table_md(rows)]
    (exp / "cv_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info("后期融合完成：%s", exp / "cv_report.md")
    for r in rows:
        if r["main"]:
            log.info("  %s %s = %.3f（95%% CI %.3f–%.3f，n=%d）", r["task"], r["metric"], r["value"], r["lo"], r["hi"],
                     r["n"])
    return 0


def main(argv=None) -> int:
    p = base_parser("v3 后期融合对照：组合 v1 病理与 v2 CT 的折外预测（平均 / 交叉拟合堆叠）。", cases=False, workers=False)
    p.add_argument("--exp", required=True, help="实验名（如 late_mean）；结果在 <output_dir>/fusion/<实验名>/")
    p.add_argument("--method", required=True, choices=METHODS, help="mean=平均（不拟合）；stack=交叉拟合堆叠")
    p.add_argument("--path-exp", default=None, help="v1 病理实验名（默认 v3.init.path_exp）")
    p.add_argument("--ct-exp", default=None, help="v2 CT 实验名（默认 v3.init.ct_exp）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
