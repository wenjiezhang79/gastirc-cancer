"""数据划分：只在内部中心内，按病人分层 8:2 分为训练集 / 内部测试集，训练集内做 K 折交叉验证。

外部中心（ext1/ext2/ext3）全部标为 external，不参与任何训练、调参、阈值选择。

命令行：
    python -m gastric_mm.split --config configs/server.yaml

输出：
- <work_dir>/splits/split_seed<种子>.csv   列：study_id, center, set(train/test/external), fold(0..K-1，非训练集为 -1)
- 运行目录下 split_report.md（各集合/各折的标签分布）

划分文件一旦生成，默认不覆盖（保证所有实验使用同一划分）；确需重新划分时加 --overwrite。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import Config, load_config
from .labels import derive_labels
from .manifest import load_manifest
from .paths import WorkPaths
from .utils.cli import base_parser, run_main
from .utils.errors import UserError
from .utils.runio import Run


def _stratum_keys(labels: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    keys = pd.DataFrame(index=labels.index)
    for c in by:
        if c not in labels.columns:
            raise UserError(f"split.stratify_by 中的 {c} 不是有效的标签列", "可用：lauren, msi_h, ebv_pos, her2_pos 等。")
        keys[c] = labels[c].map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    return keys


def greedy_assign(keys: pd.DataFrame, weights: List[float], rng: np.random.RandomState) -> np.ndarray:
    """迭代分层（贪心）：按"最稀有取值"由少到多依次处理病例，每例分到当前最欠缺它各项取值的组。

    weights 为各组目标比例（如 [0.8, 0.2] 或 5 个 0.2）。每个分层变量分别保持均衡，
    稀有阳性（MSI、EBV）也能在各组之间均匀分布。返回每例的组号。
    """
    n = len(keys)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cols = list(keys.columns)
    totals = {c: keys[c].value_counts().to_dict() for c in cols}
    vals = keys.values
    rarity = np.array([min(totals[c][vals[i, j]] for j, c in enumerate(cols)) if cols else n for i in range(n)],
                      dtype=float)
    order = np.lexsort((rng.rand(n), rarity))
    counts = [{c: {} for c in cols} for _ in w]
    sizes = np.zeros(len(w))
    out = np.zeros(n, dtype=int)
    for i in order:
        row = dict(zip(cols, vals[i]))
        cost = np.zeros(len(w))
        for b in range(len(w)):
            for c in cols:
                v = row[c]
                cost[b] += (counts[b][c].get(v, 0) + 1) / (w[b] * totals[c][v])
            cost[b] += (sizes[b] + 1) / (w[b] * n)
        best = np.flatnonzero(np.isclose(cost, cost.min()))
        b = int(rng.choice(best))
        out[i] = b
        sizes[b] += 1
        for c in cols:
            counts[b][c][row[c]] = counts[b][c].get(row[c], 0) + 1
    return out


def make_split(labels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sc = cfg.split
    internal = cfg.manifest.internal_center
    rng = np.random.RandomState(int(sc.get("seed", cfg.get("seed", 0))))
    n_folds = int(sc.n_folds)
    test_frac = float(sc.test_fraction)

    out = pd.DataFrame({"study_id": labels["study_id"], "center": labels["center"]})
    out["set"] = np.where(out["center"] == internal, "", "external")
    out["fold"] = -1

    idx_int = labels.index[labels["center"] == internal]
    if len(idx_int) == 0:
        raise UserError(f"清单中没有内部中心（{internal}）的病例，无法划分。")
    keys = _stratum_keys(labels.loc[idx_int], list(sc.stratify_by))
    grp = greedy_assign(keys, [1 - test_frac, test_frac], rng)
    out.loc[idx_int, "set"] = np.where(grp == 1, "test", "train")
    train_idx = idx_int[grp == 0]
    folds = greedy_assign(keys.loc[train_idx], [1.0] * n_folds, rng)
    out.loc[train_idx, "fold"] = folds
    out["stratum"] = ""
    out.loc[idx_int, "stratum"] = keys.astype(str).agg("|".join, axis=1)
    return out


def split_report(split: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> str:
    df = split.merge(labels.drop(columns=["center"]), on="study_id", how="left")
    groups: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    groups["训练集"] = df[df["set"] == "train"]
    for k in range(int(cfg.split.n_folds)):
        groups[f"  第{k}折"] = df[(df["set"] == "train") & (df["fold"] == k)]
    groups["内部测试集"] = df[df["set"] == "test"]
    for c in cfg.manifest.centers:
        if c != cfg.manifest.internal_center:
            g = df[df["center"] == c]
            if len(g):
                groups[f"外部 {c}"] = g
    cols = ["lauren", "her2_pos", "msi_h", "ebv_pos", f"pdl1_cps{int(cfg.labels.pdl1_main_threshold)}", "cldn182_pos"]
    L = ["# 数据划分报告", "", f"- 随机种子：{cfg.split.get('seed')}　折数：{cfg.split.n_folds}　测试集比例：{cfg.split.test_fraction}",
         f"- 分层变量：{', '.join(cfg.split.stratify_by)}", "",
         "| 集合 | 例数 | Lauren 肠/弥/混 | " + " | ".join(c + " 阳性/有标签" for c in cols[1:]) + " | OS 事件 |",
         "|" + "---|" * (len(cols) + 3)]
    for name, g in groups.items():
        la = "/".join(str(int((g["lauren"] == k).sum())) for k in range(3))
        cells = [f"{int((g[c] == 1).sum())}/{int(g[c].notna().sum())}" for c in cols[1:]]
        L.append(f"| {name} | {len(g)} | {la} | " + " | ".join(cells) + f" | {int((g['os_event'] == 1).sum())} |")
    L.append("")
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "split", args.run_name)
    log = run.logger
    res = load_manifest(cfg, log, check_paths=False)
    labels = derive_labels(res.df, cfg)
    wp = WorkPaths(cfg)
    seed = cfg.split.get("seed", cfg.get("seed"))
    out_path = wp.splits / f"split_seed{seed}.csv"
    if out_path.exists() and not args.overwrite:
        raise UserError(
            f"划分文件已存在：{out_path.name}（所有实验应使用同一划分）",
            "如确需重新划分（例如清单新增了病例），加 --overwrite；旧文件会被备份为 .bak。",
        )
    split = make_split(labels, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.replace(out_path.with_suffix(".csv.bak"))
    split.to_csv(out_path, index=False, encoding="utf-8-sig")
    split.to_csv(run.path(out_path.name), index=False, encoding="utf-8-sig")
    run.write_text(split_report(split, labels, cfg), "split_report.md")
    counts = split["set"].value_counts().to_dict()
    log.info("划分完成：训练 %d、内部测试 %d、外部 %d。文件：%s",
             counts.get("train", 0), counts.get("test", 0), counts.get("external", 0), out_path)
    return 0


def load_split(cfg: Config) -> pd.DataFrame:
    wp = WorkPaths(cfg)
    seed = cfg.split.get("seed", cfg.get("seed"))
    p = wp.splits / f"split_seed{seed}.csv"
    if not p.exists():
        raise UserError(f"找不到划分文件 {p.name}", "请先运行 bash scripts/01b_split.sh 生成数据划分。")
    return pd.read_csv(p, dtype={"study_id": str})


def main(argv=None) -> int:
    p = base_parser("在内部中心内按病人分层划分训练集/内部测试集，并在训练集内做 K 折交叉验证。",
                    cases=False, workers=False)
    p.add_argument("--overwrite", action="store_true", help="覆盖已有划分文件（旧文件备份为 .bak）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
