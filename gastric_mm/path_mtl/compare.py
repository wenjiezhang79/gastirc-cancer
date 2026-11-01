"""比较多个实验（如 7 个编码器的初筛结果），按预先规定的指标排名。

排名指标（v1.compare.primary，预先规定，默认 [lauren, msi_h]）：各任务主要指标的平均
（多分类=宏平均 AUC，二分类=AUC，生存=C-index），基于各实验的折外预测（种子平均）。
只在所有实验共有的病例上比较；给出每个实验的 95% CI，以及与第一名的**配对 bootstrap 差值**
（同一批 bootstrap 抽样），用来判断差距是否超出随机波动。其他任务的折外指标附在后面仅供参考，不参与排名。

命令行：
    python -m gastric_mm.path_mtl.compare --config configs/server.yaml --exps screen_uni2_h,screen_virchow2,...
    python -m gastric_mm.path_mtl.compare --config configs/server.yaml --pattern "screen_*"

输出：<output_dir>/path_mtl/_compare/<时间>/compare_report.md、compare.csv
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import load_config
from ..models.path_model import load_model
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.log import setup_logging
from ..utils.runio import write_csv
from .common import exp_root, load_labels_and_split
from .evaluate import merged, primary_value


def _score_fn(spec, tasks):
    def fn(df):
        vals = [primary_value(df, spec, t) for t in tasks]
        return float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
    return fn


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    root = exp_root(cfg)
    if args.exps:
        names = [x.strip() for x in args.exps.split(",") if x.strip()]
    elif args.pattern:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.pattern))
    else:
        raise UserError("请用 --exps 或 --pattern 指定要比较的实验")
    ready = [n for n in names if (root / n / "oof_predictions.csv").exists()]
    not_ready = sorted(set(names) - set(ready))
    if len(ready) < 2:
        raise UserError(f"可比较的实验不足 2 个（已完成：{ready}；未完成或不存在：{not_ready}）",
                        "请先完成各实验的全部折训练（出现 cv_report.md）。")
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = root / "_compare" / ts
    i = 1
    while out.exists():
        i += 1
        out = root / "_compare" / f"{ts}-{i}"
    out.mkdir(parents=True)
    log = setup_logging(out / "run.log")
    primary = [x.strip() for x in (args.primary.split(",") if args.primary else cfg.v1.compare.primary)]
    n_boot = int(args.n_boot or cfg.v1.get("eval", {}).get("n_boot", 1000))
    labels = load_labels_and_split(cfg, log).drop(columns=["fold"])

    data, specs, metas = {}, {}, {}
    for n in ready:
        d = root / n
        model, _ = load_model(next(d.glob("seed*/fold*/model.pt")))
        specs[n] = model.spec
        missing = [t for t in primary if t not in model.spec.all_tasks]
        if missing:
            raise UserError(f"实验 {n} 没有训练任务 {missing}，无法按 {primary} 比较")
        metas[n] = json.loads((d / "experiment.json").read_text(encoding="utf-8"))
        data[n] = merged(pd.read_csv(d / "oof_predictions.csv", dtype={"study_id": str}), labels)
    common = set.intersection(*[set(df.study_id) for df in data.values()])
    log.info("比较 %d 个实验，共有病例 %d 例；排名指标：%s 的平均", len(ready), len(common), "+".join(primary))
    for n in ready:
        data[n] = data[n][data[n].study_id.isin(common)].sort_values("study_id").reset_index(drop=True)

    # 同一批 bootstrap 下标，按第一个主要任务的标签分层
    ref = data[ready[0]]
    strata_col = primary[0] if primary[0] in ref else None
    strata = ref[strata_col].fillna(-1).to_numpy() if strata_col else None
    idx_list = list(M.bootstrap_indices(len(ref), n_boot, seed=0, strata=strata))
    boots: Dict[str, np.ndarray] = {}
    points: Dict[str, float] = {}
    for n in ready:
        fn = _score_fn(specs[n], primary)
        points[n] = fn(data[n])
        boots[n] = np.array([fn(data[n].iloc[i]) for i in idx_list])
    order = sorted(ready, key=lambda n: -points[n] if np.isfinite(points[n]) else np.inf)
    best = order[0]
    rows: List[dict] = []
    for rank, n in enumerate(order, 1):
        b = boots[n]
        ok = np.isfinite(b)
        lo, hi = (np.percentile(b[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan))
        diff = boots[n] - boots[best]
        okd = np.isfinite(diff)
        dlo, dhi = (np.percentile(diff[okd], [2.5, 97.5]) if okd.sum() > 10 and n != best else (np.nan, np.nan))
        r = {"rank": rank, "experiment": n, "encoders": "+".join(metas[n]["encoders"]),
             "aggregator": metas[n]["aggregator"], "score": points[n], "lo": lo, "hi": hi,
             "diff_vs_best": points[n] - points[best], "diff_lo": dlo, "diff_hi": dhi}
        for t in primary:
            r[t] = primary_value(data[n], specs[n], t)
        rows.append(r)
    write_csv(rows, out / "compare.csv")

    L = ["# 实验比较", "", f"- 排名指标（预先规定）：{' + '.join(primary)} 主要指标的平均，基于折外预测",
         f"- 共有病例：{len(common)}；bootstrap {n_boot} 次（按 {strata_col} 分层，各实验使用同一批抽样）",
         f"- 未完成、未纳入比较：{', '.join(not_ready) or '无'}", "",
         "| 排名 | 实验 | 编码器 | 聚合器 | 排名指标 (95% CI) | " + " | ".join(primary)
         + " | 与第一名之差 (95% CI) |",
         "|---|---|---|---|---|" + "---|" * len(primary) + "---|"]
    for r in rows:
        d = "—" if r["experiment"] == best else f"{r['diff_vs_best']:+.3f}（{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}）"
        L.append(f"| {r['rank']} | {r['experiment']} | {r['encoders']} | {r['aggregator']} | "
                 f"{r['score']:.3f}（{r['lo']:.3f}–{r['hi']:.3f}） | "
                 + " | ".join(f"{r[t]:.3f}" for t in primary) + f" | {d} |")
    L += ["", "差值置信区间不包含 0，说明与第一名的差距超出了随机波动；包含 0 则不能认为两者有差别。", "",
          "## 其他任务的折外指标（仅供参考，不参与排名）", ""]
    other = sorted({t for n in ready for t in specs[n].all_tasks} - set(primary))
    if other:
        L += ["| 实验 | " + " | ".join(other) + " |", "|---|" + "---|" * len(other)]
        for n in order:
            L.append(f"| {n} | " + " | ".join(
                f"{primary_value(data[n], specs[n], t):.3f}" if t in specs[n].all_tasks else "—" for t in other) + " |")
    else:
        L.append("无（各实验只训练了排名任务）。")
    L.append("")
    (out / "compare_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info("第一名：%s（%.3f）。报告：%s", best, points[best], out / "compare_report.md")
    return 0


def main(argv=None) -> int:
    p = base_parser("比较多个 v1 实验（如编码器初筛），按预先规定的指标排名。", cases=False, workers=False)
    p.add_argument("--exps", default=None, help="实验名列表，逗号分隔")
    p.add_argument("--pattern", default=None, help="实验名通配符，如 'screen_*'")
    p.add_argument("--primary", default=None, help="覆盖排名任务（默认取 v1.compare.primary）")
    p.add_argument("--n-boot", type=int, default=None, help="bootstrap 次数")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
