"""比较多个 v2 CT 实验（编码器初筛、影像组学基线、消融），按预先规定的指标排名。

排名指标（v2.compare.primary，预先规定，默认 [os, lauren]）：折外 OS C-index 与 Lauren 宏平均 AUC 的平均。
只在所有实验共有的病例上比较；给出 95% CI 与"与第一名的配对 bootstrap 差值"（同一批抽样）。
辅助参考（v2.compare.auxiliary，默认 cT3–4 vs cT1–2 的 AUC）与其他任务只报告，不参与排名。
注意：影像组学只覆盖有分割的病例，与深度模型一起比较时"共有病例"会变少；报告中会注明病例数。

命令行：
    python -m gastric_mm.ct_mtl.compare --config configs/server.yaml --pattern "screen_*"
    python -m gastric_mm.ct_mtl.compare --config configs/server.yaml --exps screen_radimagenet,screen_ctfm,radiomics_bw25

输出：<output_dir>/ct_mtl/_compare/<时间>/compare_report.md、compare.csv
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import load_config
from ..mtl.experiment import exp_root, read_meta
from ..path_mtl.common import load_labels_and_split
from ..path_mtl.evaluate import merged, primary_value
from ..training.task_spec import TaskSpec
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.log import setup_logging
from ..utils.runio import write_csv

NAMESPACE = "ct_mtl"


def _score_fn(spec, tasks):
    def fn(df):
        vals = [primary_value(df, spec, t) for t in tasks]
        return float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
    return fn


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    root = exp_root(cfg, NAMESPACE)
    if args.exps:
        names = [x.strip() for x in args.exps.split(",") if x.strip()]
    elif args.pattern:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.pattern)) \
            if root.exists() else []
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
    primary = [x.strip() for x in (args.primary.split(",") if args.primary else cfg.v2.compare.primary)]
    aux = [x.strip() for x in (cfg.v2.compare.get("auxiliary", []) or [])]
    n_boot = int(args.n_boot or cfg.v2.get("eval", {}).get("n_boot", 1000))
    labels = load_labels_and_split(cfg, log).drop(columns=["fold"])
    data, specs, metas = {}, {}, {}
    for n in ready:
        metas[n] = read_meta(root / n)
        specs[n] = TaskSpec.from_dict(metas[n]["spec"])
        miss = [t for t in primary if t not in specs[n].all_tasks]
        if miss:
            raise UserError(f"实验 {n} 没有训练任务 {miss}，无法按 {primary} 比较")
        data[n] = merged(pd.read_csv(root / n / "oof_predictions.csv", dtype={"study_id": str}), labels)
    common = set.intersection(*[set(df.study_id) for df in data.values()])
    for n in ready:
        data[n] = data[n][data[n].study_id.isin(common)].sort_values("study_id").reset_index(drop=True)
    log.info("比较 %d 个实验，共有病例 %d 例；排名指标：%s 的平均", len(ready), len(common), "+".join(primary))
    ref = data[ready[0]]
    strata_col = next((t for t in primary if t in ref and t not in specs[ready[0]].survival), None)
    strata = ref[strata_col].fillna(-1).to_numpy() if strata_col else None
    idx_list = list(M.bootstrap_indices(len(ref), n_boot, seed=0, strata=strata))
    boots: Dict[str, np.ndarray] = {}
    points: Dict[str, float] = {}
    for n in ready:
        fn = _score_fn(specs[n], primary)
        points[n] = fn(data[n])
        boots[n] = np.array([fn(data[n].iloc[j]) for j in idx_list])
    order = sorted(ready, key=lambda n: -points[n] if np.isfinite(points[n]) else np.inf)
    best = order[0]
    rows: List[dict] = []
    for rank, n in enumerate(order, 1):
        b = boots[n]
        ok = np.isfinite(b)
        lo, hi = np.percentile(b[ok], [2.5, 97.5]) if ok.sum() > 10 else (np.nan, np.nan)
        diff = boots[n] - boots[best]
        okd = np.isfinite(diff)
        dlo, dhi = np.percentile(diff[okd], [2.5, 97.5]) if okd.sum() > 10 and n != best else (np.nan, np.nan)
        r = {"rank": rank, "experiment": n, "encoder": metas[n].get("encoder"), "kind": metas[n].get("kind"),
             "score": points[n], "lo": lo, "hi": hi, "diff_vs_best": points[n] - points[best], "diff_lo": dlo,
             "diff_hi": dhi}
        for t in primary + aux:
            r[t] = primary_value(data[n], specs[n], t) if t in specs[n].all_tasks else float("nan")
        rows.append(r)
    write_csv(rows, out / "compare.csv")
    L = ["# CT 实验比较", "", f"- 排名指标（预先规定）：{' + '.join(primary)} 主要指标的平均（OS 为 C-index，Lauren 为宏平均 AUC），基于折外预测",
         f"- 共有病例：{len(common)}；bootstrap {n_boot} 次（各实验使用同一批抽样）",
         f"- 未完成、未纳入比较：{', '.join(not_ready) or '无'}", "",
         "| 排名 | 实验 | 编码器 | 排名指标 (95% CI) | " + " | ".join(primary) + " | 与第一名之差 (95% CI) |",
         "|---|---|---|---|" + "---|" * len(primary) + "---|"]
    for r in rows:
        d = "—" if r["experiment"] == best else f"{r['diff_vs_best']:+.3f}（{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}）"
        L.append(f"| {r['rank']} | {r['experiment']} | {r['encoder']} | {r['score']:.3f}（{r['lo']:.3f}–{r['hi']:.3f}） | "
                 + " | ".join(f"{r[t]:.3f}" for t in primary) + f" | {d} |")
    L += ["", "差值置信区间不包含 0，说明与第一名的差距超出了随机波动；包含 0 则不能认为两者有差别。", ""]
    if aux:
        L += ["## 辅助参考（不参与排名）", "", "cT 分期由影像科医生在同一份 CT 上判读；编码器连 cT3–4 与 cT1–2 都区分不了，"
              "说明它可能没有“看到”肿瘤。", "", "| 实验 | " + " | ".join(aux) + " |", "|---|" + "---|" * len(aux)]
        for r in rows:
            L.append(f"| {r['experiment']} | " + " | ".join("—" if not np.isfinite(r[t]) else f"{r[t]:.3f}" for t in aux) + " |")
        L.append("")
    other = sorted({t for n in ready for t in specs[n].all_tasks} - set(primary) - set(aux))
    L += ["## 其他任务的折外指标（仅供参考，不参与排名）", ""]
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
    p = base_parser("比较多个 v2 CT 实验（编码器初筛、影像组学基线等），按预先规定的指标排名。", cases=False, workers=False)
    p.add_argument("--exps", default=None, help="实验名列表，逗号分隔")
    p.add_argument("--pattern", default=None, help="实验名通配符，如 'screen_*'")
    p.add_argument("--primary", default=None, help="覆盖排名任务（默认取 v2.compare.primary）")
    p.add_argument("--n-boot", type=int, default=None, help="bootstrap 次数")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
