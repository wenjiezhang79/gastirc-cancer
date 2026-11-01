"""早停诊断：每个任务"自己最好在第几个 epoch" vs "实际停在第几个 epoch"，以及停早（或停晚）的代价。

背景（docs/v3_diagnosis.md）：早停监测的是内层验证集上**各任务损失的平均**（CLAUDE.md 6.5 第 15 条）。
有信号的任务往往要训练较久才到最好，没有信号的任务几轮后就开始过拟合；平均之后，停止时间可能被后者拉早。
本工具只读取各实验已有的训练记录（seed*/fold*/history.csv 与 done.json，仅含内部训练集交叉验证的信息），
不读取任何特征、标签、测试集或外部队列；v1（path_mtl）、v2（ct_mtl）、v3（fusion）的实验都可以一起列出。

命令行：
    python -m gastric_mm.fusion.diagnose --config configs/server.yaml --exps path_mtl/uni2h,ct_mtl/ct_rad,fusion_main
    # 名字不带前缀时默认在 fusion 下

输出：<output_dir>/fusion_diagnose/<时间>/diagnose_report.md、diagnose.csv
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.runio import Run
from .common import NAMESPACE


def _exp_dir(cfg, ref: str):
    ns, name = (ref.split("/", 1) if "/" in ref else (NAMESPACE, ref))
    d = resolve_path(cfg, cfg.paths.output_dir) / ns / name
    runs = sorted(d.glob("seed*/fold*/history.csv"))
    if not runs:
        raise UserError(f"实验 {ns}/{name} 没有训练记录（seed*/fold*/history.csv）",
                        "请确认实验名正确且至少完成了一个折；不带前缀的名字默认在 fusion 下，单模态写成 path_mtl/<名> 或 ct_mtl/<名>。")
    return f"{ns}/{name}", d


def diagnose_experiment(d) -> pd.DataFrame:
    """每个（种子, 折）× 任务一行：该任务验证损失最低点的 epoch、实际停止 epoch、停止时比最低点高出的损失。"""
    rows = []
    for h_path in sorted(d.glob("seed*/fold*/history.csv")):
        done = h_path.parent / "done.json"
        if not done.exists():
            continue
        h = pd.read_csv(h_path)
        best = int(json.loads(done.read_text(encoding="utf-8"))["best_epoch"])
        n_ep = len(h)
        mon = h["val_monitor"].to_numpy(float)
        after = mon[min(best + 5, n_ep - 1)] - mon[best]          # 停止点之后 5 个 epoch 总损失上升多少（过拟合速度）
        for c in h.columns:
            if not c.startswith("val_") or c == "val_monitor":
                continue
            v = h[c].to_numpy(float)
            if not np.isfinite(v).any():
                continue
            j = int(np.nanargmin(v))
            rows.append({"seed": h_path.parent.parent.name, "fold": h_path.parent.name, "task": c[4:],
                         "stop_epoch": best, "task_best_epoch": j, "n_epochs": n_ep,
                         "loss_at_stop": float(v[best]), "loss_min": float(v[j]),
                         "excess": float(v[best] - v[j]), "excess_rel": float((v[best] - v[j]) / max(abs(v[j]), 1e-8)),
                         "monitor_rise_5": float(after)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("task", sort=False)
    out = pd.DataFrame({
        "stop_epoch": g["stop_epoch"].mean(),
        "task_best_epoch": g["task_best_epoch"].mean(),
        "task_best_min": g["task_best_epoch"].min(),
        "task_best_max": g["task_best_epoch"].max(),
        "gap": g.apply(lambda x: float((x["task_best_epoch"] - x["stop_epoch"]).mean())),
        "excess_rel": g["excess_rel"].mean(),
        "n_runs": g.size(),
    })
    return out.reset_index()


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "fusion_diagnose", args.run_name)
    log = run.logger
    refs = [_exp_dir(cfg, x.strip()) for x in args.exps.split(",") if x.strip()]
    tasks_order: List[str] = []
    summ: Dict[str, pd.DataFrame] = {}
    allrows = []
    stop_info = {}
    for lab, d in refs:
        df = diagnose_experiment(d)
        if df.empty:
            raise UserError(f"实验 {lab} 没有已完成的折（缺少 done.json）", "请等至少一个折训练完成后再诊断。")
        df.insert(0, "experiment", lab)
        allrows.append(df)
        s = summarize(df)
        summ[lab] = s.set_index("task")
        for t in s.task:
            if t not in tasks_order:
                tasks_order.append(t)
        first = df.drop_duplicates(["seed", "fold"])
        stop_info[lab] = (float(first.stop_epoch.mean()), float(first.n_epochs.mean()), float(first.monitor_rise_5.mean()),
                          int(len(first)))
    pd.concat(allrows, ignore_index=True).to_csv(run.path("diagnose.csv"), index=False, encoding="utf-8-sig")
    names = [lab for lab, _ in refs]
    L = ["# 早停诊断", "",
         "- 只读取训练记录（内部训练集交叉验证），不涉及测试集与外部队列。",
         "- **实际停止**：内层验证总损失（各任务损失平均）最低的 epoch；**任务最好**：该任务自己的内层验证损失最低的 epoch。",
         "- **差距** = 任务最好 − 实际停止（正数表示该任务还没训练够就停了）；**代价** = 停止时该任务验证损失比其最低点高出的比例。", "",
         "## 各实验的停止情况", "", "| 实验 | 折数 | 实际停止 epoch（均值） | 训练 epoch 数（均值） | 停止后 5 个 epoch 总损失上升 |", "|---|---|---|---|---|"]
    for lab in names:
        s, n, r, k = stop_info[lab]
        L.append(f"| {lab} | {k} | {s:.1f} | {n:.1f} | {r:+.3f} |")
    L += ["", "## 每个任务最好的 epoch（均值；括号内为差距 / 代价）", "",
          "| 任务 | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for t in tasks_order:
        cells = []
        for lab in names:
            if t in summ[lab].index:
                r = summ[lab].loc[t]
                cells.append(f"{r.task_best_epoch:.1f}（{r.gap:+.1f} / {100 * r.excess_rel:.1f}%）")
            else:
                cells.append("—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "解读：某任务差距为较大的正数、代价较高，说明总损失早停对它不利（停得太早）；"
          "无信号的任务通常最好的 epoch 很早，会把总损失的最低点拉早。", ""]
    run.write_text("\n".join(L), "diagnose_report.md")
    log.info("报告：%s", run.path("diagnose_report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("早停诊断：各任务自己最好的 epoch vs 实际停止的 epoch（只读训练记录）。", cases=False, workers=False)
    p.add_argument("--exps", required=True, help="实验名，逗号分隔；默认在 fusion 下，也可写 path_mtl/<名>、ct_mtl/<名>")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
