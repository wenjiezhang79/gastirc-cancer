"""v3 比较报告：融合模型 vs 最佳单模态 vs 后期融合（及各消融），按**预先规定的判定规则**给出结论。

判定规则（docs/decisions.md 第十四节第 7 小节，2026-11-01 与研究者确认；写在 v3.compare，结果出来后不得修改）：
- 主结论任务（v3.compare.primary，默认 Lauren、HER2、PD-L1 CPS≥5、Claudin18.2、OS）的主要指标
  （宏平均 AUC / AUC / C-index）取平均；
- "最佳单模态"：每个任务分别取病理、CT 中折外点估计较好者；"后期融合"：每个任务取平均、堆叠中较好者（保守）；
- 融合 − 最佳单模态、融合 − 后期融合，两个配对 bootstrap 差值（按病人重抽样 v3.compare.n_boot 次）的 95% CI 下限都 > 0，
  才称"明显超过"；只在两种模态都有的病例上判定（三类模型在每例都有预测），种子平均后的折外预测；
- MSI、EBV（v3.compare.noninferiority.tasks）不进主结论平均；做非劣效检查：融合 − 病理单模态的差值 95% CI 下限 ≥ −margin。
各任务的差值、全部病例上的结果只作参考。

命令行：
    python -m gastric_mm.fusion.compare --config configs/server.yaml --main fusion_main \\
        --late late_mean,late_stack --others fusion_concat,fusion_scratch
    # 单模态实验默认取 v3.init.path_exp / ct_exp（--path-exp / --ct-exp 可指定）；
    # --others 中的名字默认在 fusion 命名空间，也可写 path_mtl/<名>、ct_mtl/<名>

输出：<output_dir>/fusion_compare/<时间>/compare_report.md、compare.json、compare_tasks.csv
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import metrics as M
from ..config import load_config, resolve_path
from ..mtl.spec import spec_from_section
from ..path_mtl.common import load_labels_and_split
from ..path_mtl.evaluate import merged
from ..training.task_spec import TaskSpec
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.runio import Run
from .common import NAMESPACE, SECTION, group_of

METRIC_NAME = {"multiclass": "宏平均AUC", "binary": "AUC", "survival": "C-index"}


def _exp_dir(cfg, ref: str, default_ns: str):
    ns, name = (ref.split("/", 1) if "/" in ref else (default_ns, ref))
    d = resolve_path(cfg, cfg.paths.output_dir) / ns / name
    if not (d / "oof_predictions.csv").exists():
        raise UserError(f"实验 {ns}/{name} 没有 oof_predictions.csv（交叉验证尚未汇总，或实验名写错）",
                        "请先跑完该实验的全部折；--others 中的名字默认在 fusion 下，单模态实验写成 path_mtl/<名> 或 ct_mtl/<名>。")
    return f"{ns}/{name}", d


def task_kind(spec: TaskSpec, t: str) -> str:
    return "multiclass" if t in spec.multiclass else ("binary" if t in spec.binary else "survival")


class TaskArrays:
    """某模型、某任务在共同病例上的数组，便于 bootstrap 快速计算。"""

    def __init__(self, df: pd.DataFrame, spec: TaskSpec, t: str):
        self.kind = task_kind(spec, t)
        if self.kind == "multiclass":
            cols = [f"{t}_p{k}" for k in range(spec.multiclass[t])]
            self.a = (df[t].to_numpy(float), df[cols].to_numpy(float) if all(c in df for c in cols)
                      else np.full((len(df), len(cols)), np.nan))
        elif self.kind == "binary":
            self.a = (df[t].to_numpy(float), df[f"{t}_prob"].to_numpy(float) if f"{t}_prob" in df
                      else np.full(len(df), np.nan))
        else:
            self.a = (df[f"{t}_time"].to_numpy(float), df[f"{t}_event"].to_numpy(float),
                      df[f"{t}_risk"].to_numpy(float) if f"{t}_risk" in df else np.full(len(df), np.nan))

    def value(self, idx=None) -> float:
        a = self.a if idx is None else tuple(x[idx] for x in self.a)
        if self.kind == "multiclass":
            return M.multiclass_auc(a[0], a[1])[0]
        if self.kind == "binary":
            return M.auc_binary(a[0], a[1])
        return M.c_index(a[0], a[1], a[2])


def boot_matrix(arrays: Dict[str, Dict[str, TaskArrays]], tasks: List[str], n: int, n_boot: int,
                seed: int = 0) -> Dict[str, np.ndarray]:
    """{模型: [B, 任务数]}，所有模型共用同一批重抽样下标（配对）。"""
    out = {m: np.full((n_boot, len(tasks)), np.nan) for m in arrays}
    for b, idx in enumerate(M.bootstrap_indices(n, n_boot, seed)):
        for m, d in arrays.items():
            for j, t in enumerate(tasks):
                out[m][b, j] = d[t].value(idx)
    return out


def ci(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.3f}"


def fmt_ci(d, lo, hi) -> str:
    return f"{fmt(d)}（{fmt(lo)} 至 {fmt(hi)}）"


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = cfg[SECTION]
    cc = sec.get("compare", {}) or {}
    init = sec.get("init", {}) or {}
    run = Run(cfg, "fusion_compare", args.run_name)
    log = run.logger
    pname, cname = args.path_exp or init.get("path_exp"), args.ct_exp or init.get("ct_exp")
    if not pname or not cname:
        raise UserError("比较需要 v1 病理与 v2 CT 单模态实验", "请用 --path-exp / --ct-exp 指定，或在 v3.init 中填写。")
    refs = {"main": _exp_dir(cfg, args.main, NAMESPACE), "path": _exp_dir(cfg, pname, "path_mtl"),
            "ct": _exp_dir(cfg, cname, "ct_mtl")}
    late = [_exp_dir(cfg, x.strip(), NAMESPACE) for x in (args.late or "").split(",") if x.strip()]
    others = [_exp_dir(cfg, x.strip(), NAMESPACE) for x in (args.others or "").split(",") if x.strip()]
    if not late:
        raise UserError("没有指定后期融合实验（--late）", "判定规则要求同时与后期融合比较；请先运行 32_v3_late_fusion.sh。")
    models: Dict[str, Tuple[str, object]] = {"融合": refs["main"], "病理单模态": refs["path"], "CT 单模态": refs["ct"]}
    late_keys = []
    for i, (lab, d) in enumerate(late):
        key = f"后期融合:{lab.split('/')[-1]}"
        models[key] = (lab, d)
        late_keys.append(key)
    other_keys = []
    for lab, d in others:
        key = f"对照:{lab}"
        models[key] = (lab, d)
        other_keys.append(key)
    oofs = {k: pd.read_csv(d / "oof_predictions.csv", dtype={"study_id": str}) for k, (_, d) in models.items()}
    df = load_labels_and_split(cfg, log)
    lab = df[df.set == "train"].drop(columns=["fold"])
    spec_all = spec_from_section(cfg, SECTION)
    main_oof = oofs["融合"]
    tasks = [t for t in spec_all.all_tasks if t not in spec_all.regression
             and (f"{t}_prob" in main_oof or f"{t}_risk" in main_oof or f"{t}_p0" in main_oof)]
    primary = [t for t in cc.get("primary", []) if t in tasks]
    missing_primary = [t for t in cc.get("primary", []) if t not in tasks]
    if missing_primary:
        raise UserError(f"融合实验 {args.main} 没有主结论任务 {missing_primary} 的预测",
                        "主结论必须用多任务融合模型（含全部主结论任务）；单任务模型请放在 --others 中。")
    ni = cc.get("noninferiority", {}) or {}
    ni_tasks = [t for t in ni.get("tasks", []) if t in tasks]
    margin = float(ni.get("margin", 0.05))
    n_boot = int(cc.get("n_boot", 2000))

    # ---- 共同病例：两种模态都有，且所有模型都有预测 ----
    grp = group_of(main_oof)
    both_ids = set(main_oof.loc[grp == "both", "study_id"])
    common = set(both_ids)
    for k, o in oofs.items():
        common &= set(o.study_id)
    lost = len(both_ids) - len(common)
    common = sorted(common)
    if len(common) < 20:
        raise UserError(f"两种模态都有、且所有模型都有预测的病例只有 {len(common)} 例，无法比较",
                        "请确认各实验使用同一划分、同一清单，且都已跑完全部折。")
    log.info("共同病例（两种模态都有）%d 例%s；bootstrap %d 次", len(common), f"（另有 {lost} 例缺某个模型的预测）" if lost else "",
             n_boot)
    arrays: Dict[str, Dict[str, TaskArrays]] = {}
    for k, o in oofs.items():
        d = merged(o[o.study_id.isin(common)].set_index("study_id").loc[common].reset_index(), lab)
        arrays[k] = {t: TaskArrays(d, spec_all, t) for t in tasks}
    point = {k: {t: arrays[k][t].value() for t in tasks} for k in arrays}
    boots = boot_matrix(arrays, tasks, len(common), n_boot)
    ti = {t: j for j, t in enumerate(tasks)}

    best_single = {t: ("病理单模态" if np.nan_to_num(point["病理单模态"][t], nan=-1) >= np.nan_to_num(point["CT 单模态"][t], nan=-1)
                       else "CT 单模态") for t in tasks}
    best_late = {t: max(late_keys, key=lambda k: np.nan_to_num(point[k][t], nan=-1)) for t in tasks}

    def comp(model_of_task: Dict[str, str], ts: List[str]) -> Tuple[float, np.ndarray]:
        p = float(np.mean([point[model_of_task[t]][t] for t in ts]))
        b = np.mean(np.stack([boots[model_of_task[t]][:, ti[t]] for t in ts], 1), 1)
        return p, b

    main_of = {t: "融合" for t in tasks}
    pm, bm = comp(main_of, primary)
    ps, bs = comp(best_single, primary)
    pl, bl = comp(best_late, primary)
    d1, d2 = bm - bs, bm - bl
    lo1, hi1 = ci(d1)
    lo2, hi2 = ci(d2)
    passed = bool(np.isfinite(lo1) and np.isfinite(lo2) and lo1 > 0 and lo2 > 0)

    # ---- 非劣效（MSI、EBV） ----
    ni_rows = []
    for t in ni_tasks:
        d = boots["融合"][:, ti[t]] - boots["病理单模态"][:, ti[t]]
        lo, hi = ci(d)
        diff = point["融合"][t] - point["病理单模态"][t]
        if np.isfinite(lo) and lo >= -margin:
            verdict = "通过（没有明显变差）"
        elif np.isfinite(hi) and hi < -margin:
            verdict = "不通过（融合明显较差；临床上建议用病理模型的预测）"
        else:
            verdict = "证据不足（置信区间过宽，无法判断）"
        ni_rows.append({"task": t, "diff": diff, "lo": lo, "hi": hi, "verdict": verdict,
                        "n_pos": int(np.nansum(arrays["融合"][t].a[0] == 1)) if task_kind(spec_all, t) == "binary" else None})

    # ---- 各任务并列 ----
    task_rows = []
    for t in tasks:
        r = {"task": t, "metric": METRIC_NAME[task_kind(spec_all, t)], "primary": t in primary,
             "noninferiority": t in ni_tasks}
        for k in models:
            r[k] = point[k][t]
        for name, ref in (("最佳单模态", best_single[t]), ("后期融合", best_late[t])):
            d = boots["融合"][:, ti[t]] - boots[ref][:, ti[t]]
            lo, hi = ci(d)
            r[f"融合−{name}"] = point["融合"][t] - point[ref][t]
            r[f"融合−{name}_lo"], r[f"融合−{name}_hi"] = lo, hi
            r[f"{name}_取自"] = ref
        task_rows.append(r)
    pd.DataFrame(task_rows).to_csv(run.path("compare_tasks.csv"), index=False, encoding="utf-8-sig")

    # ---- 对照 / 消融 ----
    other_rows = []
    for k in other_keys:
        ts = [t for t in primary if np.isfinite(point[k][t])]
        if len(ts) == len(primary):
            po, bo = comp({t: k for t in tasks}, primary)
            lo, hi = ci(bm - bo)
            other_rows.append({"model": k, "composite": po, "diff": pm - po, "lo": lo, "hi": hi})
        else:
            other_rows.append({"model": k, "composite": float("nan"), "diff": float("nan"), "lo": float("nan"),
                               "hi": float("nan"), "note": "不含全部主结论任务（如单任务模型），只看各任务"})

    # ---- 全部病例（各模型在自己有预测的病例上） ----
    all_rows = {}
    for k, o in oofs.items():
        d = merged(o, lab)
        all_rows[k] = {t: (TaskArrays(d, spec_all, t).value(), int(len(d))) for t in tasks}

    # ---- 报告 ----
    names = list(models)
    L = ["# v3 比较报告（内部训练集折外预测）", "",
         f"- 融合：{refs['main'][0]}；病理单模态：{refs['path'][0]}；CT 单模态：{refs['ct'][0]}；"
         f"后期融合：{', '.join(l for l, _ in late)}" + (f"；对照：{', '.join(l for l, _ in others)}" if others else ""),
         f"- 共同病例：两种模态都有、且所有模型都有预测的 {len(common)} 例" + (f"（{lost} 例因缺某个模型的预测未纳入）" if lost else "")
         + f"；配对 bootstrap {n_boot} 次（按病人重抽样）。", "",
         "## 预先规定的判定规则（2026-11-01 确认，结果出来后不得修改）", "",
         f"- 主结论任务：{', '.join(primary)}；取各任务主要指标（宏平均 AUC / AUC / C-index）的平均。",
         "- 最佳单模态 = 每个任务取病理、CT 中较好者；后期融合 = 每个任务取各后期融合方法中较好者（均从保守出发）。",
         "- 融合 − 最佳单模态、融合 − 后期融合两个差值的 95% CI 下限都 > 0，才称“融合明显超过”。",
         f"- {', '.join(ni_tasks)} 不进主结论；非劣效检查：融合 − 病理单模态的差值 95% CI 下限 ≥ −{margin}。", "",
         "## 主结论", "",
         "| 比较 | 主结论任务平均 | 差值（95% CI） |", "|---|---|---|",
         f"| 融合 | {fmt(pm)} | |",
         f"| 最佳单模态 | {fmt(ps)} | 融合 − 最佳单模态：{fmt_ci(pm - ps, lo1, hi1)} |",
         f"| 后期融合 | {fmt(pl)} | 融合 − 后期融合：{fmt_ci(pm - pl, lo2, hi2)} |", "",
         f"**结论：{'融合明显超过最佳单模态与后期融合（两个差值的 95% CI 下限均 > 0）' if passed else '未达到“融合明显超过”的预先规定标准'}。**",
         ""]
    if ni_rows:
        L += ["## 非劣效检查（相对病理单模态）", "", "| 任务 | 阳性数 | 融合 − 病理（95% CI） | 结论 |", "|---|---|---|---|"]
        for r in ni_rows:
            L.append(f"| {r['task']} | {'' if r['n_pos'] is None else r['n_pos']} | {fmt_ci(r['diff'], r['lo'], r['hi'])} | {r['verdict']} |")
        L += ["", "阳性少时置信区间较宽，可能因“证据不足”而无法判断；外部队列以合并结果为主要参考。", ""]
    L += ["## 各任务并列（两种模态都有的病例；只作参考）", "",
          "| 任务 | 指标 | 主结论 | " + " | ".join(names) + " | 融合 − 最佳单模态（95% CI） | 融合 − 后期融合（95% CI） |",
          "|" + "---|" * (len(names) + 5)]
    for r in task_rows:
        L.append(f"| {r['task']} | {r['metric']} | {'是' if r['primary'] else ('非劣效' if r['noninferiority'] else '')} | "
                 + " | ".join(fmt(r[k]) for k in names)
                 + f" | {fmt_ci(r['融合−最佳单模态'], r['融合−最佳单模态_lo'], r['融合−最佳单模态_hi'])}（{r['最佳单模态_取自']}）"
                 + f" | {fmt_ci(r['融合−后期融合'], r['融合−后期融合_lo'], r['融合−后期融合_hi'])} |")
    L.append("")
    if other_rows:
        L += ["## 对照与消融（主结论任务平均，与融合主模型配对比较）", "",
              "| 模型 | 主结论任务平均 | 融合主模型 − 该模型（95% CI） | 说明 |", "|---|---|---|---|"]
        for r in other_rows:
            L.append(f"| {r['model']} | {fmt(r['composite'])} | {fmt_ci(r['diff'], r['lo'], r['hi'])} | {r.get('note', '')} |")
        L += ["", "消融只用于说明和讨论，不用来更换主模型（多任务权重除外，见 docs/decisions.md 第十四节第 4 小节）。", ""]
    L += ["## 全部病例上的结果（各模型在自己有预测的病例上；样本不同，不做配对比较）", "",
          "| 任务 | " + " | ".join(f"{k}（n）" for k in names) + " |", "|" + "---|" * (len(names) + 1)]
    for t in tasks:
        L.append(f"| {t} | " + " | ".join(f"{fmt(all_rows[k][t][0])}（{all_rows[k][t][1]}）" for k in names) + " |")
    L.append("")
    run.write_text("\n".join(L), "compare_report.md")
    out = {"common_n": len(common), "primary": primary, "composite": {"融合": pm, "最佳单模态": ps, "后期融合": pl},
           "diff_vs_best_single": {"diff": pm - ps, "lo": lo1, "hi": hi1},
           "diff_vs_late": {"diff": pm - pl, "lo": lo2, "hi": hi2}, "passed": passed,
           "best_single_by_task": best_single, "best_late_by_task": best_late, "noninferiority": ni_rows,
           "tasks": task_rows, "others": other_rows, "n_boot": n_boot}
    (run.dir / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info("主结论任务平均：融合 %.3f；最佳单模态 %.3f；后期融合 %.3f", pm, ps, pl)
    log.info("融合 − 最佳单模态 %.3f（%.3f 至 %.3f）；融合 − 后期融合 %.3f（%.3f 至 %.3f）→ %s", pm - ps, lo1, hi1,
             pm - pl, lo2, hi2, "明显超过" if passed else "未达到预先规定的标准")
    log.info("报告：%s", run.path("compare_report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v3 比较报告：融合 vs 最佳单模态 vs 后期融合（预先规定的判定规则），并列出各消融。",
                    cases=False, workers=False)
    p.add_argument("--main", required=True, help="融合主模型实验名（fusion 下）")
    p.add_argument("--late", required=True, help="后期融合实验名，逗号分隔（如 late_mean,late_stack）")
    p.add_argument("--others", default=None, help="其他对照 / 消融实验，逗号分隔（默认在 fusion 下；也可写 path_mtl/<名>）")
    p.add_argument("--path-exp", default=None, help="v1 病理实验名（默认 v3.init.path_exp）")
    p.add_argument("--ct-exp", default=None, help="v2 CT 实验名（默认 v3.init.ct_exp）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
