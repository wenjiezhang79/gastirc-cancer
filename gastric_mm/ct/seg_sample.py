"""人工勾画病例抽样：从**内部训练集**中按 cT 分期（可加其他变量）分层抽取 200–300 例，供医生三维勾画肿瘤，
用于训练第二阶段 nnU-Net。

规则：
- 只从划分文件中 set=train 的病例里抽（内部测试集、外部中心一律不用）；
- 需要有 CT；默认只抽已标中心点、且肿瘤可见度不是 invisible 的病例；
- 已有人工勾画（mask_source=manual）的训练集病例直接计入名单，其余名额按分层比例分配（最大余数法），
  可用 v2.seg_sample.min_per_group 给少见的 cT 分组（如 T1）设最少例数；
- 名单一旦生成默认不覆盖（医生已经按名单开始勾画），确需重抽加 --overwrite（旧名单备份为 .bak）。

命令行：
    python -m gastric_mm.ct.seg_sample --config configs/server.yaml

输出：
- ``<work_dir>/ct/seg_sample.csv``：study_id、cT 分组、可见度、Borrmann、是否已有勾画（只含研究编号）
- 运行目录 seg_sample_report.md：抽样前后各分组的分布对照
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..labels import derive_labels
from ..manifest import load_manifest
from ..paths import WorkPaths
from ..split import load_split
from ..utils.cli import base_parser, run_main
from ..utils.errors import UserError
from ..utils.runio import Run

STRATA_COLUMNS = ("ct_t_group", "borrmann", "tumor_visibility")


def _allocate(sizes: Dict[str, int], k: int, minimum: Dict[str, int]) -> Dict[str, int]:
    """把 k 个名额按各层大小比例分配（最大余数法），并满足各层最少例数（不超过该层大小）。"""
    alloc = {s: min(int(minimum.get(s, 0)), n) for s, n in sizes.items()}
    left = k - sum(alloc.values())
    if left < 0:
        raise UserError(f"min_per_group 要求的最少例数之和（{sum(alloc.values())}）超过了要抽的例数 {k}",
                        "请调小 v2.seg_sample.min_per_group，或调大 v2.seg_sample.n。")
    rest = {s: n - alloc[s] for s, n in sizes.items()}
    tot = sum(rest.values())
    if tot == 0 or left == 0:
        return alloc
    quota = {s: left * r / tot for s, r in rest.items()}
    base = {s: int(np.floor(q)) for s, q in quota.items()}
    rem = left - sum(base.values())
    order = sorted(quota, key=lambda s: -(quota[s] - base[s]))
    for s in order[:rem]:
        base[s] += 1
    return {s: alloc[s] + min(base[s], rest[s]) for s in sizes}


def sample_cases(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """df：内部训练集候选病例（含 ct_t_group 等列与 already_manual）。返回入选名单。"""
    sc = cfg.v2.seg_sample
    n = int(sc.n)
    by = list(sc.get("stratify_by", ["ct_t_group"]) or [])
    for c in by:
        if c not in STRATA_COLUMNS:
            raise UserError(f"v2.seg_sample.stratify_by 中的 {c} 无效", f"可用：{', '.join(STRATA_COLUMNS)}")
    rng = np.random.RandomState(int(sc.get("seed", 0)))
    df = df.copy()
    df["stratum"] = df[by].astype(object).where(df[by].notna(), "NA").astype(str).agg("|".join, axis=1) if by else "all"
    fixed = df[df["already_manual"]]
    pool = df[~df["already_manual"]]
    k = max(0, n - len(fixed))
    if k > len(pool):
        raise UserError(f"符合条件的候选病例只有 {len(pool)} 例（另有已勾画 {len(fixed)} 例），不够抽 {n} 例",
                        "请调小 v2.seg_sample.n，或检查中心点、可见度是否已填写。")
    minimum: Dict[str, int] = {}
    mg = dict(sc.get("min_per_group", {}) or {})
    if mg and "ct_t_group" in by:
        # 最少例数按 cT 分组给出；多变量分层时按各层在该分组内的比例拆分
        for g, m in mg.items():
            layers = pool[pool["ct_t_group"] == g]["stratum"].value_counts()
            if len(layers) == 0:
                continue
            got = fixed[fixed["ct_t_group"] == g]
            need = max(0, int(m) - len(got))
            sub = _allocate(layers.to_dict(), min(need, int(layers.sum())), {})
            for s, v in sub.items():
                minimum[s] = minimum.get(s, 0) + v
    sizes = pool["stratum"].value_counts().to_dict()
    alloc = _allocate(sizes, k, minimum)
    chosen: List[str] = []
    for s, m in sorted(alloc.items()):
        ids = pool.loc[pool["stratum"] == s, "study_id"].tolist()
        if m:
            chosen += list(rng.choice(ids, size=m, replace=False))
    out = pd.concat([fixed, pool[pool["study_id"].isin(chosen)]])
    return out.sort_values("study_id").reset_index(drop=True)


def _report(cand: pd.DataFrame, chosen: pd.DataFrame, cfg: Config, n_excluded: dict) -> str:
    L = ["# 人工勾画病例抽样报告", "",
         f"- 抽样范围：内部训练集；入选 {len(chosen)} 例（其中已有人工勾画 {int(chosen['already_manual'].sum())} 例）",
         f"- 分层变量：{', '.join(cfg.v2.seg_sample.get('stratify_by', []) or ['无'])}；随机种子 {cfg.v2.seg_sample.get('seed')}",
         "- 未纳入候选：" + "，".join(f"{k} {v} 例" for k, v in n_excluded.items() if v) + "", ""]
    for col in ("ct_t_group", "borrmann", "tumor_visibility"):
        a = cand[col].fillna("缺失").value_counts()
        b = chosen[col].fillna("缺失").value_counts()
        L += [f"## {col}", "", "| 取值 | 候选 | 入选 | 入选比例 |", "|---|---|---|---|"]
        for v in sorted(set(a.index) | set(b.index), key=str):
            na, nb = int(a.get(v, 0)), int(b.get(v, 0))
            L.append(f"| {v} | {na} | {nb} | {nb / na * 100:.0f}% |" if na else f"| {v} | {na} | {nb} | — |")
        L.append("")
    L += ["名单文件只含研究编号，可直接交给影像科医生；勾画规范见 docs/ct_annotation_protocol.md。", ""]
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_seg_sample", args.run_name)
    log = run.logger
    wp = WorkPaths(cfg)
    out_path = wp.ct_seg_sample
    if out_path.exists() and not args.overwrite:
        raise UserError("勾画病例名单已存在（ct/seg_sample.csv），医生可能已按名单开始勾画",
                        "如确需重新抽样，加 --overwrite（旧名单备份为 .bak）。")
    res = load_manifest(cfg, log, check_paths=False)
    df = res.df.copy()
    lab = derive_labels(df, cfg)[["study_id", "ct_t_group"]]
    split = load_split(cfg)[["study_id", "set"]]
    df = df.merge(lab, on="study_id", how="left").merge(split, on="study_id", how="left")
    sc = cfg.v2.seg_sample
    excl = {}
    m_train = df["set"] == "train"
    excl["非内部训练集"] = int((~m_train).sum())
    d = df[m_train]
    m_ct = d["ct_dir"].notna()
    excl["无 CT"] = int((~m_ct).sum())
    d = d[m_ct]
    bad_vis = set(sc.get("exclude_visibility", []) or [])
    m_vis = ~d["tumor_visibility"].isin(bad_vis)
    excl[f"可见度为 {'/'.join(sorted(bad_vis))}"] = int((~m_vis).sum())
    d = d[m_vis]
    if sc.get("require_point", True):
        m_pt = d[["ct_point_x", "ct_point_y", "ct_point_z"]].notna().all(axis=1)
        excl["未标中心点"] = int((~m_pt).sum())
        d = d[m_pt]
    d = d.copy()
    d["already_manual"] = (d["mask_source"] == "manual") & d["tumor_mask_path"].notna()
    n_ct = int(d["ct_t_group"].notna().sum())
    if n_ct < len(d) and "ct_t_group" in (sc.get("stratify_by") or []):
        log.warning("候选病例中有 %d 例没有结构化 cT（ct_t 列），按“缺失”单独成层", len(d) - n_ct)
    chosen = sample_cases(d, cfg)
    cols = ["study_id", "ct_t_group", "tumor_visibility", "borrmann", "already_manual"]
    out = chosen[cols].copy()
    out["already_manual"] = out["already_manual"].map({True: 1, False: 0})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.replace(out_path.with_suffix(".csv.bak"))
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    out.to_csv(run.path("seg_sample.csv"), index=False, encoding="utf-8-sig")
    run.write_text(_report(d, chosen, cfg, excl), "seg_sample_report.md")
    log.info("入选 %d 例（已有勾画 %d 例）。名单：%s", len(out), int(out["already_manual"].sum()), out_path)
    return 0


def main(argv=None) -> int:
    p = base_parser("从内部训练集按 cT 分期分层抽取人工勾画病例（训练 nnU-Net 用）。", cases=False, workers=False)
    p.add_argument("--overwrite", action="store_true", help="覆盖已有名单（旧名单备份为 .bak）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
