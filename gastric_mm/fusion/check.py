"""v3 模态可用性检查：每例有没有病理特征、CT 肿瘤 ROI、CT 区域 ROI，缺失时尽量写出原因；按中心、人群汇总。

为什么要做：缺失往往与中心、人群有关（例如某中心多数病人在外院做 CT）。融合模型对缺失模态用"屏蔽"处理，
避免把"缺没缺"当成捷径；本报告让研究者看清楚哪些病例缺了什么、缺失比例在各中心是否明显不同。
只检查特征文件是否存在、读取清单中的路径列是否填写，不读取任何标签。

命令行：
    python -m gastric_mm.fusion.check --config configs/server.yaml

输出（运行目录 <output_dir>/fusion_check/<时间>/）：
    modality_availability.csv   逐例清单（只含 study_id，不含任何路径）
    availability_report.md      按集合、中心、人群汇总的缺失比例与原因
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_config
from ..manifest import load_manifest
from ..paths import WorkPaths
from ..split import load_split
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from .common import GROUP_NAMES, SECTION, availability, make_stores


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def reasons(row: pd.Series, avail: pd.Series, cfg, wp: WorkPaths, rois: List[str]) -> Dict[str, str]:
    out = {"path_reason": "", "ct_reason": ""}
    sec = cfg[SECTION]
    sid = row["study_id"]
    if not avail["has_path"]:
        if _blank(row.get("slide_path")):
            out["path_reason"] = "清单未填 slide_path（无切片）"
        else:
            miss = [e for e in sec.path_encoders if not wp.wsi_features(e, sid).exists()]
            out["path_reason"] = f"病理特征文件缺失（{', '.join(miss)}；切片未导入或提取失败）"
    if not avail["has_ct"]:
        if _blank(row.get("ct_dir")):
            out["ct_reason"] = "清单未填 ct_dir（无静脉期增强 CT，或尚未下载）"
        elif not wp.ct_native(sid).exists() and not wp.ct_features(sec.ct_encoder, sid).exists():
            out["ct_reason"] = "未生成 native NIfTI（DICOM 未转换，或没有可用的静脉期序列，见 DICOM 汇总）"
        elif not wp.ct_features(sec.ct_encoder, sid).exists():
            out["ct_reason"] = f"CT 特征未提取（编码器 {sec.ct_encoder}）"
        else:
            out["ct_reason"] = f"特征文件中没有所需 ROI（{', '.join(rois)}）"
    no_point = all(_blank(row.get(c)) for c in ("ct_point_x", "ct_point_y", "ct_point_z"))
    no_mask = _blank(row.get("tumor_mask_path"))
    for r in rois:
        key = f"{r}_reason"
        out[key] = ""
        if avail["has_ct"] and not avail[f"has_{r}"]:
            if r == "tumor" and str(row.get("tumor_visibility", "")).strip() == "invisible":
                out[key] = "肿瘤可见度为 invisible（CT 上看不到肿瘤）"
            elif no_point and no_mask:
                out[key] = "没有中心点与分割（无法定位 ROI）"
            else:
                out[key] = "该 ROI 特征缺失（裁剪或提取失败）"
    return out


def _pct(a, n) -> str:
    return f"{a}（{100 * a / n:.1f}%）" if n else "0"


def build_report(tab: pd.DataFrame, rois: List[str], cfg) -> str:
    sec = cfg[SECTION]
    L = ["# 模态可用性报告（v3 融合）", "",
         f"- 病理编码器：{', '.join(sec.path_encoders)}；CT 编码器：{sec.ct_encoder}；CT ROI：{', '.join(rois)}",
         "- 只检查特征文件是否存在与清单路径列是否填写，不读取标签；逐例结果见 modality_availability.csv（只含 study_id）。",
         "- 融合模型的病例池 = 内部训练集中至少有一种模态的病例；两种都没有的病例不进模型。", ""]
    L += ["## 按集合与中心", "",
          "| 集合 | 中心 | 例数 | 两种都有 | 只有病理 | 只有 CT | 两种都没有 | " + " | ".join(f"缺 {r} ROI（有 CT 者）" for r in rois) + " |",
          "|" + "---|" * (7 + len(rois))]
    warn = []
    ref = None
    for (st, ce), g in tab.groupby(["set", "center"], sort=True):
        n = len(g)
        c = g["group"].value_counts().to_dict()
        has_ct = g[g.has_ct == 1]
        roi_cells = [_pct(int((has_ct[f"has_{r}"] == 0).sum()), len(has_ct)) for r in rois]
        L.append(f"| {st} | {ce} | {n} | {_pct(c.get('both', 0), n)} | {_pct(c.get('path_only', 0), n)} | "
                 f"{_pct(c.get('ct_only', 0), n)} | {_pct(c.get('none', 0), n)} | " + " | ".join(roi_cells) + " |")
        if st == "train":
            ref = (1 - g.has_ct.mean(), 1 - g.has_path.mean())
    if ref is not None:
        for ce, g in tab[tab.set != "train"].groupby("center"):
            for name, val, r0 in (("CT", 1 - g.has_ct.mean(), ref[0]), ("病理", 1 - g.has_path.mean(), ref[1])):
                if abs(val - r0) > 0.10:
                    warn.append(f"- **提示：** 中心 {ce} 缺{name}比例 {100 * val:.1f}%，与内部训练集 {100 * r0:.1f}% 相差超过 10 个百分点。"
                                "缺失与中心有关时，融合模型用屏蔽处理可避免把缺失当捷径；外部结果请同时看按模态可用性分层的报告。")
    L += [""] + (warn + [""] if warn else [])
    if "population" in tab:
        L += ["## 内部训练集按人群", "", "| 人群 | 例数 | 两种都有 | 只有病理 | 只有 CT | 两种都没有 |", "|---|---|---|---|---|---|"]
        tr = tab[tab.set == "train"]
        for pop, g in tr.groupby(tab["population"].fillna("（空）")):
            c = g["group"].value_counts().to_dict()
            L.append(f"| {pop} | {len(g)} | " + " | ".join(_pct(c.get(k, 0), len(g)) for k in ("both", "path_only", "ct_only", "none")) + " |")
        L.append("")
    L += ["## 缺失原因（全部病例）", ""]
    for col, title in [("path_reason", "缺病理"), ("ct_reason", "缺 CT（整个）")] + [(f"{r}_reason", f"有 CT 但缺 {r} ROI") for r in rois]:
        cnt = Counter(v for v in tab[col] if v)
        if cnt:
            L.append(f"### {title}")
            L.append("")
            for k, v in cnt.most_common():
                L.append(f"- {k}：{v} 例")
            L.append("")
    L += ["## 各组含义", ""] + [f"- {k}：{v}" for k, v in GROUP_NAMES.items()] + [""]
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "fusion_check", args.run_name)
    log = run.logger
    res = load_manifest(cfg, log, check_paths=False)
    man = res.df
    split = load_split(cfg)[["study_id", "set", "fold"]]
    df = man.merge(split, on="study_id", how="left")
    df["set"] = df["set"].fillna("（不在划分中）")
    pstore, cstore = make_stores(cfg)
    rois = list(cstore.roles)
    avail = availability(list(df.study_id), pstore, cstore).set_index("study_id")
    wp = WorkPaths(cfg)
    rows = []
    for _, r in df.iterrows():
        a = avail.loc[r["study_id"]]
        d = {"study_id": r["study_id"], "center": r["center"], "set": r["set"], "population": r.get("population"),
             "has_path": int(a.has_path), **{f"has_{x}": int(a[f"has_{x}"]) for x in rois}, "has_ct": int(a.has_ct),
             "group": a.group}
        d.update(reasons(r, a, cfg, wp, rois))
        rows.append(d)
    tab = pd.DataFrame(rows)
    tab.to_csv(run.path("modality_availability.csv"), index=False, encoding="utf-8-sig")
    run.write_text(build_report(tab, rois, cfg), "availability_report.md")
    tr = tab[tab.set == "train"]
    c = tr["group"].value_counts().to_dict()
    log.info("内部训练集 %d 例：两种都有 %d、只有病理 %d、只有 CT %d、两种都没有 %d", len(tr), c.get("both", 0),
             c.get("path_only", 0), c.get("ct_only", 0), c.get("none", 0))
    log.info("报告：%s", run.path("availability_report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("v3 模态可用性检查：每例有哪些模态、缺失原因，按中心与人群汇总（不读取标签）。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
