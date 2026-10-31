"""DICOM 信息汇总：扫描所有 CT，汇总期相、层厚、层间距、厂家、卷积核、像素间距等。

**不输出任何患者身份信息**：只读取白名单字段（见 dicom_io.HEADER_TAGS），不输出姓名、ID、日期、
UID、机构和文件路径。病例只用 study_id 标识。

命令行：
    python -m gastric_mm.ct.summary --config configs/server.yaml            # 按清单的 ct_dir 扫描
    python -m gastric_mm.ct.summary --config configs/server.yaml --root /path/to/ct_root
        # 直接扫描根目录下的每个子文件夹（每个子文件夹视为一个检查），自动编号 scan-0001…，
        # 编号与文件夹名的对照表写在运行目录的 private/ 下，不要外传。

输出（运行目录）：
- dicom_series.csv     每个序列一行
- dicom_selected.csv   每例被选为分析序列的那一个
- dicom_summary.md     各中心分布、问题清单、推荐的重采样体素
- dicom_summary.png    层厚与像素间距分布图
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run, write_csv
from ..utils.runner import exit_code, process_cases, resolve_workers
from .dicom_io import SeriesInfo, scan_dicom_dir, select_series, select_thin_series, same_acquisition

SERIES_COLUMNS = [
    "study_id", "center", "series_idx", "series_number", "selected", "thin_for_annotation", "phase", "phase_basis",
    "n_images", "rows", "cols", "pixel_spacing_mm", "slice_thickness_mm", "slice_spacing_mm", "spacing_uniform",
    "n_gaps", "z_coverage_mm", "manufacturer", "model", "kernel", "kernel_sharp", "kvp", "contrast_agent",
    "contrast_delay_s", "rel_acq_time_s", "acq_group", "frame_of_ref", "gantry_tilt", "localizer", "axial",
    "series_description", "select_reason",
]


def scan_case(case: dict, cfg: Config) -> dict:
    ct = cfg.ct
    d = resolve_path(cfg, case["ct_dir"])
    if d is None or not d.is_dir():
        raise CaseError("ct_dir 不存在")
    series = scan_dicom_dir(d, ct.phase_keywords, ct.phase_delay_s, float(ct.slice_spacing_tolerance))
    if not series:
        raise CaseError("文件夹中没有可读取的 DICOM 图像")
    try:
        chosen, reason = select_series(series, ct.phase, float(ct.preferred_thickness_mm), int(ct.min_slices),
                                       case.get("ct_series_number"))
    except CaseError as e:
        chosen, reason = None, str(e)
    thin = select_thin_series(series, chosen, int(ct.min_slices)) if chosen is not None else None
    rows = series_rows(series, case, cfg, chosen, thin, reason)
    res = {"series_rows": rows, "n_series": len(series)}
    if chosen is None:
        res["status"] = "failed"
        res["message"] = reason
    else:
        res["message"] = f"共 {len(series)} 个序列，选中序列 {chosen.series_number}（{chosen.phase}，" \
                         f"{_fmt(chosen.slice_thickness)} mm）"
    return res


def _fmt(v, nd=2):
    if v is None:
        return ""
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def series_rows(series: List[SeriesInfo], case: dict, cfg: Config, chosen: Optional[SeriesInfo],
                thin: Optional[SeriesInfo], reason: str) -> List[dict]:
    include_desc = bool(cfg.get_path("ct.summary.include_series_description", True))
    t0 = min((s.acq_time_s for s in series if s.acq_time_s is not None), default=None)
    # 采集分组：同一坐标系 + 采集时间接近
    groups: List[SeriesInfo] = []
    gid: Dict[str, str] = {}
    for s in series:
        for i, g in enumerate(groups):
            if same_acquisition(s, g):
                gid[s.uid] = chr(ord("A") + i)
                break
        else:
            groups.append(s)
            gid[s.uid] = chr(ord("A") + len(groups) - 1)
    fors = list(OrderedDict.fromkeys(s.for_hash for s in series))
    rows = []
    for i, s in enumerate(series):
        rows.append({
            "study_id": case["study_id"], "center": case.get("center"), "series_idx": i + 1,
            "series_number": s.series_number, "selected": "Y" if chosen is not None and s.uid == chosen.uid else "",
            "thin_for_annotation": "Y" if thin is not None and s.uid == thin.uid else "",
            "phase": s.phase, "phase_basis": s.phase_basis, "n_images": s.n_images, "rows": s.rows, "cols": s.cols,
            "pixel_spacing_mm": s.pixel_spacing[1] if s.pixel_spacing else None,
            "slice_thickness_mm": s.slice_thickness, "slice_spacing_mm": s.spacing_median,
            "spacing_uniform": "Y" if s.uniform else "N", "n_gaps": s.n_gaps, "z_coverage_mm": s.z_coverage,
            "manufacturer": s.manufacturer, "model": s.model, "kernel": s.kernel,
            "kernel_sharp": "Y" if s.kernel_sharp else "", "kvp": s.kvp,
            "contrast_agent": "Y" if s.contrast_agent else "N", "contrast_delay_s": s.contrast_delay_s,
            "rel_acq_time_s": (s.acq_time_s - t0) if (s.acq_time_s is not None and t0 is not None) else None,
            "acq_group": gid.get(s.uid, ""), "frame_of_ref": f"F{fors.index(s.for_hash) + 1}" if s.for_hash else "",
            "gantry_tilt": s.tilt, "localizer": "Y" if s.is_localizer else "", "axial": "Y" if s.axial else "N",
            "series_description": s.description if include_desc else "",
            "select_reason": reason if chosen is not None and s.uid == chosen.uid else "",
        })
    return rows


# ---------------- 汇总 ----------------
def _round_th(v) -> str:
    if v is None or v != v:
        return "未知"
    return f"{round(float(v) * 4) / 4:g}"


def build_summary(all_rows: List[dict], results: List[dict], cfg: Config) -> str:
    sel = [r for r in all_rows if r["selected"] == "Y"]
    centers = [c for c in cfg.manifest.centers if any(r["center"] == c for r in all_rows)] or \
        sorted({r["center"] for r in all_rows if r["center"]})
    L = ["# DICOM 信息汇总", "",
         f"- 扫描病例：{len(results)}；成功选出分析序列：{len(sel)}；序列总数：{len(all_rows)}",
         f"- 目标期相：{cfg.ct.phase}；优先层厚：{cfg.ct.preferred_thickness_mm} mm", ""]

    def table(title, key, rows_src, fmt=lambda x: x):
        L.append(f"### {title}")
        L.append("")
        vals = sorted({fmt(r[key]) for r in rows_src}, key=lambda x: str(x))
        L.append("| 值 | " + " | ".join(centers) + " | 合计 |")
        L.append("|---|" + "---|" * (len(centers) + 1))
        for v in vals:
            cells = [sum(1 for r in rows_src if r["center"] == c and fmt(r[key]) == v) for c in centers]
            L.append(f"| {v} | " + " | ".join(str(x) for x in cells) + f" | {sum(cells)} |")
        L.append("")

    L += ["## 被选中的分析序列", ""]
    table("层厚（mm，按 0.25 取整）", "slice_thickness_mm", sel, _round_th)
    table("层间距（mm）", "slice_spacing_mm", sel, _round_th)
    table("期相", "phase", sel)
    table("厂家", "manufacturer", sel, lambda x: x or "未知")
    table("卷积核", "kernel", sel, lambda x: x or "未知")
    L.append("### 像素间距（mm）")
    L.append("")
    L.append("| 中心 | 例数 | 中位数 | 最小 | 最大 |")
    L.append("|---|---|---|---|---|")
    for c in centers:
        v = [r["pixel_spacing_mm"] for r in sel if r["center"] == c and r["pixel_spacing_mm"] is not None]
        if v:
            L.append(f"| {c} | {len(v)} | {np.median(v):.3f} | {min(v):.3f} | {max(v):.3f} |")
    L.append("")

    L += ["## 全部序列", ""]
    table("期相（全部序列，不含定位像）", "phase", [r for r in all_rows if not r["localizer"]])
    thin_n = Counter(r["center"] for r in all_rows if r["thin_for_annotation"] == "Y")
    L.append("同一次采集中另有薄层软组织重建的病例数：" + "，".join(f"{c} {thin_n.get(c, 0)}" for c in centers))
    L.append("")

    # 问题
    L += ["## 需要注意的病例", ""]
    probs: "OrderedDict[str, List[str]]" = OrderedDict()
    for r in results:
        if r.get("status") == "failed":
            probs.setdefault(f"未选出分析序列：{r.get('message', '')}", []).append(r["study_id"])
    for r in sel:
        if "人工核对" in (r["select_reason"] or ""):
            probs.setdefault("未识别出目标期相，按层厚/层数选择，需人工核对", []).append(r["study_id"])
        if r["spacing_uniform"] == "N" or (r["n_gaps"] or 0) > 0:
            probs.setdefault("层间距不均匀或缺层", []).append(r["study_id"])
        if r["gantry_tilt"]:
            probs.setdefault("机架倾斜", []).append(r["study_id"])
        if r["kernel_sharp"]:
            probs.setdefault("分析序列为锐利卷积核", []).append(r["study_id"])
        if r["phase_basis"] == "延迟时间":
            probs.setdefault("期相由造影延迟时间推断（序列描述中没有关键词）", []).append(r["study_id"])
    if probs:
        L.append("| 问题 | 例数 | 病例（最多 30 个） |")
        L.append("|---|---|---|")
        for k, ids in probs.items():
            L.append(f"| {k} | {len(ids)} | {', '.join(ids[:30])}{' ...' if len(ids) > 30 else ''} |")
    else:
        L.append("无。")
    L.append("")

    # 推荐体素
    if sel:
        ps = [r["pixel_spacing_mm"] for r in sel if r["pixel_spacing_mm"]]
        th = [_round_th(r["slice_thickness_mm"]) for r in sel]
        mode_th, n_mode = Counter(th).most_common(1)[0]
        L += ["## 推荐的重采样体素", "",
              f"- 层面内：所选序列像素间距中位数 {np.median(ps):.3f} mm → 建议 {round(np.median(ps) * 20) / 20:.2f} mm",
              f"- 层厚：最常见 {mode_th} mm（{n_mode}/{len(sel)} 例）→ 建议 z 方向采用该层厚，薄层病例多层平均降到该层厚",
              f"- 当前配置 ct.target_spacing = {list(cfg.ct.target_spacing)}", ""]
        per_c = {c: Counter(_round_th(r["slice_thickness_mm"]) for r in sel if r["center"] == c) for c in centers}
        modes = {c: cnt.most_common(1)[0][0] for c, cnt in per_c.items() if cnt}
        if len(set(modes.values())) > 1:
            L.append("- **注意：各中心最常见层厚不同**（" + "，".join(f"{c} {m} mm" for c, m in modes.items())
                     + "）。建议统一到各中心都有的最粗层厚，避免外部验证时层厚不一致。")
            L.append("")
    return "\n".join(L)


def plot_summary(sel: List[dict], centers: List[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for c in centers:
        th = [r["slice_thickness_mm"] for r in sel if r["center"] == c and r["slice_thickness_mm"] is not None]
        ps = [r["pixel_spacing_mm"] for r in sel if r["center"] == c and r["pixel_spacing_mm"] is not None]
        if th:
            axes[0].hist(th, bins=np.arange(0, 10.5, 0.25), alpha=0.6, label=c)
        if ps:
            axes[1].hist(ps, bins=np.arange(0.4, 1.3, 0.025), alpha=0.6, label=c)
    axes[0].set_xlabel("slice thickness (mm)")
    axes[1].set_xlabel("pixel spacing (mm)")
    for a in axes:
        a.set_ylabel("cases")
        if a.get_legend_handles_labels()[0]:
            a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "dicom_summary", args.run_name)
    log = run.logger
    if args.root:
        root = Path(args.root)
        if not root.is_dir():
            raise UserError("--root 指定的文件夹不存在", "请检查路径。")
        subs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        cases = [{"study_id": f"scan-{i + 1:04d}", "center": "", "ct_dir": str(p)} for i, p in enumerate(subs)]
        write_csv([{"scan_id": c["study_id"], "folder": Path(c["ct_dir"]).name} for c in cases],
                  run.private_dir / "scan_id_map.csv")
        log.info("根目录模式：%d 个子文件夹；编号对照表在 private/（不要外传）", len(cases))
        if args.limit:
            cases = cases[: args.limit]
    else:
        res = load_manifest(cfg, log)
        miss = missing_path_set(res)
        cases = [c for c in to_cases(res.df) if c.get("ct_dir") and (c["study_id"], "ct_dir") not in miss]
        cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, scan_case, cfg, log, resolve_workers(args, cfg), "DICOM 扫描")
    all_rows = [r for res_ in results for r in res_.get("series_rows", [])]
    run.write_table(all_rows, "dicom_series.csv", SERIES_COLUMNS)
    sel = [r for r in all_rows if r["selected"] == "Y"]
    run.write_table(sel, "dicom_selected.csv", SERIES_COLUMNS)
    run.write_table([{k: v for k, v in r.items() if k != "series_rows"} for r in results], "status.csv",
                    ["study_id", "status", "n_series", "message", "elapsed_s"])
    run.write_text(build_summary(all_rows, results, cfg), "dicom_summary.md")
    centers = sorted({r["center"] for r in sel}, key=lambda c: (list(cfg.manifest.centers) + [c]).index(c))
    if sel:
        plot_summary(sel, centers, run.path("dicom_summary.png"))
    log.info("汇总完成：%s", run.path("dicom_summary.md"))
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("扫描 CT 的 DICOM 头信息，汇总期相、层厚、厂家、像素间距等（不输出患者身份信息）。")
    p.add_argument("--root", default=None, help="不用清单，直接扫描该根目录下的每个子文件夹")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
