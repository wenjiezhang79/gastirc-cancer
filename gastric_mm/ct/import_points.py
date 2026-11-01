"""导入医生标注的肿瘤中心点，换算为原始 NIfTI 的体素坐标（ct_point_x/y/z，从 0 开始）。

医生不需要手抄坐标。标注文件放在 ct.points.annotation_dir 下，以 study_id 命名，支持：
- 3D Slicer 标记点：``<study_id>.mrk.json``（新版）或 ``<study_id>.fcsv``（旧版）；
  恰好一个点。坐标系 RAS / LPS 自动识别。
- ITK-SNAP 标签：``<study_id>_point.nii.gz``：在肿瘤中心涂一小团标签，取其质心。

标注可以在 native/ 或 native_thin/（薄层）上完成：都按物理坐标换算到 native/ 的体素坐标。

输出（运行目录）：
- private/manifest_with_points.csv   原清单的副本（含原始路径，不要外传），已填入导入的坐标（**不会修改原清单**，核对后请手动替换）
- points_report.csv          每例来源、坐标、是否在图像内、与原坐标的距离等

命令行：
    python -m gastric_mm.ct.import_points --config configs/server.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, read_manifest_csv, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from . import transforms as T


def read_slicer_json(path: Path) -> np.ndarray:
    """读取 Slicer .mrk.json，返回 LPS 物理坐标。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise CaseError(f"标记点文件不是有效的 JSON：{e}") from None
    pts = []
    for mk in data.get("markups", []):
        cs = str(mk.get("coordinateSystem", "LPS")).upper()
        for cp in mk.get("controlPoints", []):
            pos = cp.get("position")
            if pos is None or len(pos) != 3:
                continue
            p = np.array(pos, dtype=float)
            if cs == "RAS":
                p[:2] *= -1
            pts.append(p)
    if len(pts) != 1:
        raise CaseError(f"标记点文件中应恰好有 1 个点，实际 {len(pts)} 个")
    return pts[0]


def read_fcsv(path: Path) -> np.ndarray:
    """读取 Slicer 旧版 .fcsv。默认 RAS；表头注明 LPS（或 CoordinateSystem = 1）时按 LPS。"""
    cs = "RAS"
    pts = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#"):
            if "CoordinateSystem" in line:
                v = line.split("=", 1)[-1].strip().upper()
                cs = "LPS" if v in ("LPS", "1") else "RAS"
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                pts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
            except ValueError:
                continue
    if len(pts) != 1:
        raise CaseError(f".fcsv 中应恰好有 1 个点，实际 {len(pts)} 个")
    p = pts[0]
    if cs == "RAS":
        p[:2] *= -1
    return p


def read_label_point(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    m = sitk.ReadImage(str(path))
    c = T.mask_centroid_physical(m)
    if c is None:
        raise CaseError("点标签文件中没有标注（全为 0）")
    return c


def find_annotation(ann_dir: Path, sid: str) -> Optional[Tuple[str, Path]]:
    for kind, name in (("slicer_json", f"{sid}.mrk.json"), ("slicer_fcsv", f"{sid}.fcsv"),
                       ("label", f"{sid}_point.nii.gz"), ("label", f"{sid}_point.nii")):
        p = ann_dir / name
        if p.exists():
            return kind, p
    return None


def import_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    ann_dir = resolve_path(cfg, cfg.ct.points.annotation_dir)
    found = find_annotation(ann_dir, sid)
    if found is None:
        return {"status": "skipped", "message": "没有标注文件"}
    kind, path = found
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换（03_ct_convert.sh）")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(native_p))
    reader.ReadImageInformation()
    ref = T.make_reference(reader.GetOrigin(), reader.GetSpacing(), reader.GetSize(), reader.GetDirection())
    if kind == "slicer_json":
        p = read_slicer_json(path)
    elif kind == "slicer_fcsv":
        p = read_fcsv(path)
    else:
        p = read_label_point(path)
    idx_c = T.physical_to_index(ref, p)
    idx = np.round(idx_c).astype(int)
    ok = T.inside(ref, idx_c)
    res = {"source": kind, "point_x": int(idx[0]), "point_y": int(idx[1]), "point_z": int(idx[2]),
           "inside_image": "Y" if ok else "N"}
    prev = [case.get(f"ct_point_{a}") for a in "xyz"]
    if all(v is not None for v in prev):
        pp = T.index_to_physical(ref, prev)
        res["dist_to_previous_mm"] = round(float(np.linalg.norm(pp - p)), 1)
    if not ok:
        res["status"] = "failed"
        res["message"] = "标注点不在 native 图像范围内（可能标在了其他检查或未转换的图像上）"
        return res
    res["message"] = f"{kind} → ({idx[0]}, {idx[1]}, {idx[2]})"
    return res


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "import_points", args.run_name)
    log = run.logger
    ann_dir = resolve_path(cfg, cfg.ct.points.annotation_dir)
    if not ann_dir.is_dir():
        raise UserError("找不到标注文件目录（配置项 ct.points.annotation_dir）",
                        "请把医生的标注文件（<study_id>.mrk.json 或 <study_id>_point.nii.gz）放到该目录。")
    res = load_manifest(cfg, log, check_paths=False)
    cases = select_cases(to_cases(res.df), parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, import_case, cfg, log, 1, "导入中心点")
    run.write_table(results, "points_report.csv",
                    ["study_id", "status", "source", "point_x", "point_y", "point_z", "inside_image",
                     "dist_to_previous_mm", "message"])
    raw = read_manifest_csv(resolve_path(cfg, cfg.paths.manifest))
    n = 0
    by_id = {r["study_id"]: r for r in results if r.get("status") in ("ok", "warning")}
    for i, sid in raw["study_id"].items():
        r = by_id.get(sid)
        if r:
            for a in "xyz":
                raw.at[i, f"ct_point_{a}"] = str(r[f"point_{a}"])
            n += 1
    out = run.private_dir / "manifest_with_points.csv"  # 清单含原始路径，放在 private/
    raw.to_csv(out, index=False, encoding="utf-8-sig")
    moved = [r for r in results if (r.get("dist_to_previous_mm") or 0) > 10]
    if moved:
        log.warning("%d 例新导入的点与清单中原有的点相距 >10 mm：%s，请核对",
                    len(moved), ", ".join(r["study_id"] for r in moved))
    log.info("已导入 %d 例。核对 points_report.csv 后，用 private/manifest_with_points.csv 替换原清单。", n)
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("导入医生在 Slicer / ITK-SNAP 中标注的肿瘤中心点，生成带坐标的清单副本。", workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
