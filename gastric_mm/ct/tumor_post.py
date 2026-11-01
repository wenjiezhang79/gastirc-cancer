"""② nnU-Net 推理结果后处理：贴回原图网格 → 连通域 → 用中心点挑出肿瘤 → 候选分割 + 待审核清单。

逐例：
1. 读取 ``<work_dir>/ct/nnunet/predict_output/<病例名>.nii.gz``，按裁剪框贴回 native 网格；
2. 连通域分析，去掉小于 v2.tumor_seg.min_component_ml 的碎块；
3. 选连通域：包含中心点的 → 否则离中心点最近、且距离 ≤ max_point_dist_mm 的 → 否则"未找到肿瘤"；
   没有中心点时取最大连通域（并提示）；
4. 写出 ``<work_dir>/ct/tumor_auto/<study_id>_tumor.nii.gz``（只保留选中的连通域）与质控图。

所有自动分割都需要医生审核：在 ITK-SNAP / 3D Slicer 中打开 native 图像与候选分割，修改后另存到审核目录，
再运行 ``import_reviewed`` 导入（mask_source=auto_reviewed）。待审核清单按"有提示的优先"排序。

命令行：
    python -m gastric_mm.ct.tumor_post --config configs/server.yaml

输出：``<work_dir>/ct/tumor_auto/``（含 review_list.csv）、``<work_dir>/ct/qc/tumor_auto/``；
运行目录 status.csv、review_list.csv、overview_tumor_auto_pNN.png。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..config import Config, load_config
from ..manifest import load_manifest, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run, write_csv
from ..utils.runner import exit_code, process_cases
from . import qc as QC
from . import transforms as T
from .nnunet_data import box_path, case_id, nnunet_dirs

QC_WINDOW = (40.0, 400.0)
FLAG_ZH = {"no_point": "无中心点（取最大连通域）", "nearest": "中心点不在分割内（取最近连通域）",
           "not_found": "中心点附近未找到肿瘤", "multi": "另有较大的连通域", "volume": "体积异常",
           "touches_box": "分割触及裁剪框边界（可能不完整）", "empty": "模型没有分出肿瘤"}


def paste_back(pred, native, box: dict):
    """把裁剪框内的预测贴回 native 网格（uint8）。"""
    import SimpleITK as sitk

    start, size = box["start_xyz"], box["size_xyz"]
    pa = sitk.GetArrayFromImage(pred)
    if list(pa.shape[::-1]) != list(size):
        raise CaseError(f"预测结果大小 {list(pa.shape[::-1])} 与裁剪框 {list(size)} 不一致（推理输入是否被重新生成过？）")
    out = np.zeros(native.GetSize()[::-1], dtype=np.uint8)
    x0, y0, z0 = start
    sx, sy, sz = size
    out[z0:z0 + sz, y0:y0 + sy, x0:x0 + sx] = (pa > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(native)
    return img


def select_component(mask, native, point_idx, cfg: Config, box: dict) -> dict:
    """返回 {mask, flags, n_components, point_dist_mm, ml}。"""
    import SimpleITK as sitk

    ts = cfg.v2.tumor_seg
    vox_ml = float(np.prod(native.GetSpacing())) / 1000.0
    min_vox = max(1, int(np.ceil(float(ts.get("min_component_ml", 0.5)) / vox_ml)))
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(mask, True), minimumObjectSize=min_vox)
    arr = sitk.GetArrayFromImage(cc)
    n = int(arr.max())
    flags: List[str] = []
    out = {"n_components": n, "point_dist_mm": None, "flags": flags}
    if n == 0:
        flags.append("empty")
        out.update(mask=None, ml=0.0)
        return out
    chosen = None
    if point_idx is not None:
        pi = np.clip(np.round(point_idx).astype(int), 0, np.array(native.GetSize()) - 1)
        lab = int(arr[pi[2], pi[1], pi[0]])
        if lab > 0:
            chosen, dist = lab, 0.0
        else:
            p = T.index_to_physical(native, point_idx)
            best = (None, np.inf)
            for lb in range(1, n + 1):
                comp = sitk.Cast(cc == lb, sitk.sitkUInt8)
                d = T.distance_to_mask_mm(comp, p)
                if d is not None and d < best[1]:
                    best = (lb, d)
            chosen, dist = best
            if chosen is None or dist > float(ts.get("max_point_dist_mm", 15)):
                flags.append("not_found")
                out.update(mask=None, ml=0.0, point_dist_mm=None if chosen is None else round(float(dist), 1))
                return out
            flags.append("nearest")
        out["point_dist_mm"] = round(float(dist), 1)
    else:
        chosen = 1   # RelabelComponent 按体积从大到小编号
        flags.append("no_point")
    sel = (arr == chosen).astype(np.uint8)
    ml = float(sel.sum()) * vox_ml
    others = [float((arr == lb).sum()) * vox_ml for lb in range(1, n + 1) if lb != chosen]
    if others and max(others) >= max(1.0, 0.2 * ml):
        flags.append("multi")
    lo, hi = (float(v) for v in ts.get("warn_volume_ml", [1.0, 400.0]))
    if not lo <= ml <= hi:
        flags.append("volume")
    x0, y0, z0 = box["start_xyz"]
    sx, sy, sz = box["size_xyz"]
    full = np.array(native.GetSize())
    zyx = np.argwhere(sel > 0)
    lo_i, hi_i = zyx.min(0)[::-1], zyx.max(0)[::-1]
    b_lo, b_hi = np.array([x0, y0, z0]), np.array([x0 + sx - 1, y0 + sy - 1, z0 + sz - 1])
    touch = ((lo_i <= b_lo) & (b_lo > 0)) | ((hi_i >= b_hi) & (b_hi < full - 1))
    if touch.any():
        flags.append("touches_box")
    img = sitk.GetImageFromArray(sel)
    img.CopyInformation(native)
    out.update(mask=img, ml=round(ml, 2))
    return out


def post_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    bp = box_path(cfg, sid)
    pred_p = Path(case["_pred_dir"]) / f"{case_id(sid)}.nii.gz"
    if not bp.exists() or not pred_p.exists():
        return {"status": "skipped", "message": "没有该例的推理结果"}
    out_p = wp.ct_tumor_auto(sid)
    if out_p.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "已有候选分割"}
    native = sitk.ReadImage(str(wp.ct_native(sid)))
    box = json.loads(bp.read_text(encoding="utf-8"))
    if list(box.get("native_size_xyz", native.GetSize())) != list(native.GetSize()):
        raise CaseError("native 图像与生成推理输入时不一致（是否重新转换过？），请重新运行 --mode predict")
    full = paste_back(sitk.ReadImage(str(pred_p)), native, box)
    pt = [case.get(f"ct_point_{a}") for a in "xyz"]
    point_idx = pt if all(v is not None for v in pt) else None
    r = select_component(full, native, point_idx, cfg, box)
    res = {"auto_ml": r["ml"], "n_components": r["n_components"], "point_dist_mm": r["point_dist_mm"],
           "flags": " ".join(r["flags"])}
    if r["mask"] is not None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(r["mask"], str(out_p), True)
    elif out_p.exists():
        out_p.unlink()
    st_p = wp.ct_stomach(sid)
    st = sitk.GetArrayFromImage(sitk.ReadImage(str(st_p))) if st_p.exists() else None
    m = sitk.GetArrayFromImage(r["mask"]) if r["mask"] is not None else sitk.GetArrayFromImage(full)
    if point_idx is not None:
        center = [int(round(v)) for v in point_idx]
    else:
        c = T.mask_centroid_physical(r["mask"] if r["mask"] is not None else full)
        center = [int(round(v)) for v in T.physical_to_index(native, c)] if c is not None else \
            [s // 2 for s in native.GetSize()]
    arr = T.window_normalize(sitk.GetArrayFromImage(native), *QC_WINDOW, "minmax")
    title = f"{sid}  auto={r['ml']:g}ml  {' '.join(r['flags']) or 'ok'}"
    QC.case_qc(arr, native.GetSpacing(), center, wp.ct_qc("tumor_auto", sid), title, m, st,
               point_xyz=center if point_idx is not None else None)
    res["_center_z"] = center[2]
    if any(f != "nearest" for f in r["flags"]):
        res["status"] = "warning"
    if r["flags"]:
        res["message"] = "；".join(FLAG_ZH.get(f, f) for f in r["flags"])
    return res


def _run(args) -> int:
    import SimpleITK as sitk

    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_tumor_post", args.run_name)
    log = run.logger
    wp = WorkPaths(cfg)
    d = nnunet_dirs(cfg)
    pred_dir = Path(args.pred_dir) if args.pred_dir else d["predict_output"]
    if not pred_dir.is_dir() or not any(pred_dir.glob("*.nii.gz")):
        raise UserError("没有找到 nnU-Net 推理结果",
                        "请先运行 nnunet_data --mode predict，并在 nnU-Net 环境中运行 nnunet_predict.sh（或加 --run）。")
    res = load_manifest(cfg, log, check_paths=False)
    cases = [c for c in to_cases(res.df) if box_path(cfg, c["study_id"]).exists()]
    for c in cases:
        c.update(_pred_dir=str(pred_dir), _overwrite=args.overwrite)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, post_case, cfg, log, 1, "肿瘤分割后处理")
    run.write_table(results, "status.csv", ["study_id", "status", "auto_ml", "n_components", "point_dist_mm",
                                            "flags", "message"])
    rows = []
    for r in results:
        if r.get("status") == "skipped" and "推理结果" in r.get("message", ""):
            continue
        fl = r.get("flags", "")
        pri = 1 if any(f in fl for f in ("not_found", "empty", "multi", "volume", "touches_box", "no_point")) else \
            (2 if "nearest" in fl else 3)
        rows.append({"study_id": r["study_id"], "priority": pri, "status": r.get("status"), "auto_ml": r.get("auto_ml"),
                     "n_components": r.get("n_components"), "point_dist_mm": r.get("point_dist_mm"), "flags": fl,
                     "message": r.get("message", "")})
    rows.sort(key=lambda x: (x["priority"], x["study_id"]))
    cols = ["study_id", "priority", "status", "auto_ml", "n_components", "point_dist_mm", "flags", "message"]
    run.write_table(rows, "review_list.csv", cols)
    write_csv(rows, wp.ct / "tumor_auto" / "review_list.csv", cols)
    items = []
    for r in results:
        sid = r["study_id"]
        if r.get("status") == "failed" or "_center_z" not in r:
            continue
        native = sitk.ReadImage(str(wp.ct_native(sid)))
        z = int(np.clip(r["_center_z"], 0, native.GetSize()[2] - 1))
        img = T.window_normalize(sitk.GetArrayFromImage(native)[z], *QC_WINDOW, "minmax")
        mp = wp.ct_tumor_auto(sid)
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(mp)))[z] if mp.exists() else None
        items.append((sid, img, m, "!" if r.get("status") == "warning" else ""))
    pages = QC.overview(items, run.dir / "overview_tumor_auto", int(cfg.ct.qc.overview_per_page)) if items else []
    n1 = sum(1 for r in rows if r["priority"] == 1)
    log.info("候选分割 %d 例，其中有提示需优先审核 %d 例。审核清单：%s；总览图 %d 页",
             sum(1 for r in rows if r.get("auto_ml")), n1, wp.ct / "tumor_auto" / "review_list.csv", len(pages))
    log.info("审核后把分割另存到一个目录（文件名 <study_id>_tumor.nii.gz），再运行 import_reviewed 导入。")
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("nnU-Net 推理结果后处理：贴回原图、按中心点选肿瘤连通域、生成候选分割与审核清单。", workers=False)
    p.add_argument("--pred-dir", default=None, help="推理结果目录（默认 <work_dir>/ct/nnunet/predict_output）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
