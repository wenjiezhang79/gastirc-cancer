"""以肿瘤中心点（或分割质心）为中心的三维裁剪，窗宽窗位标准化，以及质控图。

按配置 ct.crop.specs 中的每个"裁剪规格"分别输出（体素大小、尺寸、窗宽、标准化方式、中心来源）：
- ``<work_dir>/ct/crops/<规格>/<study_id>_img.nii.gz``   float32，已标准化
- ``<work_dir>/ct/crops/<规格>/<study_id>_mask.nii.gz``  uint8（有肿瘤分割时）
- ``<work_dir>/ct/qc/<规格>/<study_id>.png``              单例三视图（青色十字=中心，红=肿瘤，黄=胃）
- 运行目录 ``overview_<规格>_pNN.png``                    总览拼图；``status.csv`` 质控指标

裁剪直接从 native 图像按物理坐标重采样到目标网格（只重采样 ROI），薄层先多层平均。
质控指标：中心点是否在图像内、需要填充的比例、中心点是否落在肿瘤分割内、中心点到胃分割的距离等。

命令行：
    python -m gastric_mm.ct.crop --config configs/server.yaml [--specs tumor_96mm]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases, resolve_workers
from . import qc as QC
from . import transforms as T

CENTER_MODES = ("point", "mask_centroid", "point_or_mask")


def _specs(cfg: Config, names: Optional[List[str]] = None) -> List[dict]:
    specs = list(cfg.ct.crop.specs)
    for s in specs:
        for k in ("name", "spacing", "size_vox", "window"):
            if k not in s:
                raise UserError(f"裁剪规格缺少 {k}", "请对照 configs/default.yaml 中 ct.crop.specs 的写法。")
        if s.get("center", "point_or_mask") not in CENTER_MODES:
            raise UserError(f"裁剪规格 {s['name']} 的 center 应为 {CENTER_MODES} 之一")
    if names:
        unknown = set(names) - {s["name"] for s in specs}
        if unknown:
            raise UserError(f"未知的裁剪规格：{', '.join(unknown)}", f"可用：{', '.join(s['name'] for s in specs)}")
        specs = [s for s in specs if s["name"] in names]
    return specs


def find_spec(cfg: Config, name: str) -> dict:
    """按名称查找裁剪规格：ct.crop.specs 与 v2.features.roi_specs（v2 编码器专用规格）。"""
    pool = list(cfg.ct.crop.specs) + list(cfg.get_path("v2.features.roi_specs", []) or [])
    for s in pool:
        if s.get("name") == name:
            return dict(s)
    raise UserError(f"找不到裁剪规格 {name}",
                    f"请在 ct.crop.specs 或 v2.features.roi_specs 中定义；现有：{', '.join(s.get('name', '?') for s in pool)}")


def case_centers(case: dict, cfg: Config, native):
    """返回 (中心点物理坐标或 None, 肿瘤分割质心物理坐标或 None)。"""
    import SimpleITK as sitk

    point_phys = None
    pt = [case.get(f"ct_point_{a}") for a in "xyz"]
    if all(v is not None for v in pt):
        if not T.inside(native, pt):
            raise CaseError(f"中心点 ({pt[0]}, {pt[1]}, {pt[2]}) 不在图像范围内 {native.GetSize()}")
        point_phys = T.index_to_physical(native, pt)
    mask_c = None
    mp = resolve_path(cfg, case.get("tumor_mask_path"))
    if mp is not None and mp.exists():
        mask_c = T.mask_centroid_physical(sitk.Cast(sitk.ReadImage(str(mp)) > 0, sitk.sitkUInt8))
    return point_phys, mask_c


def spec_center(spec: dict, point_phys, mask_c):
    """按规格的 center 方式选中心，返回 (中心, 来源) 或 (None, None)。"""
    mode = spec.get("center", "point_or_mask")
    if mode == "point":
        return (point_phys, "point") if point_phys is not None else (None, None)
    if mode == "mask_centroid":
        return (mask_c, "mask") if mask_c is not None else (None, None)
    if point_phys is not None:
        return point_phys, "point"
    return (mask_c, "mask") if mask_c is not None else (None, None)


def roi_array(native, center, spec: dict, cfg: Config):
    """从 native 图像按规格生成 ROI 数组 (z, y, x)（已按规格的窗宽与标准化方式处理）。返回 (数组, 填充比例)。"""
    import SimpleITK as sitk

    ref = T.roi_reference(center, spec["spacing"], spec["size_vox"])
    hu = T.resample_image(native, ref, default=float(cfg.ct.hu_clip[0]),
                          average_thin=bool(cfg.ct.get("average_thin_slices", True)))
    arr = sitk.GetArrayFromImage(hu)
    win = spec["window"]
    norm = T.window_normalize(arr, float(win["level"]), float(win["width"]), spec.get("normalize", "minmax"))
    return norm.astype(np.float32), T.outside_fraction(native, ref)


def crop_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    specs = case["_specs"]
    overwrite = bool(case.get("_overwrite"))
    if not overwrite and all(wp.ct_crop(s["name"], sid).exists() for s in specs):
        return {"status": "skipped", "message": "已有裁剪结果"}
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换（03_ct_convert.sh）")
    native = sitk.ReadImage(str(native_p))
    mask = None
    mp = resolve_path(cfg, case.get("tumor_mask_path"))
    if mp is not None and mp.exists():
        mask = sitk.Cast(sitk.ReadImage(str(mp)) > 0, sitk.sitkUInt8)
    stomach = None
    sp = wp.ct_stomach(sid)
    if sp.exists():
        stomach = sitk.Cast(sitk.ReadImage(str(sp)) > 0, sitk.sitkUInt8)

    res: dict = {}
    warns: List[str] = []
    point_phys = None
    pt = [case.get(f"ct_point_{a}") for a in "xyz"]
    if all(v is not None for v in pt):
        res["point_in_image"] = "Y" if T.inside(native, pt) else "N"
        if res["point_in_image"] == "N":
            raise CaseError(f"中心点 ({pt[0]}, {pt[1]}, {pt[2]}) 不在图像范围内 {native.GetSize()}；"
                            "请确认坐标是在本流程生成的 native 图像上标注的（体素坐标从 0 开始）")
        point_phys = T.index_to_physical(native, pt)
    mask_c = T.mask_centroid_physical(mask) if mask is not None else None
    if mask is not None:
        res["mask_ml"] = round(T.mask_volume_ml(mask), 2)
        if mask_c is None:
            warns.append("肿瘤分割为空")
    if point_phys is not None and mask is not None and mask_c is not None:
        d = T.distance_to_mask_mm(mask, point_phys)
        res["point_to_tumor_mm"] = None if d is None else round(d, 1)
        if d is not None and d > 5:
            warns.append(f"中心点不在肿瘤分割内（距 {d:.0f} mm）")
        res["point_to_mask_centroid_mm"] = round(float(np.linalg.norm(point_phys - mask_c)), 1)
    if point_phys is not None and stomach is not None:
        d = T.distance_to_mask_mm(stomach, point_phys)
        res["point_to_stomach_mm"] = None if d is None else round(d, 1)
        thr = float(cfg.ct.crop.get("point_stomach_max_dist_mm", 20))
        if d is not None and d > thr:
            warns.append(f"中心点距胃分割 {d:.0f} mm（>{thr:g}），请复核标点")
    if stomach is not None:
        res["stomach_ml"] = round(T.mask_volume_ml(stomach), 1)

    avg = bool(cfg.ct.get("average_thin_slices", True))
    for s in specs:
        mode = s.get("center", "point_or_mask")
        if mode == "point":
            center, src = point_phys, "point"
        elif mode == "mask_centroid":
            center, src = mask_c, "mask"
        else:
            center, src = (point_phys, "point") if point_phys is not None else (mask_c, "mask")
        name = s["name"]
        if center is None:
            res[f"{name}_center"] = "无"
            warns.append(f"{name}：没有中心点也没有分割，未裁剪")
            continue
        ref = T.roi_reference(center, s["spacing"], s["size_vox"])
        hu = T.resample_image(native, ref, default=float(cfg.ct.hu_clip[0]), average_thin=avg)
        arr = sitk.GetArrayFromImage(hu)
        win = s["window"]
        norm = T.window_normalize(arr, float(win["level"]), float(win["width"]), s.get("normalize", "minmax"))
        out = sitk.GetImageFromArray(norm.astype(np.float32))
        out.CopyInformation(ref)
        p_img = wp.ct_crop(name, sid, "img")
        p_img.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(p_img), True)
        pad = T.outside_fraction(native, ref)
        res[f"{name}_center"] = src
        res[f"{name}_pad_frac"] = round(pad, 3)
        if pad > 0.2:
            warns.append(f"{name}：{pad * 100:.0f}% 的区域超出图像范围（已用空气填充）")
        m_arr = None
        if mask is not None:
            mr = T.resample_mask(mask, ref, average_thin=avg)
            sitk.WriteImage(mr, str(wp.ct_crop(name, sid, "mask")), True)
            m_arr = sitk.GetArrayFromImage(mr)
            res[f"{name}_mask_in_crop"] = round(float(m_arr.sum() * np.prod(ref.GetSpacing()) / 1000.0
                                                      / max(res.get("mask_ml", 0) or 1e-9, 1e-9)), 3)
        st_arr = sitk.GetArrayFromImage(T.resample_label(stomach, ref)) if stomach is not None else None
        disp = T.window_normalize(arr, float(win["level"]), float(win["width"]), "minmax")
        center_vox = [int(v) // 2 for v in s["size_vox"]]
        # 图中文字用英文，避免服务器缺少中文字体
        title = f"{sid}  {name}  center={src}  pad={pad * 100:.0f}%"
        if "point_to_stomach_mm" in res and res["point_to_stomach_mm"] is not None:
            title += f"  to_stomach={res['point_to_stomach_mm']:g}mm"
        QC.case_qc(disp, ref.GetSpacing(), center_vox, wp.ct_qc(name, sid), title, m_arr, st_arr,
                   point_xyz=center_vox)
    if warns:
        res["status"] = "warning"
        res["message"] = "；".join(warns)
    return res


def make_overviews(cfg: Config, specs: List[dict], cases: List[dict], results: List[dict], run: Run) -> List[Path]:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    by_id = {r["study_id"]: r for r in results}
    out = []
    for s in specs:
        items = []
        for c in cases:
            sid = c["study_id"]
            p = wp.ct_crop(s["name"], sid, "img")
            if not p.exists():
                continue
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            win = s["window"]
            if s.get("normalize", "minmax") != "minmax":
                img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6)
            z = img.shape[0] // 2
            mp = wp.ct_crop(s["name"], sid, "mask")
            m = sitk.GetArrayFromImage(sitk.ReadImage(str(mp)))[z] if mp.exists() else None
            note = "!" if by_id.get(sid, {}).get("status") in ("warning", "failed") else ""
            items.append((sid, img[z], m, note))
        if items:
            sp = s["spacing"]
            out += QC.overview(items, run.dir / f"overview_{s['name']}", int(cfg.ct.qc.overview_per_page),
                               aspect=float(sp[1]) / float(sp[0]))
    return out


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_crop", args.run_name)
    log = run.logger
    specs = _specs(cfg, parse_id_list(args.specs))
    res = load_manifest(cfg, log)
    miss = missing_path_set(res)
    wp = WorkPaths(cfg)
    cases = []
    for c in to_cases(res.df):
        if not c.get("ct_dir") or (c["study_id"], "ct_dir") in miss:
            continue
        if (c["study_id"], "tumor_mask_path") in miss:
            c["tumor_mask_path"] = None
        c["_specs"] = [dict(s) for s in specs]
        c["_overwrite"] = args.overwrite
        cases.append(c)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    log.info("裁剪规格：%s", "；".join(f"{s['name']}（{list(s['size_vox'])} × {list(s['spacing'])} mm）" for s in specs))
    results = process_cases(cases, crop_case, cfg, log, resolve_workers(args, cfg), "CT 裁剪")
    run.write_table(results, "status.csv")
    pages = make_overviews(cfg, specs, cases, results, run)
    log.info("总览图 %d 页（运行目录 overview_*.png）；单例质控图在 %s", len(pages), wp.ct / "qc")
    log.info("提示：质控图是患者影像，只在服务器上查看，不要拷出。")
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("以肿瘤中心点为中心裁剪三维 ROI，窗宽窗位标准化，生成质控图和总览图。")
    p.add_argument("--specs", default=None, help="只处理指定的裁剪规格（逗号分隔，默认全部）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
