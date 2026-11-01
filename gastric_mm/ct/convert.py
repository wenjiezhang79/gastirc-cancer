"""DICOM → NIfTI 转换与整卷重采样。

每例：
1. 扫描 ct_dir，按期相/层厚选出分析序列（规则与 DICOM 汇总相同；清单 ct_series_number 可强制指定）。
2. 读取为 HU，统一方向（ct.native_orientation，默认 LPS），保存为
   ``<work_dir>/ct/native/<study_id>.nii.gz``（int16）——**医生在这份图像上标点和勾画**。
   同目录下 ``<study_id>.json`` 记录序列信息（序列号、期相、层厚、坐标系编号等，不含身份信息）。
3. 若 ct.save_thin_series=true，另存同一次采集中最薄的软组织重建到 ``native_thin/``（供医生在薄层上标注）。
4. 若 ct.save_resampled=true，按 ct.target_spacing 整卷重采样（薄层先多层平均）保存到 ``resampled/``；
   清单中有肿瘤分割时同时重采样到 ``resampled_mask/``。

命令行：
    python -m gastric_mm.ct.convert --config configs/server.yaml [--limit 5] [--overwrite]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases, resolve_workers
from . import transforms as T
from .dicom_io import SeriesInfo, read_series_image, scan_dicom_dir, select_series, select_thin_series


def series_meta(s: SeriesInfo, img, reason: str = "") -> dict:
    return {
        "series_number": s.series_number, "phase": s.phase, "phase_basis": s.phase_basis,
        "slice_thickness_mm": s.slice_thickness, "slice_spacing_mm": s.spacing_median, "n_images": s.n_images,
        "manufacturer": s.manufacturer, "kernel": s.kernel, "frame_of_ref_hash": s.for_hash,
        "acq_time_s": s.acq_time_s, "select_reason": reason,
        "size_xyz": list(img.GetSize()), "spacing_xyz": [round(v, 4) for v in img.GetSpacing()],
        "origin_xyz": [round(v, 3) for v in img.GetOrigin()], "direction": [round(v, 4) for v in img.GetDirection()],
    }


def _write(img, path: Path) -> None:
    import SimpleITK as sitk
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    sitk.WriteImage(img, str(tmp), True)
    tmp.replace(path)


def convert_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    ct = cfg.ct
    wp = WorkPaths(cfg)
    sid = case["study_id"]
    overwrite = bool(case.get("_overwrite"))
    native_p = wp.ct_native(sid)
    want_resampled = bool(ct.get("save_resampled", True))
    want_thin = bool(ct.get("save_thin_series", False))
    mask_src = resolve_path(cfg, case.get("tumor_mask_path"))
    done = native_p.exists() and (not want_resampled or wp.ct_resampled(sid).exists()) and \
        (not want_resampled or mask_src is None or wp.ct_resampled_mask(sid).exists())
    if done and not overwrite:
        return {"status": "skipped", "message": "已有转换结果"}

    d = resolve_path(cfg, case["ct_dir"])
    series = scan_dicom_dir(d, ct.phase_keywords, ct.phase_delay_s, float(ct.slice_spacing_tolerance))
    if not series:
        raise CaseError("文件夹中没有可读取的 DICOM 图像")
    chosen, reason = select_series(series, ct.phase, float(ct.preferred_thickness_mm), int(ct.min_slices),
                                   case.get("ct_series_number"))
    allow_nu = bool(ct.get("allow_nonuniform", False))
    img = read_series_image(chosen, allow_nu)
    img = T.orient(img, str(ct.native_orientation))
    native = T.to_int16_hu(img, tuple(ct.hu_clip))
    _write(native, native_p)
    meta = series_meta(chosen, native, reason)
    wp.ct_native_meta(sid).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    res = {"series_number": chosen.series_number, "phase": chosen.phase,
           "thickness_mm": chosen.slice_thickness, "native_size": "x".join(str(v) for v in native.GetSize()),
           "native_spacing": "x".join(f"{v:.3g}" for v in native.GetSpacing())}
    msgs = [f"序列 {chosen.series_number}（{chosen.phase}，{chosen.slice_thickness:g} mm）"
            if chosen.slice_thickness else f"序列 {chosen.series_number}"]
    if "人工核对" in reason:
        msgs.append("期相未识别，需人工核对")
        res["status"] = "warning"

    if want_thin:
        thin = select_thin_series(series, chosen, int(ct.min_slices))
        if thin is not None:
            timg = T.to_int16_hu(T.orient(read_series_image(thin, allow_nu), str(ct.native_orientation)),
                                 tuple(ct.hu_clip))
            _write(timg, wp.ct_native_thin(sid))
            wp.ct_native_thin_meta(sid).write_text(
                json.dumps(series_meta(thin, timg, "同一次采集的薄层重建"), ensure_ascii=False, indent=2),
                encoding="utf-8")
            res["thin_series"] = thin.series_number
            msgs.append(f"薄层序列 {thin.series_number}（{thin.slice_thickness:g} mm）已另存")

    if want_resampled:
        ref = T.spacing_reference(native, ct.target_spacing)
        rs = T.resample_image(native, ref, default=float(ct.hu_clip[0]),
                              average_thin=bool(ct.get("average_thin_slices", True)))
        rs = T.to_int16_hu(rs, tuple(ct.hu_clip))
        _write(rs, wp.ct_resampled(sid))
        res["resampled_size"] = "x".join(str(v) for v in rs.GetSize())
        if mask_src is not None and mask_src.exists():
            mask = sitk.ReadImage(str(mask_src))
            if T.mask_volume_ml(mask) == 0:
                msgs.append("肿瘤分割为空")
                res["status"] = "warning"
            m_native = T.resample_label(sitk.Cast(mask > 0, sitk.sitkUInt8), native)
            if T.mask_volume_ml(m_native) == 0 and T.mask_volume_ml(mask) > 0:
                raise CaseError("肿瘤分割与 CT 图像在空间上不重叠：分割可能不是在本流程生成的 native 图像上勾画的")
            rm = T.resample_mask(mask, ref, average_thin=bool(ct.get("average_thin_slices", True)))
            _write(rm, wp.ct_resampled_mask(sid))
            res["mask_ml"] = round(T.mask_volume_ml(rm), 2)
    res["message"] = "；".join(msgs)
    return res


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_convert", args.run_name)
    log = run.logger
    res = load_manifest(cfg, log)
    miss = missing_path_set(res)
    cases = [c for c in to_cases(res.df) if c.get("ct_dir") and (c["study_id"], "ct_dir") not in miss]
    for c in cases:
        if (c["study_id"], "tumor_mask_path") in miss:
            c["tumor_mask_path"] = None
        c["_overwrite"] = args.overwrite
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, convert_case, cfg, log, resolve_workers(args, cfg), "DICOM→NIfTI")
    run.write_table(results, "status.csv")
    log.info("结果目录：%s", WorkPaths(cfg).ct)
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("DICOM 转 NIfTI（原始分辨率、统一方向），并整卷重采样到目标体素。")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
