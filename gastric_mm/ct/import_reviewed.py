"""② 导入医生审核后的肿瘤分割（mask_source=auto_reviewed）。

医生在 ITK-SNAP / 3D Slicer 中打开 native 图像和候选分割（tumor_post 的输出），修改后另存到审核目录，
文件名以 study_id 开头（``<study_id>_tumor.nii.gz`` 或 ``<study_id>.nii.gz``）。本程序：
- 检查分割与 native 图像是否同一网格（不同则按物理坐标重采样并提示）、是否为空；
- 与自动候选比较（Dice、体积变化），统计医生修改了多少（报告可贴回）；
- 统一保存到 ``<work_dir>/ct/tumor_reviewed/<study_id>_tumor.nii.gz``；
- 生成清单副本 ``private/manifest_with_reviewed.csv``（已填 tumor_mask_path 与 mask_source=auto_reviewed；
  **已有人工勾画 manual 的病例不覆盖**）。**不会修改原清单**，核对后请手动替换。

命令行：
    python -m gastric_mm.ct.import_reviewed --config configs/server.yaml --src-dir <审核目录>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config, load_config, resolve_path, root_dir
from ..manifest import load_manifest, read_manifest_csv, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from . import transforms as T
from .nnunet_data import _same_grid
from .transfer import find_mask


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def import_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    src = find_mask(Path(case["_src_dir"]), sid)
    if src is None:
        return {"status": "skipped", "message": "审核目录中没有该例"}
    if case.get("mask_source") == "manual" and case.get("tumor_mask_path"):
        return {"status": "skipped", "message": "该例已有人工勾画（manual），不覆盖"}
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI")
    native = sitk.ReadImage(str(native_p))
    m = sitk.Cast(sitk.ReadImage(str(src)) > 0, sitk.sitkUInt8)
    note = []
    if not _same_grid(m, native):
        m = T.resample_label(m, native)
        note.append("与 native 网格不同，已按物理坐标对齐（请确认是在 native 图像上审核的）")
    m.CopyInformation(native)
    ml = T.mask_volume_ml(m)
    if ml == 0:
        raise CaseError("审核后的分割为空（若该例确实无法分割，请在清单中把可见度标为 invisible）")
    res = {"reviewed_ml": round(ml, 2)}
    ap = wp.ct_tumor_auto(sid)
    if ap.exists():
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(ap)))
        b = sitk.GetArrayFromImage(m)
        if a.shape == b.shape:
            res["dice_vs_auto"] = round(dice(a, b), 3)
            res["auto_ml"] = round(float((a > 0).sum()) * float(np.prod(native.GetSpacing())) / 1000.0, 2)
            res["edited"] = "N" if res["dice_vs_auto"] >= 0.98 else "Y"
    else:
        res["edited"] = "无候选"
    out = wp.ct_tumor_reviewed(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(m, str(out), True)
    if note:
        res["status"] = "warning"
        res["message"] = "；".join(note)
    return res


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_import_reviewed", args.run_name)
    log = run.logger
    src = Path(args.src_dir)
    if not src.is_dir():
        raise UserError("--src-dir 指定的审核目录不存在", "请填写医生保存审核后分割的目录。")
    res = load_manifest(cfg, log, check_paths=False)
    cases = select_cases(to_cases(res.df), parse_id_list(args.study_ids), args.limit)
    for c in cases:
        c["_src_dir"] = str(src)
    results = process_cases(cases, import_case, cfg, log, 1, "导入审核分割")
    cols = ["study_id", "status", "reviewed_ml", "auto_ml", "dice_vs_auto", "edited", "message"]
    run.write_table([r for r in results if "审核目录中没有" not in r.get("message", "")], "status.csv", cols)
    done = {r["study_id"] for r in results if r.get("status") in ("ok", "warning")}
    wp = WorkPaths(cfg)
    raw = read_manifest_csv(resolve_path(cfg, cfg.paths.manifest))
    root = root_dir(cfg)
    for i, sid in raw["study_id"].items():
        if sid in done:
            p = wp.ct_tumor_reviewed(sid).resolve()
            try:
                p = p.relative_to(root)
            except ValueError:
                pass
            raw.at[i, "tumor_mask_path"] = str(p)
            raw.at[i, "mask_source"] = "auto_reviewed"
    raw.to_csv(run.private_dir / "manifest_with_reviewed.csv", index=False, encoding="utf-8-sig")
    d = [r["dice_vs_auto"] for r in results if r.get("dice_vs_auto") is not None]
    L = ["# 审核分割导入报告", "", f"- 导入 {len(done)} 例",
         f"- 与自动候选的 Dice：中位数 {np.median(d):.3f}，基本未修改（≥0.98）{sum(1 for x in d if x >= 0.98)} 例，"
         f"明显修改（<0.8）{sum(1 for x in d if x < 0.8)} 例" if d else "- 没有可比较的自动候选", "",
         "核对 status.csv 后，用 private/manifest_with_reviewed.csv 替换原清单（该文件含原始路径，不要外传）。", ""]
    run.write_text("\n".join(L), "import_report.md")
    log.info("导入 %d 例。核对后用 private/manifest_with_reviewed.csv 替换原清单。", len(done))
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("导入医生审核后的肿瘤分割（mask_source=auto_reviewed），并统计医生修改的幅度。", workers=False)
    p.add_argument("--src-dir", required=True, help="审核后分割所在目录（文件名以 study_id 开头）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
