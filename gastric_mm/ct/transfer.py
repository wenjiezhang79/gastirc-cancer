"""标注转换：把在薄层（或同一次采集的其他重建）上勾画的肿瘤分割，转换到分析用的 native 图像网格上。

- 同一次扫描的不同重建（如 1.25 mm 与 5 mm）处在同一物理空间，按物理坐标重采样即可。
- 薄层 → 厚层：先沿 z 方向按目标层厚做体积平均，再取 50% 阈值（与厚层重建的部分容积一致）。
- 会检查两份图像的序列信息（native_thin/<id>.json 与 native/<id>.json）是否属于同一坐标系、同一次采集；
  不一致时该例失败（不同期相之间有呼吸和体位差异，不能直接搬）。

输入：--src-dir 下以 study_id 开头的分割文件（<study_id>.nii.gz 或 <study_id>_*.nii.gz）。
输出：--out-dir/<study_id>_tumor.nii.gz，以及运行目录 private/manifest_with_masks.csv（清单副本，已填 tumor_mask_path；含原始路径，不要外传）。

命令行：
    python -m gastric_mm.ct.transfer --config configs/server.yaml --src-dir <薄层分割目录> --out-dir <输出目录>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import Config, load_config, resolve_path, root_dir
from ..manifest import load_manifest, read_manifest_csv, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from . import transforms as T


def find_mask(src_dir: Path, sid: str) -> Optional[Path]:
    for p in sorted(src_dir.glob(f"{sid}*.nii*")):
        rest = p.name[len(sid):]
        if rest[:1] in (".", "_"):
            return p
    return None


def check_same_acquisition(wp: WorkPaths, sid: str, tol_s: float = 10.0) -> str:
    a, b = wp.ct_native_meta(sid), wp.ct_native_thin_meta(sid)
    if not (a.exists() and b.exists()):
        return "未找到两份序列信息，跳过同一次采集的核对"
    ma = json.loads(a.read_text(encoding="utf-8"))
    mb = json.loads(b.read_text(encoding="utf-8"))
    if ma.get("frame_of_ref_hash") != mb.get("frame_of_ref_hash"):
        raise CaseError("薄层与分析序列不在同一空间坐标系，不能直接转换")
    ta, tb = ma.get("acq_time_s"), mb.get("acq_time_s")
    if ta is not None and tb is not None and abs(ta - tb) > tol_s:
        raise CaseError(f"薄层与分析序列的采集时间相差 {abs(ta - tb):.0f} 秒（可能是不同期相），不能直接转换")
    return "同一次采集"


def transfer_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    src = find_mask(Path(case["_src_dir"]), sid)
    if src is None:
        return {"status": "skipped", "message": "没有对应的分割文件"}
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换")
    out = Path(case["_out_dir"]) / f"{sid}_tumor.nii.gz"
    if out.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "输出已存在", "out_file": out.name}
    check = "跳过核对" if case.get("_no_check") else check_same_acquisition(wp, sid)
    native = sitk.ReadImage(str(native_p))
    mask = sitk.ReadImage(str(src))
    v0 = T.mask_volume_ml(mask)
    if v0 == 0:
        raise CaseError("源分割为空")
    m = T.resample_mask(mask, native, average_thin=bool(cfg.ct.get("average_thin_slices", True)))
    v1 = T.mask_volume_ml(m)
    if v1 == 0:
        raise CaseError("转换后分割为空：源分割与 native 图像在空间上不重叠")
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(m, str(out), True)
    ratio = v1 / v0
    res = {"src_ml": round(v0, 2), "out_ml": round(v1, 2), "volume_ratio": round(ratio, 3), "check": check,
           "out_file": out.name, "message": f"{v0:.1f} ml → {v1:.1f} ml（{check}）"}
    if not 0.7 <= ratio <= 1.3:
        res["status"] = "warning"
        res["message"] += "；体积变化较大，请检查"
    return res


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_transfer", args.run_name)
    log = run.logger
    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    if not src_dir.is_dir():
        raise UserError("--src-dir 指定的目录不存在", "请填写存放薄层分割的目录。")
    res = load_manifest(cfg, log, check_paths=False)
    cases = select_cases(to_cases(res.df), parse_id_list(args.study_ids), args.limit)
    for c in cases:
        c.update(_src_dir=str(src_dir), _out_dir=str(out_dir), _overwrite=args.overwrite, _no_check=args.no_check)
    results = process_cases(cases, transfer_case, cfg, log, 1, "分割转换")
    run.write_table(results, "status.csv",
                    ["study_id", "status", "src_ml", "out_ml", "volume_ratio", "check", "message"])
    raw = read_manifest_csv(resolve_path(cfg, cfg.paths.manifest))
    root = root_dir(cfg)
    for i, sid in raw["study_id"].items():
        r = next((x for x in results if x["study_id"] == sid and x.get("out_file")), None)
        if r:
            p = (out_dir / r["out_file"]).resolve()
            try:
                p = p.relative_to(root)
            except ValueError:
                pass
            raw.at[i, "tumor_mask_path"] = str(p)
            if not raw.at[i, "mask_source"]:
                raw.at[i, "mask_source"] = "manual"
    raw.to_csv(run.private_dir / "manifest_with_masks.csv", index=False, encoding="utf-8-sig")  # 含原始路径
    log.info("核对 status.csv 后，可用 private/manifest_with_masks.csv 替换原清单。")
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("把薄层上勾画的分割转换到分析用 native 图像网格（同一次采集的不同重建之间）。", workers=False)
    p.add_argument("--src-dir", required=True, help="源分割目录（文件名以 study_id 开头）")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--no-check", action="store_true", help="不核对是否同一次采集（不推荐）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
