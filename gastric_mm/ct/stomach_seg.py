"""① 胃分割：调用 TotalSegmentator（独立 Python 环境）分割全部病例的胃。

流程（逐例）：native NIfTI → TotalSegmentator（``--roi_subset stomach``）→ 结果对齐回 native 网格 →
``<work_dir>/ct/stomach/<study_id>.nii.gz``。同时计算质控指标：胃体积（充盈程度）、连通域个数、
中心点到胃的距离（超过 ct.crop.point_stomach_max_dist_mm 提示复核标点），并生成质控图。

离线要求：TotalSegmentator 的权重须预先拷入 v2.external.totalseg.home_dir（作为环境变量
TOTALSEG_HOME_DIR 传给子进程）；该目录下 config.json 中的 send_usage_stats 请设为 false
（见 docs/v2_offline_setup.md）。

命令行：
    python -m gastric_mm.ct.stomach_seg --config configs/server.yaml --gpu 0 [--limit 5]

输出：``<work_dir>/ct/stomach/``、``<work_dir>/ct/qc/stomach/``；运行目录 status.csv、overview_stomach_pNN.png。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from . import qc as QC
from . import transforms as T
from .external import describe, resolve_command, run_tool, tool_env

QC_WINDOW = (40.0, 400.0)


def check_totalseg_home(cfg: Config, logger) -> Optional[Path]:
    tc = cfg.v2.external.totalseg
    home = resolve_path(cfg, tc.get("home_dir"))
    if home is None:
        logger.warning("未配置 v2.external.totalseg.home_dir：TotalSegmentator 将在默认位置（~/.totalsegmentator）找权重；"
                       "找不到时它会尝试联网下载并失败。建议显式配置权重目录。")
        return None
    if not home.is_dir():
        raise UserError("v2.external.totalseg.home_dir 指定的权重目录不存在",
                        "请把在联网电脑上准备好的 TotalSegmentator 权重目录拷到服务器，并填写其绝对路径。")
    res = home / "nnunet" / "results"
    if not res.is_dir() or not any(res.iterdir()):
        logger.warning("TotalSegmentator 权重目录下没有 nnunet/results/（权重可能未拷全），运行时可能尝试联网下载。")
    cfgf = home / "config.json"
    if cfgf.exists():
        try:
            if json.loads(cfgf.read_text(encoding="utf-8")).get("send_usage_stats", False):
                logger.warning("TotalSegmentator 的 config.json 中 send_usage_stats 为 true，会尝试联网发送使用统计；"
                               "请改为 false（见 docs/v2_offline_setup.md）。")
        except Exception:
            pass
    return home


def stomach_metrics(stomach, native, point_idx, cfg: Config) -> dict:
    import SimpleITK as sitk

    res: dict = {}
    warns: List[str] = []
    ml = T.mask_volume_ml(stomach)
    res["stomach_ml"] = round(ml, 1)
    cc = sitk.ConnectedComponent(stomach, True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    sizes = sorted((stats.GetPhysicalSize(lb) / 1000.0 for lb in stats.GetLabels()), reverse=True)
    res["n_components"] = len(sizes)
    res["largest_frac"] = round(sizes[0] / max(sum(sizes), 1e-9), 3) if sizes else 0.0
    if ml == 0:
        raise CaseError("胃分割为空（TotalSegmentator 没有找到胃）；请检查该例图像（范围是否包含上腹部）")
    if ml < float(cfg.v2.stomach.get("min_ml", 30)):
        warns.append(f"胃体积只有 {ml:.0f} ml，可能分割不全")
    if sizes and res["largest_frac"] < 0.8:
        warns.append(f"胃分割分成 {len(sizes)} 块（最大块占 {res['largest_frac'] * 100:.0f}%）")
    if point_idx is not None:
        p = T.index_to_physical(native, point_idx)
        d = T.distance_to_mask_mm(stomach, p)
        res["point_to_stomach_mm"] = None if d is None else round(d, 1)
        thr = float(cfg.ct.crop.get("point_stomach_max_dist_mm", 20))
        if d is not None and d > thr:
            warns.append(f"中心点距胃分割 {d:.0f} mm（>{thr:g}），请复核标点或胃分割")
    if warns:
        res["status"] = "warning"
        res["message"] = "；".join(warns)
    return res


def qc_center(native, stomach, point_idx) -> List[int]:
    if point_idx is not None:
        return [int(round(v)) for v in point_idx]
    c = T.mask_centroid_physical(stomach)
    return [int(round(v)) for v in T.physical_to_index(native, c)]


def segment_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    out = wp.ct_stomach(sid)
    if out.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "已有胃分割"}
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换（03_ct_convert.sh）")
    tmp = wp.ct / "_tmp_totalseg" / sid
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        tc = cfg.v2.external.totalseg
        args = list(case["_cmd"]) + ["-i", str(native_p), "-o", str(tmp), "--roi_subset", "stomach",
                                     "-d", "gpu" if case.get("_gpu") is not None else "cpu"]
        if tc.get("fast"):
            args.append("--fast")
        args += [str(a) for a in (tc.get("extra_args") or [])]
        env = tool_env({"TOTALSEG_HOME_DIR": case.get("_home")}, case.get("_gpu"))
        run_tool(args, env, float(tc.get("timeout_s", 1800)), "TotalSegmentator", sensitive=[str(tmp.parent)])
        cand = [tmp / "stomach.nii.gz", tmp / "stomach.nii"]
        found = next((p for p in cand if p.exists()), None)
        if found is None:
            raise CaseError("TotalSegmentator 运行结束但没有输出 stomach.nii.gz（版本不同？请把 run.log 贴回）")
        native = sitk.ReadImage(str(native_p))
        seg = sitk.Cast(sitk.ReadImage(str(found)) > 0, sitk.sitkUInt8)
        same = (seg.GetSize() == native.GetSize()
                and np.allclose(seg.GetSpacing(), native.GetSpacing(), atol=1e-3)
                and np.allclose(seg.GetOrigin(), native.GetOrigin(), atol=1e-2)
                and np.allclose(seg.GetDirection(), native.GetDirection(), atol=1e-3))
        if not same:
            seg = T.resample_label(seg, native)
        seg.CopyInformation(native)
        pt = [case.get(f"ct_point_{a}") for a in "xyz"]
        point_idx = pt if all(v is not None for v in pt) else None
        res = stomach_metrics(seg, native, point_idx, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(seg, str(out), True)
        arr = T.window_normalize(sitk.GetArrayFromImage(native), *QC_WINDOW, "minmax")
        center = qc_center(native, seg, point_idx)
        title = f"{sid}  stomach={res['stomach_ml']:g}ml"
        if res.get("point_to_stomach_mm") is not None:
            title += f"  point_to_stomach={res['point_to_stomach_mm']:g}mm"
        QC.case_qc(arr, native.GetSpacing(), center, wp.ct_qc("stomach", sid), title,
                   stomach=sitk.GetArrayFromImage(seg), point_xyz=center if point_idx is not None else None)
        res["resampled"] = "N" if same else "Y"
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_overview(cfg: Config, cases: List[dict], results: List[dict], run: Run) -> List[Path]:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    by_id = {r["study_id"]: r for r in results}
    items = []
    for c in cases:
        sid = c["study_id"]
        sp, npth = wp.ct_stomach(sid), wp.ct_native(sid)
        if not (sp.exists() and npth.exists()) or by_id.get(sid, {}).get("status") == "failed":
            continue
        native = sitk.ReadImage(str(npth))
        seg = sitk.ReadImage(str(sp))
        pt = [c.get(f"ct_point_{a}") for a in "xyz"]
        center = qc_center(native, seg, pt if all(v is not None for v in pt) else None)
        z = int(np.clip(center[2], 0, native.GetSize()[2] - 1))
        img = T.window_normalize(sitk.GetArrayFromImage(native)[z], *QC_WINDOW, "minmax")
        m = sitk.GetArrayFromImage(seg)[z]
        note = "!" if by_id.get(sid, {}).get("status") == "warning" else ""
        items.append((sid, img, m, note))
    if not items:
        return []
    return QC.overview(items, run.dir / "overview_stomach", int(cfg.ct.qc.overview_per_page))


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "ct_stomach_seg", args.run_name)
    log = run.logger
    tc = cfg.v2.external.totalseg
    cmd = resolve_command(tc.get("command"), "TotalSegmentator", "v2.external.totalseg.command")
    home = check_totalseg_home(cfg, log)
    log.info("命令：%s；设备：%s", describe(cmd), "GPU %d" % args.gpu if args.gpu is not None else "CPU")
    res = load_manifest(cfg, log, check_paths=False)
    wp = WorkPaths(cfg)
    cases = [c for c in to_cases(res.df) if c.get("ct_dir") and wp.ct_native(c["study_id"]).exists()]
    n_no = sum(1 for c in to_cases(res.df) if c.get("ct_dir")) - len(cases)
    if n_no:
        log.warning("%d 例有 CT 但还没有 native NIfTI（未转换或转换失败），本次跳过", n_no)
    for c in cases:
        c.update(_cmd=cmd, _home=str(home) if home else None, _gpu=args.gpu, _overwrite=args.overwrite)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, segment_case, cfg, log, 1, "胃分割")
    run.write_table(results, "status.csv", ["study_id", "status", "stomach_ml", "n_components", "largest_frac",
                                            "point_to_stomach_mm", "resampled", "message"])
    pages = make_overview(cfg, cases, results, run)
    flagged = [r["study_id"] for r in results if r.get("status") in ("warning", "failed")]
    log.info("总览图 %d 页（运行目录 overview_stomach_*.png）；需复核 %d 例（见 status.csv）", len(pages), len(flagged))
    log.info("提示：质控图是患者影像，只在服务器上查看，不要拷出。")
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("调用 TotalSegmentator 分割胃（独立环境），对齐到 native 网格并生成质控图。",
                    workers=False, gpu=True)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
