"""⑤ 影像组学特征提取（在 pyradiomics 独立环境中运行）。

逐例：native 图像 + 肿瘤分割（清单中 mask_source 属于 v2.radiomics.mask_sources 的病例）→ 统一重采样到
v2.radiomics.spacing（默认 0.8 × 0.8 × 5 mm，图像线性插值、薄层先多层平均；分割按体积占比取阈值）→
pyradiomics 按固定 bin width（默认 25 HU）提取一阶、形状、纹理特征（类别与图像类型见配置）。
5 mm 层厚下三维形状特征精度有限，只作参考。

环境：pyradiomics 只能从源码编译安装（需先装 numpy 与 C 编译器），因此装在独立环境中；该环境还需
pandas、scipy、PyYAML、SimpleITK（见 docs/v2_offline_setup.md）。本模块不依赖 torch。

命令行（用 pyradiomics 环境的 python 运行）：
    /opt/envs/radiomics/bin/python -m gastric_mm.ct.radiomics --config configs/server.yaml --num-workers 8

输出：``<work_dir>/ct/radiomics/<名称>/``：features.csv（每例一行，只含研究编号）、params.json、
cases/<study_id>.json（逐例结果，已存在的病例自动跳过）；运行目录 status.csv、radiomics_report.md。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases, resolve_workers
from . import transforms as T
from .nnunet_data import load_on_native


def radiomics_params(cfg: Config) -> dict:
    rc = cfg.v2.radiomics
    return {"spacing": [float(v) for v in rc.spacing], "bin_width": float(rc.bin_width),
            "image_types": {k: dict(v or {}) for k, v in dict(rc.image_types).items()},
            "feature_classes": list(rc.feature_classes), "min_mask_voxels": int(rc.get("min_mask_voxels", 20)),
            "average_thin_slices": bool(cfg.ct.get("average_thin_slices", True))}


def make_extractor(params: dict):
    try:
        import radiomics
        from radiomics import featureextractor
    except ImportError:
        raise UserError("当前 Python 环境中没有 pyradiomics",
                        "影像组学在独立环境中运行：请用该环境的 python 执行（如 RADIOMICS_PY=/opt/envs/radiomics/bin/python "
                        "bash scripts/25_radiomics.sh），安装方法见 docs/v2_offline_setup.md。") from None
    radiomics.setVerbosity(40)
    settings = {"binWidth": params["bin_width"], "label": 1, "force2D": False, "resampledPixelSpacing": None,
                "correctMask": True, "geometryTolerance": 1e-3, "additionalInfo": False,
                "minimumROISize": params["min_mask_voxels"]}
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllImageTypes()
    for name, kw in params["image_types"].items():
        ext.enableImageTypeByName(name, customArgs=kw or {})
    ext.disableAllFeatures()
    for c in params["feature_classes"]:
        ext.enableFeatureClassByName(c)
    return ext


def extract_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    out = Path(case["_out_dir"]) / "cases" / f"{sid}.json"
    if out.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "已有结果"}
    params = case["_params"]
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI")
    mp = resolve_path(cfg, case.get("tumor_mask_path"))
    if mp is None or not mp.exists():
        raise CaseError("tumor_mask_path 指向的文件不存在")
    native = sitk.ReadImage(str(native_p))
    mask = load_on_native(mp, native)
    ref = T.spacing_reference(native, params["spacing"])
    img = T.resample_image(native, ref, default=float(cfg.ct.hu_clip[0]), average_thin=params["average_thin_slices"])
    m = T.resample_mask(mask, ref, average_thin=params["average_thin_slices"])
    nvox = int(sitk.GetArrayViewFromImage(m).sum())
    if nvox < params["min_mask_voxels"]:
        raise CaseError(f"重采样后分割只有 {nvox} 个体素（< {params['min_mask_voxels']}），无法可靠计算影像组学")
    ext = make_extractor(params)
    try:
        r = ext.execute(img, m)
    except Exception as e:  # pyradiomics 的报错可能带临时信息，只保留类型与首行
        raise CaseError(f"pyradiomics 计算失败：{type(e).__name__}: {str(e).splitlines()[0][:200]}") from None
    feats = {k: float(np.asarray(v)) for k, v in r.items() if not k.startswith("diagnostics_")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"study_id": sid, "mask_source": case.get("mask_source"), "features": feats}),
                   encoding="utf-8")
    return {"n_features": len(feats), "mask_ml": round(T.mask_volume_ml(m), 2), "mask_source": case.get("mask_source")}


def build_table(out_dir: Path, sids: List[str]):
    import pandas as pd

    rows = []
    for sid in sids:
        p = out_dir / "cases" / f"{sid}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            row = {"study_id": d["study_id"], "mask_source": d.get("mask_source")}
            row.update(d["features"])
            rows.append(row)
    return pd.DataFrame(rows)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    rc = cfg.v2.radiomics
    run = Run(cfg, "ct_radiomics", args.run_name or rc.name)
    log = run.logger
    wp = WorkPaths(cfg)
    out_dir = wp.ct_radiomics(rc.name)
    params = radiomics_params(cfg)
    make_extractor(params)   # 先确认环境
    pp = out_dir / "params.json"
    if pp.exists() and json.loads(pp.read_text(encoding="utf-8")) != params and not args.overwrite:
        raise UserError(f"影像组学特征表 {rc.name} 已用不同的参数提取过",
                        "参数改变时请换一个名称（--set v2.radiomics.name=新名称），或加 --overwrite 全部重新提取。")
    out_dir.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(params, indent=2), encoding="utf-8")
    run.write_json(params, "params.json")
    res = load_manifest(cfg, log)
    miss = missing_path_set(res)
    srcs = set(rc.get("mask_sources", ["manual", "auto_reviewed"]) or [])
    all_cases = to_cases(res.df)
    cases = [c for c in all_cases if c.get("tumor_mask_path") and (c["study_id"], "tumor_mask_path") not in miss
             and c.get("mask_source") in srcs]
    log.info("有可用肿瘤分割（%s）的病例 %d 例（清单共 %d 例）", "/".join(sorted(srcs)), len(cases), len(all_cases))
    for c in cases:
        c.update(_out_dir=str(out_dir), _params=params, _overwrite=args.overwrite)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, extract_case, cfg, log, resolve_workers(args, cfg), "影像组学提取")
    run.write_table(results, "status.csv", ["study_id", "status", "mask_source", "mask_ml", "n_features", "message"])
    tab = build_table(out_dir, [c["study_id"] for c in all_cases])
    if len(tab):
        tab.to_csv(out_dir / "features.csv", index=False)
    by_src = tab["mask_source"].value_counts().to_dict() if len(tab) else {}
    L = ["# 影像组学提取报告", "", f"- 特征表：{rc.name}；病例 {len(tab)} 例（按分割来源：{by_src}）；"
         f"特征 {max(0, tab.shape[1] - 2)} 个", f"- 参数：{json.dumps(params, ensure_ascii=False)}", "",
         "下一步：python -m gastric_mm.ct_mtl.radiomics_model --exp radiomics_<名称> 建立影像组学基线模型（主环境）。", ""]
    run.write_text("\n".join(L), "radiomics_report.md")
    log.info("特征表：%s（%d 例）", out_dir / "features.csv", len(tab))
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("在 pyradiomics 独立环境中提取肿瘤影像组学特征（固定 bin width，统一重采样）。")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
