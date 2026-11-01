"""④ CT 特征提取：按编码器所需的裁剪规格直接从 native 图像生成 ROI → 冻结编码器前向 → HDF5。

- ROI 中心：医生中心点（主方案）；规格 center=mask_centroid 时用肿瘤分割质心（敏感性分析）；
- 2.5D：每个 ROI 取中心层 ±k 层（v2.features.slices_2d），每层 3 通道（相邻 3 层或同层复制），缩放到
  input_px，逐层提特征 → [n_层, D]；3D：整块 → [1, D]；
- 一个编码器跑一遍；不同编码器可以在不同 Python 环境中分别运行（输出格式统一，见 features_io.py）。

命令行：
    python -m gastric_mm.ct.extract_features --config configs/server.yaml --encoder radimagenet_r50 --gpu 0

输出：``<work_dir>/ct/features/<编码器>/<study_id>.h5``；运行目录 status.csv、encoder.json。已存在的病例自动跳过。
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

from ..config import Config, load_config
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.device import resolve_device
from ..utils.errors import CaseError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from .crop import case_centers, find_spec, roi_array, spec_center
from .encoders import CTEncoder, build_ct_encoder, check_specs
from .features_io import write_ct_features


def slices_2d(arr: np.ndarray, k, channels: str):
    """arr (z, y, x) ∈ [0,1] → (各层 3 通道图像 [n,3,y,x], 各层相对中心层的序号)。"""
    nz = arr.shape[0]
    zc = nz // 2
    zs = list(range(nz)) if k is None else list(range(max(0, zc - int(k)), min(nz, zc + int(k) + 1)))
    imgs = []
    for z in zs:
        if channels == "repeat":
            idx = [z, z, z]
        elif channels == "adjacent":
            idx = [max(0, z - 1), z, min(nz - 1, z + 1)]
        else:
            raise CaseError(f"未知的 2.5D 通道方式 {channels}（可选 adjacent / repeat）")
        imgs.append(arr[idx])
    return np.stack(imgs, 0).astype(np.float32), np.array(zs) - zc


def encode_roi(enc: CTEncoder, arr: np.ndarray, spec: dict, role: str, cfg: Config, device, use_bf16: bool,
               dump=None):
    import torch
    import torch.nn.functional as F

    fc = cfg.v2.features
    with torch.inference_mode():
        if enc.kind == "2d":
            k = (fc.get("slices_2d", {}) or {}).get(role)
            imgs, rel = slices_2d(arr, k, str(fc.get("channels_2d", "adjacent")))
            px = int(fc.get("input_px", 224))
            mean = torch.tensor(enc.mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(enc.std, dtype=torch.float32).view(1, 3, 1, 1)
            out = []
            bs = int(fc.get("batch_size", 32))
            for i in range(0, len(imgs), bs):
                x = torch.from_numpy(imgs[i:i + bs])
                if x.shape[-1] != px or x.shape[-2] != px:
                    x = F.interpolate(x, size=(px, px), mode="bilinear", align_corners=False)
                x = (x - mean) / std
                if dump is not None:
                    dump.append(x.numpy())
                x = x.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    f = enc.forward(enc.model, x)
                out.append(f.float().cpu().numpy())
            feats = np.concatenate(out, 0)
            z_mm = rel * float(spec["spacing"][2])
        else:
            x = torch.from_numpy(np.ascontiguousarray(enc.prepare3d(arr)))[None, None]
            if dump is not None:
                dump.append(x.numpy())
            x = x.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                f = enc.forward(enc.model, x)
            feats = f.float().cpu().numpy().reshape(1, -1)
            z_mm = np.zeros(1)
    if not np.isfinite(feats).all():
        raise CaseError(f"ROI {role} 的特征中出现 NaN/Inf（可尝试 --set v2.features.precision=fp32）")
    return feats, z_mm


def extract_case(enc: CTEncoder, specs: Dict[str, dict], case: dict, cfg: Config, device) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    out = wp.ct_features(enc.name, sid)
    if out.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "已有特征"}
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换")
    native = sitk.ReadImage(str(native_p))
    point_phys, mask_c = case_centers(case, cfg, native)
    use_bf16 = device.type == "cuda" and str(cfg.v2.features.get("precision", "bf16")) == "bf16"
    rois, zs, attrs, res, warns = {}, {}, {}, {}, []
    t0 = time.time()
    for role, spec in specs.items():
        center, src = spec_center(spec, point_phys, mask_c)
        if center is None:
            warns.append(f"{role}：没有中心点（规格要求 {spec.get('center', 'point_or_mask')}），未提取")
            continue
        arr, pad = roi_array(native, center, spec, cfg)
        dump = [] if case.get("_dump_dir") else None
        feats, z_mm = encode_roi(enc, arr, spec, role, cfg, device, use_bf16, dump)
        if dump is not None:
            from pathlib import Path
            d = Path(case["_dump_dir"])
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{sid}_{role}.npz", model_input=np.concatenate(dump, 0), features=feats)
        rois[role], zs[role] = feats, z_mm
        attrs[f"{role}_spec"], attrs[f"{role}_center"], attrs[f"{role}_pad_frac"] = spec["name"], src, round(pad, 3)
        res[f"{role}_n"] = int(len(feats))
        res[f"{role}_pad"] = round(pad, 3)
        if pad > 0.2:
            warns.append(f"{role}：{pad * 100:.0f}% 的区域超出图像范围")
    if not rois:
        raise CaseError("所有 ROI 都无法确定中心（没有中心点也没有肿瘤分割）")
    attrs.update(encoder=enc.name, kind=enc.kind, source="gastric_mm", weights_sha256=enc.weights_sha256,
                 precision="bf16" if use_bf16 else "fp32")
    write_ct_features(out, rois, attrs, zs, str(cfg.v2.features.get("dtype", "float32")))
    res["feature_dim"] = enc.feature_dim
    res["seconds"] = round(time.time() - t0, 2)
    if warns:
        res["status"] = "warning"
        res["message"] = "；".join(warns)
    return res


def _run(args) -> int:
    import torch

    cfg = load_config(args.config, args.set)
    name = args.encoder or cfg.v2.features.encoder
    run = Run(cfg, "ct_features", args.run_name or name)
    log = run.logger
    torch.manual_seed(int(cfg.get("seed", 0)))
    device = resolve_device(args.gpu)
    enc = build_ct_encoder(name, cfg)
    enc.model.to(device)
    specs = {role: find_spec(cfg, s) for role, s in enc.rois.items()}
    check_specs(enc, specs)
    log.info("CT 编码器 %s（%s，%d 维）；设备 %s；ROI：%s", enc.name, enc.kind, enc.feature_dim, device,
             "；".join(f"{r}={s['name']}（{list(s['size_vox'])} × {list(s['spacing'])} mm）" for r, s in specs.items()))
    if enc.weights_sha256:
        log.info("权重 SHA256 前 16 位：%s", enc.weights_sha256[:16])
    for k, v in enc.notes.items():
        log.warning("【需在服务器核对】%s", v)
    run.write_json({"encoder": enc.name, "kind": enc.kind, "feature_dim": enc.feature_dim,
                    "rois": {r: s for r, s in specs.items()}, "weights_sha256": enc.weights_sha256,
                    "slices_2d": cfg.v2.features.get("slices_2d"), "channels_2d": cfg.v2.features.get("channels_2d")},
                   "encoder.json")
    res = load_manifest(cfg, log, check_paths=False)
    wp = WorkPaths(cfg)
    miss = missing_path_set(res)
    cases: List[dict] = []
    for c in to_cases(res.df):
        if not c.get("ct_dir") or not wp.ct_native(c["study_id"]).exists():
            continue
        if (c["study_id"], "tumor_mask_path") in miss:
            c["tumor_mask_path"] = None
        c["_overwrite"] = args.overwrite
        cases.append(c)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    if args.dump_inputs:
        dd = run.private_dir / "encoder_inputs"
        for c in cases[: args.dump_inputs]:
            c["_dump_dir"] = str(dd)
            c["_overwrite"] = True
        log.info("前 %d 例的编码器输入与特征保存到 private/encoder_inputs/（用于与官方示例代码核对；是患者影像，不要拷出）",
                 min(args.dump_inputs, len(cases)))
    results = process_cases(cases, lambda c, k: extract_case(enc, specs, c, k, device), cfg, log, 1, "CT 特征提取")
    run.write_table(results, "status.csv")
    log.info("特征目录：%s", wp.ct / "features" / enc.name)
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("用冻结的 CT 编码器（2.5D / 3D）提取肿瘤 ROI 与区域 ROI 的特征，保存为 HDF5。",
                    workers=False, gpu=True)
    p.add_argument("--encoder", default=None, help="CT 编码器名（默认取配置 v2.features.encoder）")
    p.add_argument("--dump-inputs", type=int, default=0, metavar="N",
                   help="把前 N 例送入编码器的输入数组与输出特征另存（与官方示例代码核对预处理用），这些例会重新提取")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
