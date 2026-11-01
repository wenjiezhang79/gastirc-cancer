"""病理特征提取：读取切块坐标 → 编码器前向 → 特征存为 HDF5。

一个编码器跑一遍；不同编码器可以在不同 Python 环境中分别运行（输出格式统一）。
输出：``<work_dir>/wsi/features/<编码器>/<study_id>.h5``（见 features_io.py）。已存在的病例自动跳过。

命令行：
    python -m gastric_mm.wsi.extract_features --config configs/server.yaml --gpu 0 [--extractor uni2_h] [--limit 5]

需要先运行切块（05_wsi_tiling.sh），且切块规格（wsi.tiling.target_mpp / tile_px）一致。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.device import resolve_device
from ..utils.runner import exit_code, process_cases
from .extractors import Extractor, build_extractor, output_name
from .features_io import read_tiles, write_features
from .reader import SlideReader
from .tiling import tile_spec_name


class TileDataset:
    """按坐标读取切块（每个 DataLoader 进程各自打开切片）。"""

    def __init__(self, slide_path: str, coords: np.ndarray, level: int, read_px: int, out_px: int, transform,
                 backend: str, default_mpp):
        self.slide_path, self.coords, self.level, self.read_px = slide_path, coords, level, read_px
        self.out_px, self.transform, self.backend, self.default_mpp = out_px, transform, backend, default_mpp
        self._reader = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        from PIL import Image

        if self._reader is None:
            self._reader = SlideReader(Path(self.slide_path), self.backend, self.default_mpp)
        x, y = self.coords[i]
        a = self._reader.read_region(int(x), int(y), self.level, self.read_px, self.read_px)
        img = Image.fromarray(a)
        if img.size != (self.out_px, self.out_px):
            img = img.resize((self.out_px, self.out_px), Image.BILINEAR)
        return self.transform(img)


def extract_slide(ext: Extractor, case: dict, cfg: Config, device, out_dir_name: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    fc = cfg.features
    out = wp.wsi_features(out_dir_name, sid)
    if out.exists() and not case.get("_overwrite"):
        return {"status": "skipped", "message": "已有特征"}
    spec = tile_spec_name(cfg.wsi.tiling.target_mpp, cfg.wsi.tiling.tile_px)
    tp = wp.wsi_tiles(spec, sid)
    if not tp.exists():
        raise CaseError(f"没有切块坐标（规格 {spec}），请先运行 05_wsi_tiling.sh")
    coords, tattrs = read_tiles(tp)
    if len(coords) == 0:
        raise CaseError("该切片没有切块（组织检测未找到组织）")
    ds = TileDataset(str(resolve_path(cfg, case["slide_path"])), coords, int(tattrs["read_level"]),
                     int(tattrs["read_size_px"]), int(tattrs["tile_px"]), ext.transform,
                     cfg.wsi.get("backend", "auto"), cfg.wsi.get("default_base_mpp"))
    nw = int(fc.get("num_workers", 0))
    dl = DataLoader(ds, batch_size=int(fc.batch_size), shuffle=False, num_workers=nw,
                    pin_memory=device.type == "cuda", persistent_workers=False)
    use_bf16 = device.type == "cuda" and str(fc.get("precision", "bf16")) == "bf16"
    feats = []
    t0 = time.time()
    with torch.inference_mode():
        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = ext.forward(ext.model, batch)
            else:
                f = ext.forward(ext.model, batch)
            feats.append(f.float().cpu().numpy())
    feats = np.concatenate(feats, 0)
    if not np.isfinite(feats).all():
        raise CaseError("特征中出现 NaN/Inf（可能是精度问题，可尝试 --set features.precision=fp32）")
    attrs = {"encoder": out_dir_name, "source": "gastric_mm", "tile_spec": spec,
             "target_mpp": float(tattrs["target_mpp"]), "tile_px": int(tattrs["tile_px"]),
             "input_px": int(ext.input_px), "base_mpp": float(tattrs["base_mpp"]),
             "tile_size_level0": float(tattrs["tile_size_level0"]), "weights_sha256": ext.weights_sha256,
             "precision": "bf16" if use_bf16 else "fp32"}
    write_features(out, feats, coords, attrs, str(fc.get("dtype", "float32")))
    dt = time.time() - t0
    return {"n_tiles": int(len(coords)), "feature_dim": int(feats.shape[1]),
            "message": f"{len(coords)} 个切块，{feats.shape[1]} 维，{len(coords) / max(dt, 1e-6):.0f} 块/秒"}


def _run(args) -> int:
    import torch

    cfg = load_config(args.config, args.set)
    if args.extractor:
        cfg.features["extractor"] = args.extractor
    run = Run(cfg, "wsi_features", args.run_name or cfg.features.extractor)
    log = run.logger
    torch.manual_seed(int(cfg.get("seed", 0)))
    device = resolve_device(args.gpu)
    log.info("编码器：%s；设备：%s", cfg.features.extractor, device)
    ext = build_extractor(cfg.features.extractor, cfg)
    ext.model.to(device)
    name = output_name(ext)
    log.info("特征维数 %d，输入 %d px%s", ext.feature_dim, ext.input_px,
             f"，权重 SHA256 前 16 位 {ext.weights_sha256[:16]}" if ext.weights_sha256 else "")
    res = load_manifest(cfg, log)
    miss = missing_path_set(res)
    cases = [c for c in to_cases(res.df) if c.get("slide_path") and (c["study_id"], "slide_path") not in miss]
    for c in cases:
        c["_overwrite"] = args.overwrite
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    results = process_cases(cases, lambda c, k: extract_slide(ext, c, k, device, name), cfg, log, 1, "特征提取")
    run.write_table(results, "status.csv")
    log.info("特征目录：%s", WorkPaths(cfg).wsi / "features" / name)
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("用病理基础模型提取切块特征，保存为 HDF5。", workers=False, gpu=True)
    p.add_argument("--extractor", default=None, help="编码器名（默认取配置 features.extractor）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
