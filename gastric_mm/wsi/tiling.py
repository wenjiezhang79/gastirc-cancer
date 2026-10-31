"""病理切片：组织检测 + 20× 切块（只保存坐标）+ 质控图。

切块规格由 wsi.tiling.target_mpp（20× ≈ 0.5 µm/px）和 tile_px 决定，规格名如 ``0.50mpp_224px``。
输出：
- ``<work_dir>/wsi/tiles/<规格>/<study_id>.h5``：coords [N,2]（level-0 左上角）、tissue_mask（缩略图掩码）及读取参数
- ``<work_dir>/wsi/qc/<规格>/<study_id>.png``：缩略图 + 组织轮廓（绿）+ 保留的切块（蓝框）
- 运行目录 ``overview_pNN.png``：总览拼图；``status.csv``：每例组织面积、切块数等

命令行：
    python -m gastric_mm.wsi.tiling --config configs/server.yaml [--limit 5]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, missing_path_set, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases, resolve_workers
from .features_io import read_tiles, write_h5
from .reader import SlideReader
from .tissue import detect_tissue


def tile_spec_name(target_mpp: float, tile_px: int) -> str:
    return f"{float(target_mpp):.2f}mpp_{int(tile_px)}px"


def plan_tiles(mask: np.ndarray, mask_mpp: float, base_mpp: float, level0_wh: Tuple[int, int],
               target_mpp: float, tile_px: int, min_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """在组织掩码上按网格规划切块。返回 (level-0 坐标 [N,2], 组织占比 [N])。"""
    tile_l0 = tile_px * target_mpp / base_mpp
    w0, h0 = level0_wh
    nx, ny = int(w0 // tile_l0), int(h0 // tile_l0)
    if nx == 0 or ny == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    # 积分图计算每个切块在掩码中的组织占比
    integ = np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    scale = base_mpp / mask_mpp
    xs = np.arange(nx) * tile_l0
    ys = np.arange(ny) * tile_l0
    mx0 = np.clip(np.floor(xs * scale).astype(int), 0, mask.shape[1])
    mx1 = np.clip(np.ceil((xs + tile_l0) * scale).astype(int), 0, mask.shape[1])
    my0 = np.clip(np.floor(ys * scale).astype(int), 0, mask.shape[0])
    my1 = np.clip(np.ceil((ys + tile_l0) * scale).astype(int), 0, mask.shape[0])
    X0, Y0 = np.meshgrid(mx0, my0)
    X1, Y1 = np.meshgrid(mx1, my1)
    area = np.maximum((X1 - X0) * (Y1 - Y0), 1)
    s = integ[Y1, X1] - integ[Y0, X1] - integ[Y1, X0] + integ[Y0, X0]
    frac = s / area
    keep = frac >= min_frac
    gy, gx = np.nonzero(keep)
    coords = np.stack([np.round(xs[gx]), np.round(ys[gy])], axis=1).astype(np.int64)
    return coords, frac[keep]


def draw_qc(thumb: np.ndarray, mask: np.ndarray, coords: np.ndarray, tile_l0: float, scale: float) -> np.ndarray:
    from scipy import ndimage

    img = thumb.copy()
    edge = mask & ~ndimage.binary_erosion(mask, iterations=1)
    img[edge] = (0, 200, 0)
    t = max(1, int(round(tile_l0 * scale)))
    H, W = img.shape[:2]
    for x, y in coords:
        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = min(W - 1, x0 + t - 1), min(H - 1, y0 + t - 1)
        img[y0:y1 + 1, [x0, x1]] = (30, 30, 220)
        img[[y0, y1], x0:x1 + 1] = (30, 30, 220)
    return img


def tile_case(case: dict, cfg: Config) -> dict:
    from PIL import Image

    wc = cfg.wsi
    wp = WorkPaths(cfg)
    sid = case["study_id"]
    tmpp, tpx = float(wc.tiling.target_mpp), int(wc.tiling.tile_px)
    spec = tile_spec_name(tmpp, tpx)
    out = wp.wsi_tiles(spec, sid)
    if out.exists() and not case.get("_overwrite"):
        coords, attrs = read_tiles(out)
        return {"status": "skipped", "message": "已有切块结果", "n_tiles": int(len(coords))}
    path = resolve_path(cfg, case["slide_path"])
    with SlideReader(path, wc.get("backend", "auto"), wc.get("default_base_mpp")) as rd:
        thumb, thumb_mpp = rd.thumbnail(float(wc.tissue.thumb_mpp))
        mask, tinfo = detect_tissue(thumb, thumb_mpp, wc.tissue)
        coords, frac = plan_tiles(mask, thumb_mpp, rd.base_mpp, rd.level_dimensions[0], tmpp, tpx,
                                  float(wc.tiling.min_tissue_fraction))
        level = rd.best_level_for_mpp(tmpp)
        read_px = int(round(tpx * tmpp / rd.level_mpp(level)))
        tile_l0 = tpx * tmpp / rd.base_mpp
        attrs = {
            "spec": spec, "target_mpp": tmpp, "tile_px": tpx, "base_mpp": rd.base_mpp, "mpp_source": rd.mpp_source,
            "read_level": level, "read_size_px": read_px, "tile_size_level0": tile_l0,
            "level0_width": rd.level_dimensions[0][0], "level0_height": rd.level_dimensions[0][1],
            "mask_mpp": thumb_mpp, "tissue_area_mm2": tinfo["tissue_area_mm2"],
            "n_fragments": tinfo["n_fragments"], "backend": rd.backend,
        }
        write_h5(out, {"coords": coords, "tissue_fraction": frac.astype(np.float32),
                       "tissue_mask": mask.astype(np.uint8)}, attrs)
        qc = draw_qc(thumb, mask, coords, tile_l0, rd.base_mpp / thumb_mpp)
        qp = wp.wsi_qc(spec, sid)
        qp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(qc).save(qp)
    res = {"n_tiles": int(len(coords)), "tissue_area_mm2": round(tinfo["tissue_area_mm2"], 3),
           "n_fragments": tinfo["n_fragments"], "base_mpp": round(rd.base_mpp, 4), "read_level": level,
           "read_size_px": read_px}
    res["message"] = f"组织 {tinfo['tissue_area_mm2']:.2f} mm²，{len(coords)} 个切块"
    if len(coords) == 0:
        res["status"] = "warning"
        res["message"] += "；没有切块（组织检测可能失败，请看质控图）"
    elif len(coords) < 10:
        res["status"] = "warning"
        res["message"] += "；切块很少"
    return res


def make_overview(cfg: Config, cases: List[dict], run: Run) -> List[Path]:
    from PIL import Image, ImageDraw

    wp = WorkPaths(cfg)
    spec = tile_spec_name(cfg.wsi.tiling.target_mpp, cfg.wsi.tiling.tile_px)
    items = [(c["study_id"], wp.wsi_qc(spec, c["study_id"])) for c in cases]
    items = [(s, p) for s, p in items if p.exists()]
    per = int(cfg.wsi.qc.overview_per_page)
    cell = 320
    pages = []
    for k in range(math.ceil(len(items) / per)):
        chunk = items[k * per:(k + 1) * per]
        ncol = min(6, len(chunk))
        nrow = math.ceil(len(chunk) / ncol)
        page = Image.new("RGB", (ncol * cell, nrow * (cell + 16)), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        for i, (sid, p) in enumerate(chunk):
            im = Image.open(p)
            im.thumbnail((cell - 4, cell - 4))
            x, y = (i % ncol) * cell, (i // ncol) * (cell + 16)
            page.paste(im, (x + 2, y + 16))
            draw.text((x + 4, y + 2), sid, fill=(0, 0, 0))
        out = run.dir / f"overview_p{k + 1:02d}.png"
        page.save(out)
        pages.append(out)
    return pages


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "wsi_tiling", args.run_name)
    log = run.logger
    res = load_manifest(cfg, log)
    miss = missing_path_set(res)
    cases = [c for c in to_cases(res.df) if c.get("slide_path") and (c["study_id"], "slide_path") not in miss]
    for c in cases:
        c["_overwrite"] = args.overwrite
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    spec = tile_spec_name(cfg.wsi.tiling.target_mpp, cfg.wsi.tiling.tile_px)
    log.info("切块规格：%s（组织占比 ≥ %s）", spec, cfg.wsi.tiling.min_tissue_fraction)
    results = process_cases(cases, tile_case, cfg, log, resolve_workers(args, cfg), "组织检测与切块")
    run.write_table(results, "status.csv")
    pages = make_overview(cfg, cases, run)
    n = [r.get("n_tiles", 0) for r in results if r.get("status") != "failed"]
    if n:
        log.info("切块数：中位数 %d，最少 %d，最多 %d；总览图 %d 页", int(np.median(n)), min(n), max(n), len(pages))
    log.info("提示：质控图是患者切片图像，只在服务器上查看，不要拷出。")
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("病理切片组织检测与 20× 切块（保存坐标），生成质控图。")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
