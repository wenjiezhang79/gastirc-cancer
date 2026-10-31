"""组织检测：在低分辨率缩略图上分割组织区域。

步骤：HSV 饱和度 Otsu 阈值（不低于下限）→ 去掉白色背景和黑色杂质 → 去掉蓝/绿/黑笔迹
→ 闭运算、填洞 → 去掉面积过小的碎片。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def otsu_threshold(values: np.ndarray) -> float:
    v = values.ravel().astype(np.uint8)
    if v.size == 0:
        return 0.0
    hist = np.bincount(v, minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    if np.all(np.isnan(sigma_b)):
        return 0.0
    return float(np.nanargmax(sigma_b))


def pen_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    blue = (r < 100) & (b > 150) & (b - r > 80)
    green = (g > r + 30) & (g > b + 15)
    return blue | green


def detect_tissue(thumb: np.ndarray, thumb_mpp: float, tc) -> Tuple[np.ndarray, dict]:
    """返回 (组织掩码 bool[h,w], 信息)。tc 为配置 wsi.tissue。"""
    from PIL import Image
    from scipy import ndimage

    hsv = np.asarray(Image.fromarray(thumb).convert("HSV"))
    s, v = hsv[..., 1], hsv[..., 2]
    not_black = v >= int(tc.get("min_value_black", 40))
    cand = s[not_black]
    thr = max(otsu_threshold(cand) if cand.size else 0.0, float(tc.get("min_saturation", 20)))
    m = (s >= thr) & not_black
    m &= ~((v > int(tc.get("max_value_white", 235))) & (s < thr))
    if tc.get("filter_pen", True):
        pen = pen_mask(thumb)
        pen = ndimage.binary_dilation(pen, iterations=2)
        m &= ~pen
    it = int(tc.get("close_iters", 2))
    if it > 0:
        m = ndimage.binary_closing(m, iterations=it)
    m = ndimage.binary_fill_holes(m)
    px_area_mm2 = (thumb_mpp / 1000.0) ** 2
    min_px = max(1, int(float(tc.get("min_area_mm2", 0.02)) / px_area_mm2))
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_px
        m = keep[lab]
        n_kept = int(keep.sum())
    else:
        n_kept = 0
    info = {"saturation_threshold": thr, "tissue_area_mm2": float(m.sum() * px_area_mm2),
            "n_fragments": n_kept}
    return m.astype(bool), info
