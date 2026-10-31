"""CT 质控图：单例三视图与多例总览拼图。图中只标 study_id，不含任何身份信息。

注意：质控图本身是患者影像，只在服务器上查看，不要拷出。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _contour(ax, m2d: Optional[np.ndarray], color: str, aspect=None, origin="upper"):
    if m2d is None or not np.any(m2d):
        return
    ax.contour(m2d.astype(float), levels=[0.5], colors=[color], linewidths=0.8, origin=origin)


def case_qc(img: np.ndarray, spacing_xyz: Sequence[float], center_xyz: Sequence[int], path: Path, title: str,
            mask: Optional[np.ndarray] = None, stomach: Optional[np.ndarray] = None,
            point_xyz: Optional[Sequence[float]] = None) -> None:
    """img/mask/stomach 为 (z, y, x) 数组，img 已标准化到 [0,1]（或任意范围，按 1–99 百分位显示）。"""
    plt = _plt()
    cx, cy, cz = [int(np.clip(round(v), 0, s - 1)) for v, s in zip(center_xyz, img.shape[::-1])]
    sx, sy, sz = spacing_xyz
    lo, hi = np.percentile(img, [1, 99]) if img.max() > 1.5 or img.min() < -0.5 else (0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    views = [
        ("axial", img[cz], None if mask is None else mask[cz], None if stomach is None else stomach[cz],
         sy / sx, "upper", (cx, cy)),
        ("coronal", img[:, cy, :], None if mask is None else mask[:, cy, :], None if stomach is None else stomach[:, cy, :],
         sz / sx, "lower", (cx, cz)),
        ("sagittal", img[:, :, cx], None if mask is None else mask[:, :, cx], None if stomach is None else stomach[:, :, cx],
         sz / sy, "lower", (cy, cz)),
    ]
    for ax, (name, im, m, st, aspect, origin, pt) in zip(axes, views):
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, aspect=aspect, origin=origin, interpolation="nearest")
        _contour(ax, st, "yellow", origin=origin)
        _contour(ax, m, "red", origin=origin)
        if point_xyz is not None:
            ax.plot([pt[0]], [pt[1]], marker="+", color="cyan", markersize=12, mew=1.5)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def overview(items: List[Tuple[str, np.ndarray, Optional[np.ndarray], str]], out_prefix: Path,
             per_page: int = 48, aspect: float = 1.0) -> List[Path]:
    """items: (study_id, 轴位图, 轴位分割或 None, 备注)。每页一个网格，返回生成的文件列表。"""
    plt = _plt()
    paths = []
    n_pages = max(1, math.ceil(len(items) / per_page))
    for p in range(n_pages):
        chunk = items[p * per_page:(p + 1) * per_page]
        if not chunk:
            break
        ncol = min(8, len(chunk))
        nrow = math.ceil(len(chunk) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.2 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (sid, im, m, note) in zip(axes.ravel(), chunk):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
            _contour(ax, m, "red")
            h, w = im.shape
            ax.plot([w // 2], [h // 2], marker="+", color="cyan", markersize=8)
            ax.set_title(f"{sid}{(' ' + note) if note else ''}", fontsize=7)
        fig.tight_layout()
        path = Path(f"{out_prefix}_p{p + 1:02d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=80)
        plt.close(fig)
        paths.append(path)
    return paths
