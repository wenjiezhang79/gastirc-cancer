"""CT ROI 特征文件的读写（v2 CT 单模态模型与 v3 融合共用的接口）。

文件 ``<work_dir>/ct/features/<编码器>/<study_id>.h5``：
- 每个 ROI 角色一个数据集：``tumor`` [n, D]、``regional`` [n, D]（2.5D 编码器 n = 层数；3D 编码器 n = 1）；
- ``<角色>_z_mm`` [n]：各层相对 ROI 中心的 z 偏移（mm，3D 编码器为 0）；
- attrs：encoder、kind（2d / 3d）、feature_dim、roles（逗号分隔）、``<角色>_spec``（使用的裁剪规格）、
  ``<角色>_center``（point / mask）、weights_sha256、precision 等。
文件中不保存原始文件名或路径。缺少某个 ROI（如没有中心点也没有分割）时不写该数据集，模型按缺失处理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..wsi.features_io import _clean_attr, write_h5

ROLES = ("tumor", "regional")


def write_ct_features(path: Path, rois: Dict[str, np.ndarray], attrs: dict,
                      z_mm: Optional[Dict[str, np.ndarray]] = None, dtype: str = "float32") -> None:
    if not rois:
        raise ValueError("没有任何 ROI 特征可写")
    data: dict = {}
    dims = set()
    for role, f in rois.items():
        f = np.asarray(f)
        if f.ndim != 2 or len(f) == 0:
            raise ValueError(f"ROI {role} 的特征应为非空二维数组 [n, D]，收到形状 {f.shape}")
        if not np.isfinite(f).all():
            raise ValueError(f"ROI {role} 的特征中有 NaN/Inf")
        dims.add(int(f.shape[1]))
        data[role] = f.astype(dtype)
        z = (z_mm or {}).get(role)
        data[f"{role}_z_mm"] = np.zeros(len(f), np.float32) if z is None else np.asarray(z, np.float32)
    if len(dims) != 1:
        raise ValueError(f"各 ROI 的特征维数不一致：{sorted(dims)}")
    a = dict(attrs)
    a.update(feature_dim=int(dims.pop()), roles=",".join(rois))
    write_h5(path, data, a)


def read_ct_features(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    import h5py

    with h5py.File(path, "r") as f:
        attrs = {k: _clean_attr(v) for k, v in f.attrs.items()}
        roles = [r for r in str(attrs.get("roles", "")).split(",") if r and r in f]
        rois = {r: f[r][()] for r in roles}
    return rois, attrs
