"""病理特征与切块坐标的 HDF5 读写（统一格式）。

特征文件 ``<work_dir>/wsi/features/<编码器>/<study_id>.h5``：
- ``features``  [N, D] float32/float16
- ``coords``    [N, 2] int64，level-0 像素坐标（切块左上角）
- attrs：encoder、feature_dim、n_tiles、source（trident / gastric_mm）、倍率与切块大小等

文件中不保存原始文件名或路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

FORBIDDEN_ATTRS = {"name", "savetodir", "slide_path", "path", "filename", "wsi_path", "slide_name"}


def _clean_attr(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return v


def write_h5(path: Path, datasets: dict, attrs: Optional[dict] = None) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in (attrs or {}).items():
            if v is None or k in FORBIDDEN_ATTRS:
                continue
            f.attrs[k] = v
    tmp.replace(path)


def write_features(path: Path, features: np.ndarray, coords: np.ndarray, attrs: dict, dtype: str = "float32") -> None:
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"特征应为二维数组 [N, D]，收到形状 {features.shape}")
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    if len(coords) != len(features):
        raise ValueError(f"特征数 {len(features)} 与坐标数 {len(coords)} 不一致")
    a = dict(attrs)
    a.update(feature_dim=int(features.shape[1]), n_tiles=int(features.shape[0]))
    write_h5(path, {"features": features.astype(dtype), "coords": coords}, a)


def read_features(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    import h5py

    with h5py.File(path, "r") as f:
        feats = f["features"][()]
        coords = f["coords"][()] if "coords" in f else np.zeros((len(feats), 2), np.int64)
        attrs = {k: _clean_attr(v) for k, v in f.attrs.items()}
    return feats, coords, attrs


def read_tiles(path: Path) -> Tuple[np.ndarray, dict]:
    import h5py

    with h5py.File(path, "r") as f:
        coords = f["coords"][()]
        attrs = {k: _clean_attr(v) for k, v in f.attrs.items()}
    return coords, attrs
