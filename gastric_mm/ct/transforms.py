"""CT 几何与强度变换（纯函数，基于 SimpleITK）。

约定：
- 所有坐标换算都经过**物理坐标**（mm，LPS），因此原始图、薄层图、重采样图、裁剪图之间可以互相对应。
- SimpleITK 的索引顺序为 (x, y, z)；numpy 数组顺序为 (z, y, x)。
- 薄层降到厚层时，先沿 z 方向做滑动平均（模拟厚层重建），再插值，避免直接抽层带来的噪声和混叠。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def sitk():
    import SimpleITK
    return SimpleITK


def orient(img, orientation: str = "LPS"):
    return sitk().DICOMOrient(img, orientation)


def index_to_physical(img, idx: Sequence[float]) -> np.ndarray:
    return np.array(img.TransformContinuousIndexToPhysicalPoint([float(v) for v in idx]))


def physical_to_index(img, p: Sequence[float]) -> np.ndarray:
    return np.array(img.TransformPhysicalPointToContinuousIndex([float(v) for v in p]))


def inside(img, idx: Sequence[float], margin: float = 0.5) -> bool:
    size = img.GetSize()
    return all(-margin <= float(i) <= s - 1 + margin for i, s in zip(idx, size))


def box_average_z(img, target_z: float, min_ratio: float = 1.25):
    """若目标层厚明显大于原层间距，沿 z 方向做宽度≈目标层厚的滑动平均（几何不变）。"""
    S = sitk()
    sz = img.GetSpacing()[2]
    if target_z < sz * min_ratio:
        return img
    k = int(round(target_z / sz))
    if k <= 1:
        return img
    from scipy.ndimage import uniform_filter1d

    arr = S.GetArrayFromImage(img).astype(np.float32)
    arr = uniform_filter1d(arr, size=k, axis=0, mode="nearest")
    out = S.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def make_reference(origin, spacing, size, direction=None):
    S = sitk()
    ref = S.Image([int(v) for v in size], S.sitkUInt8)
    ref.SetSpacing([float(v) for v in spacing])
    ref.SetOrigin([float(v) for v in origin])
    if direction is not None:
        ref.SetDirection([float(v) for v in direction])
    return ref


def spacing_reference(img, spacing: Sequence[float]):
    """与 img 同方向、同物理范围，体素为 spacing 的参考网格。"""
    old_sp = np.array(img.GetSpacing())
    old_size = np.array(img.GetSize())
    new_sp = np.array([float(v) for v in spacing])
    new_size = np.maximum(1, np.round(old_size * old_sp / new_sp)).astype(int)
    # 让新网格与原网格的第一个体素中心对齐
    return make_reference(img.GetOrigin(), new_sp, new_size, img.GetDirection())


def roi_reference(center_phys: Sequence[float], spacing: Sequence[float], size: Sequence[int]):
    """以物理点为中心、方向为单位阵（LPS 轴）的裁剪网格；中心点落在体素 size//2 上。"""
    spacing = np.array([float(v) for v in spacing])
    size = np.array([int(v) for v in size])
    origin = np.array(center_phys, dtype=float) - (size // 2) * spacing
    return make_reference(origin, spacing, size)


def resample_image(img, ref, default: float = -1024.0, average_thin: bool = True):
    """线性插值把 CT 重采样到参考网格（薄层先多层平均）。返回 float32 图像。"""
    S = sitk()
    src = S.Cast(img, S.sitkFloat32)
    if average_thin:
        src = box_average_z(src, ref.GetSpacing()[2])
    return S.Resample(src, ref, S.Transform(), S.sitkLinear, float(default), S.sitkFloat32)


def resample_mask(mask, ref, threshold: float = 0.5, average_thin: bool = True):
    """把二值分割重采样到参考网格：按体积占比插值后取阈值。返回 uint8 图像。"""
    S = sitk()
    m = S.Cast(mask > 0, S.sitkFloat32)
    if average_thin:
        m = box_average_z(m, ref.GetSpacing()[2])
    r = S.Resample(m, ref, S.Transform(), S.sitkLinear, 0.0, S.sitkFloat32)
    return S.Cast(r >= threshold, S.sitkUInt8)


def resample_label(mask, ref):
    S = sitk()
    return S.Resample(mask, ref, S.Transform(), S.sitkNearestNeighbor, 0, mask.GetPixelID())


def outside_fraction(img, ref) -> float:
    """参考网格中落在原图像范围之外的体素比例（裁剪时需要填充的比例）。"""
    S = sitk()
    ones = S.Image(img.GetSize(), S.sitkUInt8) + 1
    ones.CopyInformation(img)
    r = S.Resample(ones, ref, S.Transform(), S.sitkNearestNeighbor, 0, S.sitkUInt8)
    a = S.GetArrayViewFromImage(r)
    return float(1.0 - a.mean())


def mask_centroid_physical(mask) -> Optional[np.ndarray]:
    S = sitk()
    a = S.GetArrayViewFromImage(mask)
    zyx = np.argwhere(a > 0)
    if len(zyx) == 0:
        return None
    c = zyx.mean(axis=0)[::-1]
    return index_to_physical(mask, c)


def mask_volume_ml(mask) -> float:
    S = sitk()
    a = S.GetArrayViewFromImage(mask)
    return float((a > 0).sum() * np.prod(mask.GetSpacing()) / 1000.0)


def distance_to_mask_mm(mask, p_phys: Sequence[float]) -> Optional[float]:
    """物理点到分割的距离（mm）；点在分割内为 0；点在分割图像范围外返回 None。"""
    S = sitk()
    idx = physical_to_index(mask, p_phys)
    if not inside(mask, idx):
        return None
    a = S.GetArrayViewFromImage(mask)
    if not (a > 0).any():
        return None
    dist = S.SignedMaurerDistanceMap(S.Cast(mask > 0, S.sitkUInt8), insideIsPositive=False,
                                     squaredDistance=False, useImageSpacing=True)
    i = np.clip(np.round(idx).astype(int), 0, np.array(mask.GetSize()) - 1)
    d = float(dist.GetPixel([int(v) for v in i]))
    return max(0.0, d)


def window_normalize(hu: np.ndarray, level: float, width: float, mode: str = "minmax") -> np.ndarray:
    """窗宽窗位截断后标准化：minmax → [0,1]；zscore → 均值 0 方差 1；none → 截断后的 HU。"""
    lo, hi = level - width / 2.0, level + width / 2.0
    x = np.clip(hu.astype(np.float32), lo, hi)
    if mode == "minmax":
        return (x - lo) / (hi - lo)
    if mode == "zscore":
        sd = float(x.std())
        return (x - float(x.mean())) / (sd if sd > 1e-6 else 1.0)
    if mode == "none":
        return x
    raise ValueError(f"未知的标准化方式：{mode}（可选 minmax / zscore / none）")


def to_int16_hu(img, clip: Tuple[float, float] = (-1024, 3071)):
    S = sitk()
    arr = S.GetArrayFromImage(img)
    arr = np.clip(np.round(arr), clip[0], clip[1]).astype(np.int16)
    out = S.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out
