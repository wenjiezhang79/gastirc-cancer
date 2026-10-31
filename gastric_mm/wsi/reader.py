"""病理切片读取：OpenSlide 为主，tifffile 为备用。

- OpenSlide：openslide-python + openslide-bin（预编译 wheel，无需编译；TRIDENT 也依赖它）。
- tifffile：只在没有 OpenSlide 时使用；会把整层读入内存，只适合小图（如测试用的假切片）。
  真实 SVS 多为 JPEG 压缩，tifffile 解码还需要 imagecodecs。

µm/px（MPP）读取顺序：openslide.mpp-x → aperio.MPP → 物镜倍率（20× 视为 0.5）→ 配置 wsi.default_base_mpp。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..utils.errors import CaseError

MAG_TO_MPP = 10.0  # 经验：MPP ≈ 10 / 物镜倍率（20× ≈ 0.5 µm/px）


def openslide_available() -> bool:
    try:
        import openslide  # noqa: F401
        return True
    except Exception:
        return False


class SlideReader:
    def __init__(self, path: Path, backend: str = "auto", default_base_mpp: Optional[float] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise CaseError("切片文件不存在")
        if backend == "auto":
            backend = "openslide" if openslide_available() else "tifffile"
        self.backend = backend
        self._levels_cache = {}
        if backend == "openslide":
            import openslide
            try:
                self._os = openslide.OpenSlide(str(self.path))
            except Exception as e:
                raise CaseError(f"OpenSlide 无法打开切片：{type(e).__name__}") from None
            self.level_dimensions: List[Tuple[int, int]] = list(self._os.level_dimensions)
            self.level_downsamples: List[float] = [float(d) for d in self._os.level_downsamples]
            props = dict(self._os.properties)
            self.base_mpp, self.mpp_source = self._mpp_from_props(props, default_base_mpp)
        elif backend == "tifffile":
            import tifffile
            try:
                self._tif = tifffile.TiffFile(str(self.path))
            except Exception as e:
                raise CaseError(f"tifffile 无法打开切片：{type(e).__name__}") from None
            series = self._tif.series[0]
            self._levels = list(series.levels)
            self.level_dimensions = [(int(lv.shape[1]), int(lv.shape[0])) for lv in self._levels]
            w0 = self.level_dimensions[0][0]
            self.level_downsamples = [w0 / float(w) for (w, _) in self.level_dimensions]
            desc = self._tif.pages[0].description or ""
            props = {}
            m = re.search(r"MPP\s*=\s*([0-9.]+)", desc)
            if m:
                props["aperio.MPP"] = m.group(1)
            m = re.search(r"AppMag\s*=\s*([0-9.]+)", desc)
            if m:
                props["aperio.AppMag"] = m.group(1)
            self.base_mpp, self.mpp_source = self._mpp_from_props(props, default_base_mpp)
        else:
            raise CaseError(f"未知的切片读取后端：{backend}（可选 auto / openslide / tifffile）")

    @staticmethod
    def _mpp_from_props(props: dict, default: Optional[float]) -> Tuple[float, str]:
        for key in ("openslide.mpp-x", "aperio.MPP", "tiff.XResolution_mpp"):
            v = props.get(key)
            try:
                if v is not None and float(v) > 0:
                    return float(v), key
            except ValueError:
                pass
        for key in ("openslide.objective-power", "aperio.AppMag"):
            v = props.get(key)
            try:
                if v is not None and float(v) > 0:
                    return MAG_TO_MPP / float(v), f"{key}（按倍率推算）"
            except ValueError:
                pass
        if default:
            return float(default), "配置 wsi.default_base_mpp"
        raise CaseError("切片中读不到 MPP 和物镜倍率。请在配置中设置 wsi.default_base_mpp（20× 扫描通常为 0.5）")

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    def level_mpp(self, level: int) -> float:
        return self.base_mpp * self.level_downsamples[level]

    def best_level_for_mpp(self, target_mpp: float, tol: float = 0.05) -> int:
        """分辨率不低于目标（mpp ≤ target×(1+tol)）的最粗一层。"""
        best = 0
        for i in range(self.level_count):
            if self.level_mpp(i) <= target_mpp * (1 + tol):
                best = i
        return best

    def _level_array(self, level: int) -> np.ndarray:
        if level not in self._levels_cache:
            a = self._levels[level].asarray()
            if a.ndim == 3 and a.shape[2] == 4:
                a = a[:, :, :3]
            self._levels_cache = {level: a}
        return self._levels_cache[level]

    def read_region(self, x: int, y: int, level: int, w: int, h: int) -> np.ndarray:
        """读取区域，x/y 为 level-0 坐标，w/h 为该层像素。返回 (h, w, 3) uint8，超出范围部分为白色。"""
        if self.backend == "openslide":
            im = self._os.read_region((int(x), int(y)), int(level), (int(w), int(h)))
            a = np.asarray(im)
            rgb = a[:, :, :3].astype(np.float32)
            alpha = a[:, :, 3:4].astype(np.float32) / 255.0
            return (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        arr = self._level_array(level)
        ds = self.level_downsamples[level]
        lx, ly = int(round(x / ds)), int(round(y / ds))
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        H, W = arr.shape[:2]
        x0, y0, x1, y1 = max(0, lx), max(0, ly), min(W, lx + w), min(H, ly + h)
        if x1 > x0 and y1 > y0:
            out[y0 - ly:y1 - ly, x0 - lx:x1 - lx] = arr[y0:y1, x0:x1, :3]
        return out

    def thumbnail(self, target_mpp: float) -> Tuple[np.ndarray, float]:
        """返回约 target_mpp 分辨率的缩略图及其实际 µm/px。"""
        from PIL import Image

        w0, h0 = self.level_dimensions[0]
        tw = max(1, int(round(w0 * self.base_mpp / target_mpp)))
        th = max(1, int(round(h0 * self.base_mpp / target_mpp)))
        if self.backend == "openslide":
            level = self._os.get_best_level_for_downsample(target_mpp / self.base_mpp)
            lw, lh = self.level_dimensions[level]
            im = self._os.read_region((0, 0), level, (lw, lh))
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            img = bg.resize((tw, th), Image.BILINEAR)
        else:
            level = self.best_level_for_mpp(target_mpp)
            img = Image.fromarray(self._level_array(level)).resize((tw, th), Image.BILINEAR)
        actual = w0 * self.base_mpp / tw
        return np.asarray(img.convert("RGB")), actual

    def close(self) -> None:
        try:
            if self.backend == "openslide":
                self._os.close()
            else:
                self._tif.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
