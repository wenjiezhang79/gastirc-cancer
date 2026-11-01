"""DICOM 读取：扫描头信息、按序列分组、期相识别、序列选择、读取为 SimpleITK 图像。

隐私：只读取下面 ``HEADER_TAGS`` 中列出的字段；患者姓名、ID、生日、检查号、日期、机构等一律不读。
UID 只在内部用于分组，输出时只给出哈希后的短编号。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.errors import CaseError

# 只读这些字段（白名单）
HEADER_TAGS = [
    "SpecificCharacterSet", "SOPClassUID", "SeriesInstanceUID", "FrameOfReferenceUID", "Modality", "SeriesNumber", "InstanceNumber",
    "ImageType", "SeriesDescription", "ProtocolName", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns", "Manufacturer",
    "ManufacturerModelName", "ConvolutionKernel", "KVP", "ContrastBolusAgent", "ContrastBolusStartTime",
    "AcquisitionTime", "SeriesTime", "AcquisitionNumber", "GantryDetectorTilt", "RescaleSlope", "RescaleIntercept",
]

SHARP_KERNEL_PATTERNS = ["lung", "bone", "sharp", "detail", "edge", "b6", "b7", "b8", "br6", "br7", "br8",
                         "bl5", "bl6", "i70", "i80", "yc", "yd", "fc5", "fc8"]


@dataclass
class SeriesInfo:
    uid: str                                  # 内部使用，不输出
    files: List[Path] = field(default_factory=list)   # 按层位置排序后的文件（内部使用）
    series_number: Optional[int] = None
    modality: str = ""
    image_type: Tuple[str, ...] = ()
    description: str = ""
    protocol: str = ""                        # 只用于期相识别，不输出
    for_uid: str = ""                         # 内部使用
    rows: Optional[int] = None
    cols: Optional[int] = None
    pixel_spacing: Optional[Tuple[float, float]] = None   # (行间距 y, 列间距 x)
    slice_thickness: Optional[float] = None
    spacing_median: Optional[float] = None
    spacing_min: Optional[float] = None
    spacing_max: Optional[float] = None
    uniform: bool = True
    n_gaps: int = 0
    z_coverage: Optional[float] = None
    axial: bool = True
    tilt: float = 0.0
    manufacturer: str = ""
    model: str = ""
    kernel: str = ""
    kvp: Optional[float] = None
    contrast_agent: bool = False
    acq_time_s: Optional[float] = None        # 当天秒数（不含日期）
    bolus_time_s: Optional[float] = None
    phase: str = "unknown"
    phase_basis: str = ""

    @property
    def n_images(self) -> int:
        return len(self.files)

    @property
    def is_localizer(self) -> bool:
        return any("LOCALIZER" in t.upper() for t in self.image_type)

    @property
    def for_hash(self) -> str:
        return hashlib.sha1(self.for_uid.encode()).hexdigest()[:10] if self.for_uid else ""

    @property
    def contrast_delay_s(self) -> Optional[float]:
        if self.acq_time_s is None or self.bolus_time_s is None:
            return None
        d = self.acq_time_s - self.bolus_time_s
        if d < -43200:
            d += 86400
        return d

    @property
    def kernel_sharp(self) -> bool:
        k = self.kernel.lower()
        return any(p in k for p in SHARP_KERNEL_PATTERNS)


def _time_to_s(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.match(r"^(\d{2}):?(\d{2})?:?(\d{2})?(\.\d+)?$", s)
    if not m:
        return None
    h = int(m.group(1))
    mi = int(m.group(2) or 0)
    se = int(m.group(3) or 0)
    fr = float(m.group(4) or 0)
    return h * 3600 + mi * 60 + se + fr


def _float(v) -> Optional[float]:
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v).strip()


def read_header(path: Path):
    """只读白名单字段；不是 DICOM 文件时返回 None。"""
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True, specific_tags=HEADER_TAGS)
    except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError, KeyError, IndexError):
        return None
    except Exception:
        return None
    if not getattr(ds, "SeriesInstanceUID", None) or getattr(ds, "ImagePositionPatient", None) is None:
        return None
    return ds


def iter_files(root: Path) -> List[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and p.name.upper() != "DICOMDIR":
            if p.suffix.lower() in (".txt", ".xml", ".json", ".jpg", ".png", ".pdf", ".csv", ".xlsx", ".nii", ".gz"):
                continue
            out.append(p)
    return out


def scan_dicom_dir(root: Path, phase_keywords: Optional[dict] = None, phase_delay_s: Optional[dict] = None,
                   spacing_tol: float = 0.1) -> List[SeriesInfo]:
    """扫描一个文件夹（递归），按 SeriesInstanceUID 分组，返回各序列信息（按 SeriesNumber 排序）。"""
    groups: Dict[str, List[tuple]] = {}
    first: Dict[str, object] = {}
    for f in iter_files(root):
        ds = read_header(f)
        if ds is None:
            continue
        uid = str(ds.SeriesInstanceUID)
        groups.setdefault(uid, []).append((f, ds))
        first.setdefault(uid, ds)
    series = []
    for uid, items in groups.items():
        series.append(_build_series(uid, items, phase_keywords, phase_delay_s, spacing_tol))
    series.sort(key=lambda s: (s.series_number if s.series_number is not None else 1 << 30, s.description))
    return series


def _build_series(uid: str, items: List[tuple], phase_keywords, phase_delay_s, spacing_tol) -> SeriesInfo:
    ds0 = items[0][1]
    s = SeriesInfo(uid=uid)
    s.series_number = int(ds0.SeriesNumber) if getattr(ds0, "SeriesNumber", None) not in (None, "") else None
    s.modality = _str(getattr(ds0, "Modality", ""))
    it = getattr(ds0, "ImageType", None)
    s.image_type = tuple(str(x) for x in it) if it is not None and not isinstance(it, str) else ((str(it),) if it else ())
    s.description = _str(getattr(ds0, "SeriesDescription", ""))
    s.protocol = _str(getattr(ds0, "ProtocolName", ""))
    s.for_uid = _str(getattr(ds0, "FrameOfReferenceUID", ""))
    s.rows = int(ds0.Rows) if getattr(ds0, "Rows", None) else None
    s.cols = int(ds0.Columns) if getattr(ds0, "Columns", None) else None
    ps = getattr(ds0, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        s.pixel_spacing = (float(ps[0]), float(ps[1]))
    s.slice_thickness = _float(getattr(ds0, "SliceThickness", None))
    s.manufacturer = _str(getattr(ds0, "Manufacturer", ""))
    s.model = _str(getattr(ds0, "ManufacturerModelName", ""))
    s.kernel = _str(getattr(ds0, "ConvolutionKernel", ""))
    s.kvp = _float(getattr(ds0, "KVP", None))
    s.contrast_agent = bool(_str(getattr(ds0, "ContrastBolusAgent", "")))
    s.tilt = _float(getattr(ds0, "GantryDetectorTilt", None)) or 0.0
    times = [_time_to_s(getattr(d, "AcquisitionTime", None)) for _, d in items]
    times = [t for t in times if t is not None]
    s.acq_time_s = min(times) if times else _time_to_s(getattr(ds0, "SeriesTime", None))
    s.bolus_time_s = _time_to_s(getattr(ds0, "ContrastBolusStartTime", None))

    # 按层位置排序
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        r = np.array([float(x) for x in iop[:3]])
        c = np.array([float(x) for x in iop[3:]])
        normal = np.cross(r, c)
        s.axial = abs(normal[2]) > 0.9
    else:
        normal = np.array([0.0, 0.0, 1.0])
    pos = []
    for f, d in items:
        ipp = getattr(d, "ImagePositionPatient", None)
        pos.append(float(np.dot(normal, [float(x) for x in ipp])) if ipp is not None else 0.0)
    order = np.argsort(pos)
    s.files = [items[i][0] for i in order]
    zs = np.array(pos)[order]
    if len(zs) >= 2:
        dz = np.diff(zs)
        dz_nonzero = dz[dz > 1e-3]
        if len(dz_nonzero):
            med = float(np.median(dz_nonzero))
            s.spacing_median, s.spacing_min, s.spacing_max = med, float(dz.min()), float(dz.max())
            s.uniform = (s.spacing_max - s.spacing_min) <= spacing_tol * med
            s.n_gaps = int(np.sum(dz > 1.5 * med))
        s.z_coverage = float(zs[-1] - zs[0])
    s.phase, s.phase_basis = guess_phase(s, phase_keywords or {}, phase_delay_s or {})
    return s


def _kw_match(text: str, kw: str) -> bool:
    kw = kw.lower()
    if re.fullmatch(r"[a-z0-9\-]+", kw):
        return re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", text) is not None
    return kw in text


def guess_phase(s: SeriesInfo, keywords: dict, delay_ranges: dict) -> Tuple[str, str]:
    """按关键词（序列描述、协议名）识别期相；没有关键词时按造影延迟时间判断。"""
    text = f"{s.description} {s.protocol}".lower()
    for phase in ("delayed", "arterial", "venous", "non_contrast"):
        for kw in keywords.get(phase, []):
            if _kw_match(text, str(kw)):
                return phase, "关键词"
    d = s.contrast_delay_s
    if d is not None:
        for phase in ("arterial", "venous", "delayed"):
            lo, hi = delay_ranges.get(phase, [None, None])
            if lo is not None and lo <= d < hi:
                return phase, "延迟时间"
    return "unknown", ""


def candidate_series(series: Sequence[SeriesInfo], min_slices: int) -> List[SeriesInfo]:
    return [s for s in series if s.modality in ("CT", "") and not s.is_localizer and s.n_images >= min_slices
            and s.axial]


def select_series(series: Sequence[SeriesInfo], phase: str, preferred_thickness: float, min_slices: int,
                  forced_number: Optional[int] = None) -> Tuple[SeriesInfo, str]:
    """选出分析用序列。返回 (序列, 选择理由)。没有合适序列时抛出 CaseError。"""
    cands = candidate_series(series, min_slices)
    if forced_number is not None:
        for s in series:
            if s.series_number == forced_number:
                return s, f"按清单 ct_series_number={forced_number} 指定"
        raise CaseError(f"清单指定的 ct_series_number={forced_number} 在该检查中不存在"
                        f"（现有序列号：{[x.series_number for x in series]}）")
    if not cands:
        raise CaseError(f"没有可用的 CT 序列（共 {len(series)} 个序列，均为定位像、非轴位或层数少于 {min_slices}）")

    def key(s: SeriesInfo):
        th = s.slice_thickness if s.slice_thickness is not None else (s.spacing_median or 99)
        return (abs(th - preferred_thickness), s.kernel_sharp, -s.n_images)

    target = [s for s in cands if s.phase == phase]
    if target:
        best = sorted(target, key=key)[0]
        return best, f"期相={phase}（{best.phase_basis}），层厚最接近 {preferred_thickness:g} mm"
    unknown = [s for s in cands if s.phase == "unknown"]
    if unknown:
        best = sorted(unknown, key=key)[0]
        return best, f"未识别出{phase}期序列，按层厚/层数选择了序列 {best.series_number}，请人工核对"
    found = sorted({s.phase for s in cands})
    raise CaseError(f"该检查中没有{phase}期序列（识别到的期相：{', '.join(found)}）。"
                    "如期相识别有误，可在清单中用 ct_series_number 指定序列号")


def same_acquisition(a: SeriesInfo, b: SeriesInfo, tol_s: float = 10.0) -> bool:
    """同一次采集的不同重建：同一空间坐标系且采集开始时间接近。"""
    if not a.for_uid or a.for_uid != b.for_uid:
        return False
    if a.acq_time_s is None or b.acq_time_s is None:
        return False
    return abs(a.acq_time_s - b.acq_time_s) <= tol_s


def select_thin_series(series: Sequence[SeriesInfo], chosen: SeriesInfo, min_slices: int,
                       max_ratio: float = 0.6) -> Optional[SeriesInfo]:
    """在与分析序列同一次采集的重建中，找最薄的软组织卷积核序列（供医生在薄层上标注）。"""
    th0 = chosen.slice_thickness or chosen.spacing_median or 0
    cands = [s for s in candidate_series(series, min_slices)
             if s.uid != chosen.uid and same_acquisition(s, chosen) and not s.kernel_sharp
             and (s.slice_thickness or s.spacing_median or 99) <= th0 * max_ratio]
    if not cands:
        return None
    return sorted(cands, key=lambda s: (s.slice_thickness or s.spacing_median or 99, -s.n_images))[0]


def read_series_image(s: SeriesInfo, allow_nonuniform: bool = False):
    """读取序列为 SimpleITK 图像（HU）。"""
    import SimpleITK as sitk

    if abs(s.tilt) > 0.5:
        raise CaseError(f"序列 {s.series_number} 机架倾斜 {s.tilt:g}°，暂不支持。请导出无倾斜的重建或在清单中指定其他序列")
    if not s.uniform and not allow_nonuniform:
        raise CaseError(f"序列 {s.series_number} 层间距不均匀（{s.spacing_min:.2f}–{s.spacing_max:.2f} mm，"
                        f"疑似缺层 {s.n_gaps} 处），请检查导出是否完整")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in s.files])
    reader.SetOutputPixelType(sitk.sitkFloat32)
    try:
        img = reader.Execute()
    except RuntimeError as e:
        raise CaseError(f"SimpleITK 读取序列 {s.series_number} 失败：{str(e).splitlines()[0][:200]}") from None
    if s.spacing_median:
        sp = list(img.GetSpacing())
        if abs(sp[2] - s.spacing_median) > 0.01:
            sp[2] = s.spacing_median
            img.SetSpacing(sp)
    return img
