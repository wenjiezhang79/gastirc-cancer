"""假数据生成器：生成假的 SVS 切片、DICOM CT、肿瘤/胃分割、TRIDENT 特征和病例清单。

用途：测试和在服务器上验证环境。所有数据都是随机生成的，不含任何真实患者信息。
为了检验隐私保护，文件夹名和 DICOM 头中故意放入了**假的**住院号和姓名（"金丝雀"），
测试会检查这些字符串没有出现在任何输出和日志里。

命令行：
    python -m gastric_mm.fake_data --config configs/fake_demo.yaml

生成位置为配置中的 paths.root；若该目录已有非假数据内容，程序会拒绝写入。
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, load_config, resolve_path, root_dir
from .paths import WorkPaths
from .utils.cli import base_parser, run_main
from .utils.errors import UserError
from .utils.runio import Run

MARKER = ".fake_data_marker"
CENTER_PREFIX = {"internal": "C1", "ext1": "C2", "ext2": "C3", "ext3": "C4"}
FAKE_NAMES = [("张三", "ZHANG^SAN"), ("李四", "LI^SI"), ("王五", "WANG^WU"), ("赵六", "ZHAO^LIU"),
              ("孙七", "SUN^QI"), ("周八", "ZHOU^BA"), ("吴九", "WU^JIU"), ("郑十", "ZHENG^SHI"),
              ("钱一", "QIAN^YI"), ("陈二", "CHEN^ER"), ("刘明", "LIU^MING"), ("杨光", "YANG^GUANG")]
MANUFACTURERS = {"internal": ("SIEMENS", "SOMATOM Force", "Br40"), "ext1": ("GE MEDICAL SYSTEMS", "Revolution CT", "STANDARD"),
                 "ext2": ("Philips", "iCT 256", "B"), "ext3": ("UIH", "uCT 760", "B_SOFT_B")}
VENOUS_DESC = ["Portal Venous 5mm", "静脉期 5mm", "PV 5.0 Br40", "Abdomen Venous 5.0"]
INSTITUTION_CANARY = "FAKE CANARY HOSPITAL 金丝雀医院"

MANIFEST_COLUMNS = [
    "study_id", "center", "slide_path", "ct_dir", "ct_point_x", "ct_point_y", "ct_point_z",
    "tumor_mask_path", "mask_source", "lauren", "her2_ihc", "her2_fish", "msi", "ebv", "pdl1_cps",
    "cldn182_pct", "ki67_pct", "label_source", "population", "os_months", "os_event", "dfs_months",
    "dfs_event", "pfs_months", "pfs_event", "neoadj", "immuno", "ctnm_stage", "age", "sex", "ecog",
    "tumor_visibility", "borrmann", "ct_t", "ct_n", "ct_m", "ct_stage_source",
]


# ============================================================
# 切片
# ============================================================
def make_slide_image(width: int, height: int, rng: np.random.RandomState, scale: float = 1.0) -> np.ndarray:
    """白底 + 2–3 块粉紫色"活检组织" + 一条蓝色笔迹。scale=2 表示 40×（像素加倍、物理尺寸不变）。"""
    from scipy import ndimage

    img = np.full((height, width, 3), 243, dtype=np.uint8)
    img -= rng.randint(0, 5, size=(height, width, 1)).astype(np.uint8)
    n_blobs = rng.randint(2, 4)
    for b in range(n_blobs):
        cx = width * (0.2 + 0.3 * b) + rng.uniform(-0.05, 0.05) * width
        cy = height * rng.uniform(0.35, 0.6)
        ax = width * rng.uniform(0.08, 0.12)
        ay = height * rng.uniform(0.15, 0.25)
        ang = rng.uniform(0, math.pi)
        r_max = max(ax, ay)
        x0, x1 = max(0, int(cx - r_max)), min(width, int(cx + r_max) + 1)
        y0, y1 = max(0, int(cy - r_max)), min(height, int(cy + r_max) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - cx, yy - cy
        u = (dx * math.cos(ang) + dy * math.sin(ang)) / ax
        v = (-dx * math.sin(ang) + dy * math.cos(ang)) / ay
        m = (u * u + v * v) <= 1.0
        sub = img[y0:y1, x0:x1]
        tissue = np.array([228, 150, 196], dtype=np.int16)
        noise = rng.randint(-15, 16, size=(int(m.sum()), 3))
        sub[m] = np.clip(tissue + noise, 0, 255).astype(np.uint8)
        # 细胞核（紫色小点）
        seeds = (rng.rand(*m.shape) < 1.0 / (60 * scale * scale)) & m
        nuc = ndimage.maximum_filter(seeds, size=max(2, int(4 * scale))) & m
        sub[nuc] = (110, 50, 140)
    # 蓝色笔迹：画在下边缘附近
    y0 = int(height * 0.9)
    th = max(6, int(12 * scale))
    img[y0:y0 + th, int(width * 0.1):int(width * 0.6)] = (40, 60, 200)
    return img


def write_fake_svs(path: Path, img: np.ndarray, mpp: float, app_mag: int, filename_canary: str) -> None:
    """用 tifffile 写一个 Aperio 风格的金字塔分块 TIFF（OpenSlide 能识别为 aperio 格式）。"""
    import tifffile

    h, w = img.shape[:2]
    desc0 = (f"Aperio Image Library vFAKE\r\n{w}x{h} [0,0 {w}x{h}] (256x256) RGB|AppMag = {app_mag}"
             f"|StripeWidth = 2040|ScanScope ID = FAKE|Filename = {filename_canary}|MPP = {mpp:.4f}|Left = 0|Top = 0")
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = {"compression": "zlib", "compressionargs": {"level": 1}}
    with tifffile.TiffWriter(str(path), bigtiff=False) as tif:
        tif.write(img, tile=(256, 256), photometric="rgb", description=desc0, metadata=None,
                  resolution=(1e4 / mpp, 1e4 / mpp), resolutionunit="CENTIMETER", **comp)
        level = img
        for _ in range(2):
            level = _downsample(level, 4)
            if min(level.shape[:2]) < 16:
                break
            lh, lw = level.shape[:2]
            tif.write(level, tile=(256, 256) if min(lh, lw) >= 256 else (16 * max(1, min(lh, lw) // 16),) * 2,
                      photometric="rgb", **comp,
                      description=f"Aperio Image Library vFAKE\r\n{w}x{h} -> {lw}x{lh}", metadata=None)


def _downsample(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    x = img[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
    return x.astype(np.uint8)


# ============================================================
# CT
# ============================================================
class Phantom:
    """一个简化的腹部 CT 假体（物理坐标 mm，LPS）。"""

    def __init__(self, rng: np.random.RandomState, z_center: float):
        self.stomach_c = np.array([-25.0, -10.0, z_center]) + rng.uniform(-5, 5, 3)
        self.stomach_ax = np.array([30.0, 22.0, 35.0])
        # 肿瘤：位于胃壁一侧
        d = rng.normal(size=3)
        d[2] *= 0.5
        d /= np.linalg.norm(d)
        self.tumor_c = self.stomach_c + d * self.stomach_ax * 0.8
        self.tumor_ax = np.array([12.0, 10.0, 14.0]) * rng.uniform(0.8, 1.2)
        self.air_lumen = bool(rng.rand() < 0.5)

    def _ell(self, X, Y, Z, c, ax):
        return ((X - c[0]) / ax[0]) ** 2 + ((Y - c[1]) / ax[1]) ** 2 + ((Z - c[2]) / ax[2]) ** 2

    def hu(self, X, Y, Z, phase: str, rng: np.random.RandomState) -> np.ndarray:
        out = np.full(X.shape, -1000.0, dtype=np.float32)
        body = (X / 70.0) ** 2 + ((Y - 5) / 55.0) ** 2
        out[body <= 1.0] = -100.0  # 皮下脂肪
        out[body <= 0.85] = 40.0   # 软组织
        spine = (X / 12.0) ** 2 + ((Y - 40) / 12.0) ** 2
        out[spine <= 1.0] = 500.0
        st = self._ell(X, Y, Z, self.stomach_c, self.stomach_ax)
        inner = self._ell(X, Y, Z, self.stomach_c, self.stomach_ax - 5.0)
        out[st <= 1.0] = 55.0 if phase == "venous" else 70.0
        lumen = inner <= 1.0
        out[lumen] = 5.0
        if self.air_lumen:
            out[lumen & (Y < self.stomach_c[1] - 5)] = -900.0
        tu = self._ell(X, Y, Z, self.tumor_c, self.tumor_ax) <= 1.0
        out[tu] = 90.0 if phase == "venous" else 120.0
        out += rng.normal(0, 8, size=out.shape).astype(np.float32)
        return out

    def stomach_mask(self, X, Y, Z) -> np.ndarray:
        return (self._ell(X, Y, Z, self.stomach_c, self.stomach_ax) <= 1.0).astype(np.uint8)

    def tumor_mask(self, X, Y, Z) -> np.ndarray:
        return (self._ell(X, Y, Z, self.tumor_c, self.tumor_ax) <= 1.0).astype(np.uint8)


def grid(origin, spacing, shape_xyz):
    """返回物理坐标网格 X, Y, Z（数组形状 z, y, x）。"""
    nx, ny, nz = shape_xyz
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    zs = origin[2] + np.arange(nz) * spacing[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return X, Y, Z


def _uid(*parts) -> str:
    from pydicom.uid import generate_uid
    return generate_uid(entropy_srcs=[str(p) for p in parts])


def _save_dicom(ds, path: Path) -> None:
    import pydicom

    major = int(pydicom.__version__.split(".")[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    if major >= 3:
        ds.save_as(str(path), enforce_file_format=True)
    else:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(path), write_like_original=False)


def write_dicom_series(out_dir: Path, hu_zyx: np.ndarray, origin, spacing, thickness: float, info: dict,
                       series_number: int, desc: str, acq_time: str, image_type=("ORIGINAL", "PRIMARY", "AXIAL"),
                       ext: str = "") -> None:
    """把 HU 体数据写成一个 DICOM 序列（每层一个文件）。info 中含患者"金丝雀"字段。"""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ct_storage = "1.2.840.10008.5.1.4.1.1.2"
    series_uid = _uid(info["study_uid"], series_number)
    nz = hu_zyx.shape[0]
    order = list(range(nz))
    # 打乱写入顺序和文件名顺序，检验读取时按位置排序
    rng = np.random.RandomState(series_number)
    rng.shuffle(order)
    for file_i, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = ct_storage
        sop_uid = _uid(series_uid, k)
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(out_dir / f"IM{file_i:04d}{ext}"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8
        ds.SOPClassUID = ct_storage
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "CT"
        ds.PatientName = info["patient_name"]
        ds.PatientID = info["patient_id"]
        ds.PatientBirthDate = "19600101"
        ds.PatientSex = "M"
        ds.InstitutionName = INSTITUTION_CANARY
        ds.ReferringPhysicianName = "DOC^CANARY"
        ds.AccessionNumber = "ACC" + info["patient_id"]
        ds.StudyDate = "20230115"
        ds.SeriesDate = "20230115"
        ds.StudyTime = "100000"
        ds.AcquisitionTime = acq_time
        ds.ContentTime = acq_time
        ds.StudyInstanceUID = info["study_uid"]
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = info["for_uid"]
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1
        ds.AcquisitionNumber = 1
        ds.SeriesDescription = desc
        ds.ProtocolName = "ABDOMEN " + info["patient_id"]  # 自由文本，不应被输出
        ds.ImageType = list(image_type)
        ds.Manufacturer, ds.ManufacturerModelName, ds.ConvolutionKernel = info["manufacturer"]
        ds.KVP = "120"
        ds.ContrastBolusAgent = "Iohexol"
        ds.ContrastBolusStartTime = info["bolus_time"]
        ds.SliceThickness = f"{thickness:g}"
        ds.SpacingBetweenSlices = f"{spacing[2]:g}"
        ds.PixelSpacing = [f"{spacing[1]:g}", f"{spacing[0]:g}"]  # 行间距(y), 列间距(x)
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImagePositionPatient = [f"{origin[0]:.3f}", f"{origin[1]:.3f}", f"{origin[2] + k * spacing[2]:.3f}"]
        ds.SliceLocation = f"{origin[2] + k * spacing[2]:.3f}"
        ds.GantryDetectorTilt = "0"
        ds.Rows, ds.Columns = hu_zyx.shape[1], hu_zyx.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = "-1024"
        ds.RescaleSlope = "1"
        ds.WindowCenter = "40"
        ds.WindowWidth = "400"
        stored = np.clip(np.round(hu_zyx[k]) + 1024, 0, 65535).astype(np.uint16)
        ds.PixelData = stored.tobytes()
        _save_dicom(ds, out_dir / f"IM{file_i:04d}{ext}")


def write_nifti(arr_zyx: np.ndarray, origin, spacing, path: Path) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in origin))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


# ============================================================
# TRIDENT
# ============================================================
def tissue_grid_coords(img: np.ndarray, patch_level0: int) -> np.ndarray:
    """在假切片的组织区域上取网格坐标（level-0 左上角）。"""
    h, w = img.shape[:2]
    sat = img.max(axis=2).astype(int) - img.min(axis=2).astype(int)
    coords = []
    for y in range(0, h - patch_level0 + 1, patch_level0):
        for x in range(0, w - patch_level0 + 1, patch_level0):
            c = sat[y + patch_level0 // 2, x + patch_level0 // 2]
            b = img[y + patch_level0 // 2, x + patch_level0 // 2]
            if c > 40 and not (int(b[2]) > int(b[0]) + 40):
                coords.append((x, y))
    return np.array(coords, dtype=np.int64).reshape(-1, 2)


def write_trident_h5(path: Path, feats: np.ndarray, coords: np.ndarray, patch_px: int, patch_level0: int,
                     level0_mag: int, name: str, savetodir: str) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=feats.astype(np.float32))
        d = f.create_dataset("coords", data=coords)
        d.attrs["patch_size"] = patch_px
        d.attrs["patch_size_level0"] = patch_level0
        d.attrs["level0_magnification"] = level0_mag
        d.attrs["target_magnification"] = 20
        d.attrs["overlap"] = 0
        d.attrs["name"] = name           # 原文件名（金丝雀）：导入时不得复制
        d.attrs["savetodir"] = savetodir  # 原路径（金丝雀）：导入时不得复制


# ============================================================
# 清单与总流程
# ============================================================
def _choice(rng, values, probs):
    return values[int(rng.choice(len(values), p=np.array(probs) / np.sum(probs)))]


def _maybe(rng, v, p_missing=0.15):
    return "" if rng.rand() < p_missing else v


def split_ctnm(text: str) -> dict:
    """把 cT3N1M0 这类文本拆成 ct_t / ct_n / ct_m（假数据用）。"""
    import re

    m = re.match(r"^c?(T[0-9][ab]?)(N[0-9][ab]?)(M[01])$", text or "")
    if not m:
        return {"ct_t": "", "ct_n": "", "ct_m": "", "ct_stage_source": ""}
    return {"ct_t": m.group(1), "ct_n": m.group(2), "ct_m": m.group(3), "ct_stage_source": "report"}


def random_labels(rng: np.random.RandomState) -> dict:
    d = {}
    d["lauren"] = _maybe(rng, _choice(rng, ["intestinal", "diffuse", "mixed"], [0.5, 0.35, 0.15]), 0.1)
    ihc = _maybe(rng, _choice(rng, ["0", "1+", "2+", "3+"], [0.45, 0.25, 0.18, 0.12]))
    d["her2_ihc"] = ihc
    d["her2_fish"] = (_choice(rng, ["pos", "neg"], [0.3, 0.7]) if (ihc == "2+" and rng.rand() < 0.8) else "")
    d["msi"] = _maybe(rng, _choice(rng, ["MSI-H", "MSS"], [0.1, 0.9]))
    d["ebv"] = _maybe(rng, _choice(rng, ["pos", "neg"], [0.08, 0.92]))
    d["pdl1_cps"] = _maybe(rng, f"{min(100, rng.exponential(8)):.0f}")
    d["cldn182_pct"] = _maybe(rng, f"{rng.uniform(0, 100):.0f}")
    d["ki67_pct"] = _maybe(rng, f"{rng.uniform(5, 90):.0f}")
    d["label_source"] = _choice(rng, ["biopsy", "surgery", "post_neoadj"], [0.6, 0.35, 0.05])
    pop = _choice(rng, ["resected", "unresectable"], [0.85, 0.15])
    d["population"] = pop
    d["os_months"] = f"{rng.exponential(40):.1f}"
    d["os_event"] = str(int(rng.rand() < 0.4))
    if pop == "resected":
        d["dfs_months"] = f"{rng.exponential(30):.1f}"
        d["dfs_event"] = str(int(rng.rand() < 0.45))
    else:
        d["pfs_months"] = f"{rng.exponential(8):.1f}"
        d["pfs_event"] = str(int(rng.rand() < 0.8))
    d["neoadj"] = str(int(rng.rand() < 0.2))
    d["immuno"] = str(int(rng.rand() < 0.12))
    d["ctnm_stage"] = _choice(rng, ["cT2N0M0", "cT3N1M0", "cT4aN2M0", "cT4bN3M1"], [1, 1, 1, 0.5])
    d.update(split_ctnm(d["ctnm_stage"]))   # 结构化 cTNM（由文本确定性拆分，不消耗随机数）
    d["age"] = f"{rng.uniform(35, 82):.0f}"
    d["sex"] = _choice(rng, ["M", "F"], [0.7, 0.3])
    d["ecog"] = str(_choice(rng, [0, 1, 2], [0.5, 0.4, 0.1]))
    d["tumor_visibility"] = _choice(rng, ["clear", "uncertain"], [0.85, 0.15])
    d["borrmann"] = _choice(rng, ["II", "III", "IV", "early"], [0.3, 0.45, 0.15, 0.1])
    return d


def generate(cfg: Config, logger=None, overwrite: bool = False) -> dict:
    """生成整套假数据，返回摘要（含金丝雀字符串列表）。"""
    import SimpleITK as sitk  # noqa: F401  确认依赖存在

    fc = cfg.fake_data
    root = root_dir(cfg)
    out_dir = resolve_path(cfg, cfg.paths.output_dir).resolve()
    # 运行目录可能位于 root 之内（如 fake_demo 配置），检查和清理时跳过它
    existing = [p for p in root.iterdir() if p.resolve() != out_dir] if root.exists() else []
    if existing:
        if not (root / MARKER).exists():
            raise UserError(
                f"目标目录 {root} 已有内容且不是假数据目录，为防止覆盖真实数据，拒绝写入。",
                "请在配置中把 paths.root 指向一个新的空目录（例如 data/fake_demo）。",
            )
        if not overwrite:
            raise UserError(f"假数据目录 {root} 已存在。", "如需重新生成，加 --overwrite（会删除该目录下的旧假数据）。")
        for p in existing:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("由 gastric_mm.fake_data 生成的假数据目录\n", encoding="utf-8")
    wp = WorkPaths(cfg)
    rng = np.random.RandomState(int(fc.seed))

    rows: List[dict] = []
    canaries: List[str] = [INSTITUTION_CANARY, "DOC^CANARY", "FAKE CANARY"]
    ct_shape = list(fc.ct_shape)
    ps = float(fc.ct_pixel_spacing)
    n5 = int(fc.ct_n_slices_5mm)
    slide_w, slide_h = (int(v) for v in fc.slide_size_px)
    patch_px = int(fc.trident_patch_px)
    trident_root = root / "trident" / f"20x_{patch_px}px_0px_overlap"
    enc_dims = dict(fc.trident_encoders)

    case_i = 0
    for center, n in fc.n_per_center.items():
        for j in range(int(n)):
            case_i += 1
            sid = f"{CENTER_PREFIX.get(center, 'C9')}-{j + 1:04d}"
            zh, py = FAKE_NAMES[(case_i - 1) % len(FAKE_NAMES)]
            hosp_id = f"ZY{rng.randint(100000, 999999)}"
            canaries += [hosp_id, zh, py, py.replace("^", " ")]
            label = f"{hosp_id}_{zh}"  # 带住院号和姓名的原始文件名（模拟真实情况）
            row = {c: "" for c in MANIFEST_COLUMNS}
            row.update(study_id=sid, center=center)
            row.update(random_labels(rng))
            first = (center == "internal" and j == 0)
            if first:
                row.update(lauren="diffuse", her2_ihc="2+", her2_fish="", msi="MSI-H", label_source="biopsy")
            if center == "internal" and j == 1:
                row.update(label_source="post_neoadj", neoadj="1")

            # ---- 切片 ----
            has_slide = first or rng.rand() < 0.9
            if center == "internal" and j == 2:
                has_slide = False
            if has_slide:
                is40 = (center == "internal" and j == 3)
                scale = 2 if is40 else 1
                img = make_slide_image(slide_w * scale, slide_h * scale, rng, scale)
                sp = root / "raw" / "slides" / f"{label}.svs"
                write_fake_svs(sp, img, mpp=0.5 / scale, app_mag=20 * scale, filename_canary=hosp_id)
                row["slide_path"] = str(sp) if (j % 2 == 1) else str(sp.relative_to(root))
                # TRIDENT 假特征
                patch_l0 = patch_px * scale
                coords = tissue_grid_coords(img, patch_l0)
                for k, (enc, dim) in enumerate(enc_dims.items()):
                    if center == "ext1" and k == 1:
                        continue  # 模拟某个编码器漏提了一张切片
                    feats = rng.normal(size=(len(coords), int(dim))).astype(np.float32)
                    write_trident_h5(trident_root / f"features_{enc}" / f"{label}.h5", feats, coords,
                                     patch_px, patch_l0, 20 * scale, label, str(sp.parent))

            # ---- CT ----
            has_ct = first or rng.rand() < 0.85
            if center == "internal" and j == 4:
                has_ct = False
            if has_ct:
                ct_dir = root / "raw" / "ct" / label
                z0 = -100.0 + rng.uniform(-10, 10)
                origin = (-(ct_shape[1] * ps) / 2, -(ct_shape[0] * ps) / 2, z0)
                ph = Phantom(rng, z_center=z0 + n5 * 5.0 / 2)
                info = {
                    "patient_name": py, "patient_id": hosp_id, "study_uid": _uid(sid, "study"),
                    "for_uid": _uid(sid, "for"), "manufacturer": MANUFACTURERS.get(center, MANUFACTURERS["internal"]),
                    "bolus_time": "101000",
                }
                sp5 = (ps, ps, 5.0)
                X, Y, Z = grid(origin, sp5, (ct_shape[1], ct_shape[0], n5))
                ven = ph.hu(X, Y, Z, "venous", rng)
                ext = ".dcm" if center != "internal" else ""
                write_dicom_series(ct_dir / "S3", ven, origin, sp5, 5.0, info, 3,
                                   VENOUS_DESC[case_i % len(VENOUS_DESC)], "101110", ext=ext)
                # 定位像（应被跳过）
                write_dicom_series(ct_dir / "S1", ven[:1], origin, sp5, 5.0, info, 1, "Topogram",
                                   "100500", image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"), ext=ext)
                if rng.rand() < 0.5 or first:
                    art = ph.hu(X, Y, Z, "arterial", rng)
                    write_dicom_series(ct_dir / "S2", art, origin, sp5, 5.0, info, 2, "Arterial 5mm", "101035", ext=ext)
                if center == "internal" and j in (0, 5):
                    spt = (ps, ps, 1.25)
                    nt = n5 * 4
                    Xt, Yt, Zt = grid(origin, spt, (ct_shape[1], ct_shape[0], nt))
                    thin = ph.hu(Xt, Yt, Zt, "venous", rng)
                    write_dicom_series(ct_dir / "S4", thin, origin, spt, 1.25, info, 4,
                                       "Portal Venous 1.25mm", "101110", ext=ext)
                    write_nifti(ph.tumor_mask(Xt, Yt, Zt), origin, spt,
                                root / "raw" / "masks_thin" / f"{sid}_thin_tumor.nii.gz")
                row["ct_dir"] = str(ct_dir.relative_to(root))
                # 中心点（原始 NIfTI 体素坐标，0 起）
                idx = np.round((ph.tumor_c - np.array(origin)) / np.array(sp5)).astype(int)
                if not (center == "internal" and j == 5):
                    row.update(ct_point_x=str(idx[0]), ct_point_y=str(idx[1]), ct_point_z=str(idx[2]))
                # 肿瘤分割（约一半病例）
                if first or (center == "internal" and j == 5) or rng.rand() < 0.4:
                    mp = root / "raw" / "masks" / f"{label}_tumor.nii.gz"
                    write_nifti(ph.tumor_mask(X, Y, Z), origin, sp5, mp)
                    row.update(tumor_mask_path=str(mp.relative_to(root)), mask_source="manual")
                # 胃分割（模拟 TotalSegmentator 的输出）
                write_nifti(ph.stomach_mask(X, Y, Z), origin, sp5, wp.ct_stomach(sid))
                # 医生的标注文件（演示 import_points）：Slicer 标记点（RAS）和 ITK-SNAP 点标签
                ann = resolve_path(cfg, cfg.ct.points.annotation_dir)
                if j in (0, 1):
                    lps = ph.tumor_c
                    mrk = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
                           "markups": [{"type": "Fiducial", "coordinateSystem": "RAS",
                                        "controlPoints": [{"id": "1", "label": "F-1",
                                                           "position": [-float(lps[0]), -float(lps[1]), float(lps[2])]}]}]}
                    ann.mkdir(parents=True, exist_ok=True)
                    (ann / f"{sid}.mrk.json").write_text(json.dumps(mrk, indent=2), encoding="utf-8")
                elif j == 2:
                    blob = np.zeros_like(X, dtype=np.uint8)
                    d2 = (X - ph.tumor_c[0]) ** 2 + (Y - ph.tumor_c[1]) ** 2 + ((Z - ph.tumor_c[2]) / 2) ** 2
                    blob[d2 <= 9.0] = 1
                    write_nifti(blob, origin, sp5, ann / f"{sid}_point.nii.gz")
            rows.append(row)

    # TRIDENT 中一个清单里没有的文件（模拟多余切片）
    extra = "ZY000000_无名氏"
    canaries += ["ZY000000", "无名氏"]
    for enc, dim in enc_dims.items():
        write_trident_h5(trident_root / f"features_{enc}" / f"{extra}.h5",
                         np.zeros((3, int(dim)), np.float32), np.zeros((3, 2), np.int64), patch_px, patch_px, 20,
                         extra, str(root / "raw" / "slides"))

    import csv
    mpath = resolve_path(cfg, cfg.paths.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {"n_cases": len(rows), "root": str(root), "canaries": sorted(set(canaries))}
    (root / "canary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if logger:
        logger.info("假数据已生成：%d 例，目录 %s", len(rows), root)
    return summary


# ============================================================
# v1 用：假特征队列（埋入已知信号，检验模型确实能学到东西）
# ============================================================
def generate_feature_cohort(cfg: Config, logger=None, overwrite: bool = False) -> dict:
    """直接生成假的病理特征（不生成切片/CT）、病例清单和划分文件。

    埋入的信号：
    - Lauren：每类在一个特征维度上偏移（部分切块）；
    - MSI-H：约 40% 的切块在 3 个维度上明显偏移（模拟局部形态改变）；
    - 预后：潜在风险 z 决定生存时间，并以较弱幅度体现在 2 个维度上；
    - HER2：弱信号；EBV、PD-L1、Claudin18.2 无信号（检验模型不会凭空"学到"）。
    """
    import csv

    from .manifest import validate_manifest, read_manifest_csv
    from .labels import derive_labels
    from .split import make_split
    from .wsi.features_io import write_features

    fc = cfg.fake_data.get("feature_cohort", {}) or {}
    n_int = int(fc.get("n_internal", 200))
    n_ext = int(fc.get("n_external", 60))
    dims = dict(fc.get("encoders", {"uni2_h": 32, "conch_v15": 16}))
    t_lo, t_hi = (int(v) for v in fc.get("tiles", [40, 200]))
    k = float(fc.get("signal", 1.0))           # 信号强度倍数（测试用小队列时调高）
    rng = np.random.RandomState(int(fc.get("seed", 11)))
    root = root_dir(cfg)
    out_dir = resolve_path(cfg, cfg.paths.output_dir).resolve()
    existing = [p for p in root.iterdir() if p.resolve() != out_dir] if root.exists() else []
    if existing and not (root / MARKER).exists():
        raise UserError(f"目标目录 {root} 已有内容且不是假数据目录，拒绝写入。", "请把 paths.root 指向新的空目录。")
    if existing and not overwrite:
        raise UserError(f"假数据目录 {root} 已存在。", "如需重新生成，加 --overwrite。")
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("由 gastric_mm.fake_data 生成的假数据目录（特征队列）\n", encoding="utf-8")
    wp = WorkPaths(cfg)
    D0 = max(dims.values())
    mix = {e: rng.normal(size=(D0, d)) / np.sqrt(D0) + (np.eye(D0, d) if d <= D0 else 0) for e, d in dims.items()}
    # 维数越低的假编码器噪声越大（模拟“较弱”的编码器），便于检验比较流程
    noise = {e: 0.1 if d >= D0 else 0.8 for e, d in dims.items()}

    rows = []
    centers = ["internal"] * n_int + [["ext1", "ext2", "ext3"][i % 3] for i in range(n_ext)]
    counters = {}
    for center in centers:
        counters[center] = counters.get(center, 0) + 1
        sid = f"{CENTER_PREFIX[center]}-{counters[center]:04d}"
        lauren = int(rng.choice(3, p=[0.5, 0.35, 0.15]))
        msi = int(rng.rand() < 0.15)
        her2 = int(rng.rand() < 0.15)
        z = rng.normal()
        pop = "resected" if rng.rand() < 0.85 else "unresectable"
        base = 1 / 60.0 if pop == "resected" else 1 / 12.0
        t_ev = rng.exponential(1 / (base * np.exp(0.9 * z)))
        t_c = rng.uniform(6, 90)
        os_m, os_e = min(t_ev, t_c), int(t_ev <= t_c)
        n = rng.randint(t_lo, t_hi + 1)
        x = rng.normal(size=(n, D0)).astype(np.float32)
        sel = rng.rand(n) < 0.5
        x[sel, lauren] += 0.6 * k
        if msi:
            x[rng.rand(n) < 0.25, 3:6] += 0.9 * k
        x[:, 6:8] += 0.5 * k * z
        if her2:
            x[rng.rand(n) < 0.3, 8] += 1.0 * k
        coords = np.stack([np.arange(n) % 20 * 256, np.arange(n) // 20 * 256], 1)
        for e, M_ in mix.items():
            f = x @ M_ + noise[e] * rng.normal(size=(n, M_.shape[1]))
            write_features(wp.wsi_features(e, sid), f.astype(np.float32), coords,
                           {"encoder": e, "source": "fake"})
        def miss(v, p=0.1):
            return "" if rng.rand() < p else v
        r = {c: "" for c in MANIFEST_COLUMNS}
        r.update(
            study_id=sid, center=center, lauren=miss(["intestinal", "diffuse", "mixed"][lauren]),
            her2_ihc=miss("3+" if her2 else str(rng.choice(["0", "1+"]))), msi=miss("MSI-H" if msi else "MSS"),
            ebv=miss("pos" if rng.rand() < 0.08 else "neg"), pdl1_cps=miss(f"{rng.exponential(6):.0f}"),
            cldn182_pct=miss(f"{rng.uniform(0, 100):.0f}"), ki67_pct=miss(f"{rng.uniform(5, 90):.0f}"),
            label_source="biopsy", population=pop, os_months=f"{os_m:.1f}", os_event=str(os_e),
            neoadj="0", immuno="0",
        )
        if pop == "resected":
            r.update(dfs_months=f"{os_m * rng.uniform(0.5, 1):.1f}", dfs_event=str(int(os_e or rng.rand() < 0.2)))
        else:
            r.update(pfs_months=f"{os_m * rng.uniform(0.3, 0.8):.1f}", pfs_event=str(int(os_e or rng.rand() < 0.5)))
        rows.append(r)

    mpath = resolve_path(cfg, cfg.paths.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    res = validate_manifest(read_manifest_csv(mpath), cfg, check_paths=False)
    split = make_split(derive_labels(res.df, cfg), cfg)
    sp = wp.splits / f"split_seed{cfg.split.get('seed', cfg.get('seed'))}.csv"
    sp.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(sp, index=False)
    if logger:
        logger.info("假特征队列已生成：内部 %d 例、外部 %d 例，编码器 %s", n_int, n_ext, dims)
    return {"n_internal": n_int, "n_external": n_ext, "encoders": dims}


# ============================================================
# v2 / v3 用：多模态假特征队列（病理特征 + CT ROI 特征 + 影像组学表，埋入已知信号）
# ============================================================
def generate_mm_cohort(cfg: Config, logger=None, overwrite: bool = False) -> dict:
    """生成多模态假特征队列（不生成图像）：清单（含结构化 cT）、划分、病理特征、CT ROI 特征、影像组学特征表。

    埋入的信号（检验模型能学到、且两种模态互补）：
    - 预后：潜在风险 = 病理部分 z_p + CT 部分 z_c；病理特征只含 z_p，CT 特征只含 z_c；
    - cT 分期由 z_c 决定（CT 区域 ROI 中体现）；
    - Lauren：病理与 CT 肿瘤 ROI 中都有（CT 较弱）；MSI-H 只在病理中；
    - 维数越低的假 CT 编码器噪声越大（模拟"较弱"的编码器）；EBV、PD-L1、Claudin18.2 无信号；
    - CT 缺失约 12%，单个 ROI 缺失少量；影像组学表只覆盖有分割的病例（约 70%）。
    """
    import csv

    from .ct.features_io import write_ct_features
    from .labels import derive_labels
    from .manifest import read_manifest_csv, validate_manifest
    from .split import make_split
    from .wsi.features_io import write_features

    fc = cfg.fake_data.get("mm_cohort", {}) or {}
    n_int, n_ext = int(fc.get("n_internal", 300)), int(fc.get("n_external", 60))
    p_enc = dict(fc.get("path_encoders", {"uni2_h": 32}))
    c_enc = dict(fc.get("ct_encoders", {"fake_ct_a": 24, "fake_ct_b": 12}))
    s_lo, s_hi = (int(v) for v in fc.get("slices", [5, 9]))
    k = float(fc.get("signal", 1.0))
    rad_name = str(fc.get("radiomics_name", cfg.get_path("v2.radiomics.name", "original_bw25")))
    rng = np.random.RandomState(int(fc.get("seed", 13)))
    root = root_dir(cfg)
    out_dir = resolve_path(cfg, cfg.paths.output_dir).resolve()
    existing = [p for p in root.iterdir() if p.resolve() != out_dir] if root.exists() else []
    if existing and not (root / MARKER).exists():
        raise UserError(f"目标目录 {root} 已有内容且不是假数据目录，拒绝写入。", "请把 paths.root 指向新的空目录。")
    if existing and not overwrite:
        raise UserError(f"假数据目录 {root} 已存在。", "如需重新生成，加 --overwrite。")
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("由 gastric_mm.fake_data 生成的假数据目录（多模态特征队列）\n", encoding="utf-8")
    wp = WorkPaths(cfg)
    Dp = max(p_enc.values())
    Dc = max(c_enc.values())
    mix_p = {e: rng.normal(size=(Dp, d)) / np.sqrt(Dp) + (np.eye(Dp, d) if d <= Dp else 0) for e, d in p_enc.items()}
    mix_c = {e: rng.normal(size=(Dc, d)) / np.sqrt(Dc) + np.eye(Dc, d) for e, d in c_enc.items()}
    noise_c = {e: 0.2 if d >= Dc else 3.0 for e, d in c_enc.items()}
    t_groups = ["T1", "T2", "T3", "T4a"]
    rows, rad_rows = [], []
    centers = ["internal"] * n_int + [["ext1", "ext2", "ext3"][i % 3] for i in range(n_ext)]
    counters: Dict[str, int] = {}
    for center in centers:
        counters[center] = counters.get(center, 0) + 1
        sid = f"{CENTER_PREFIX[center]}-{counters[center]:04d}"
        lauren = int(rng.choice(3, p=[0.5, 0.35, 0.15]))
        msi = int(rng.rand() < 0.15)
        zp, zc = rng.normal(), rng.normal()
        ti = int(np.clip(np.digitize(zc + 0.5 * rng.normal(), [-1.0, -0.2, 0.7]), 0, 3))
        pop = "resected" if rng.rand() < 0.85 else "unresectable"
        base = 1 / 60.0 if pop == "resected" else 1 / 12.0
        t_ev = rng.exponential(1 / (base * np.exp(0.8 * zp + 0.8 * zc)))
        t_c = rng.uniform(6, 90)
        os_m, os_e = min(t_ev, t_c), int(t_ev <= t_c)
        # ---- 病理 ----
        n = rng.randint(30, 120)
        x = rng.normal(size=(n, Dp)).astype(np.float32)
        x[rng.rand(n) < 0.5, lauren] += 0.6 * k
        if msi:
            x[rng.rand(n) < 0.3, 3:6] += 0.9 * k
        x[:, 6:8] += 0.5 * k * zp
        coords = np.stack([np.arange(n) % 20 * 256, np.arange(n) // 20 * 256], 1)
        for e, Mx in mix_p.items():
            write_features(wp.wsi_features(e, sid), (x @ Mx + 0.1 * rng.normal(size=(n, Mx.shape[1]))).astype(np.float32),
                           coords, {"encoder": e, "source": "fake"})
        # ---- CT ----
        has_ct = rng.rand() > 0.12
        if has_ct:
            ns = rng.randint(s_lo, s_hi + 1)
            tum = rng.normal(size=(ns, Dc))
            tum[:, lauren] += 0.5 * k
            tum[:, 3:5] += 0.6 * k * zc
            reg = rng.normal(size=(ns + 4, Dc))
            reg[:, 5] += 0.6 * k * (ti - 1.5)
            reg[:, 6] += 0.4 * k * zc
            miss_t, miss_r = rng.rand() < 0.04, rng.rand() < 0.06
            for e, Mx in mix_c.items():
                rois = {}
                if not miss_t:
                    rois["tumor"] = tum @ Mx + noise_c[e] * rng.normal(size=(ns, Mx.shape[1]))
                if not miss_r or miss_t:
                    rois["regional"] = reg @ Mx + noise_c[e] * rng.normal(size=(ns + 4, Mx.shape[1]))
                write_ct_features(wp.ct_features(e, sid), rois, {"encoder": e, "kind": "2d", "source": "fake"})
        has_mask = has_ct and rng.rand() < 0.7
        if has_mask:
            f = rng.normal(size=30)
            f[0] += 0.8 * k * zc
            f[1] = f[0] * 0.98 + 0.05 * rng.normal()      # 与 f0 高度相关（检验去冗余）
            f[2] += 0.6 * k * (lauren == 1)
            f[3] += 0.5 * k * (ti - 1.5)
            r = {"study_id": sid, "mask_source": "manual" if rng.rand() < 0.3 else "auto_reviewed"}
            r.update({f"original_feat{j:02d}": float(v) for j, v in enumerate(f)})
            r["original_const"] = 1.0                      # 常数特征（检验剔除）
            rad_rows.append(r)

        def miss(v, p=0.1):
            return "" if rng.rand() < p else v
        r = {c: "" for c in MANIFEST_COLUMNS}
        r.update(study_id=sid, center=center, lauren=miss(["intestinal", "diffuse", "mixed"][lauren]),
                 her2_ihc=miss(str(rng.choice(["0", "1+", "3+"], p=[0.5, 0.35, 0.15]))),
                 msi=miss("MSI-H" if msi else "MSS"), ebv=miss("pos" if rng.rand() < 0.08 else "neg"),
                 pdl1_cps=miss(f"{rng.exponential(6):.0f}"), cldn182_pct=miss(f"{rng.uniform(0, 100):.0f}"),
                 label_source="biopsy", population=pop, os_months=f"{os_m:.1f}", os_event=str(os_e),
                 neoadj="0", immuno="0", ct_t=miss(t_groups[ti], 0.08), ct_stage_source="report",
                 ct_dir=("raw/ct/" + sid) if has_ct else "", tumor_visibility="clear")
        if pop == "resected":
            r.update(dfs_months=f"{os_m * rng.uniform(0.5, 1):.1f}", dfs_event=str(int(os_e or rng.rand() < 0.2)))
        else:
            r.update(pfs_months=f"{os_m * rng.uniform(0.3, 0.8):.1f}", pfs_event=str(int(os_e or rng.rand() < 0.5)))
        rows.append(r)
    mpath = resolve_path(cfg, cfg.paths.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if rad_rows:
        import pandas as pd
        rd = wp.ct_radiomics(rad_name)
        rd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rad_rows).to_csv(rd / "features.csv", index=False)
    res = validate_manifest(read_manifest_csv(mpath), cfg, check_paths=False)
    split = make_split(derive_labels(res.df, cfg), cfg)
    sp = wp.splits / f"split_seed{cfg.split.get('seed', cfg.get('seed'))}.csv"
    sp.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(sp, index=False)
    if logger:
        logger.info("多模态假特征队列已生成：内部 %d 例、外部 %d 例；病理 %s；CT %s；影像组学 %d 例",
                    n_int, n_ext, p_enc, c_enc, len(rad_rows))
    return {"n_internal": n_int, "n_external": n_ext, "n_radiomics": len(rad_rows)}


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "fake_data", args.run_name)
    if args.features_only:
        generate_feature_cohort(cfg, run.logger, overwrite=args.overwrite)
    elif args.mm_cohort:
        generate_mm_cohort(cfg, run.logger, overwrite=args.overwrite)
    else:
        generate(cfg, run.logger, overwrite=args.overwrite)
    return 0


def main(argv=None) -> int:
    p = base_parser("生成假的切片、CT、分割、TRIDENT 特征和病例清单（测试与演示用）。", cases=False, workers=False)
    p.add_argument("--overwrite", action="store_true", help="删除并重新生成已有的假数据目录")
    p.add_argument("--features-only", action="store_true",
                   help="只生成假特征队列（埋入已知信号，供 v1 模型测试与演示）")
    p.add_argument("--mm-cohort", action="store_true",
                   help="生成多模态假特征队列（病理 + CT 特征 + 影像组学表，供 v2/v3 模型测试与演示）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
