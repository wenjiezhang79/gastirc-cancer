"""工作目录（处理结果）的统一布局。所有处理结果按 study_id 命名。

<work_dir>/
├── splits/                         数据划分文件
├── ct/native/<id>.nii.gz           原始分辨率 NIfTI（医生在此标注）+ <id>.json 序列信息
├── ct/native_thin/<id>.nii.gz      可选：同一次采集的薄层重建（用于标注）
├── ct/resampled/<id>.nii.gz        整卷重采样（int16 HU）
├── ct/resampled_mask/<id>.nii.gz   整卷重采样后的肿瘤分割
├── ct/stomach/<id>.nii.gz          胃分割（v2 由 TotalSegmentator 生成）
├── ct/crops/<规格>/<id>_img.nii.gz 裁剪结果（及 _mask）
├── ct/qc/<规格>/<id>.png           单例质控图
├── wsi/tiles/<切块规格>/<id>.h5    切块坐标
├── wsi/qc/<切块规格>/<id>.png      组织检测与切块质控图
└── wsi/features/<编码器>/<id>.h5   病理特征
"""

from __future__ import annotations

from pathlib import Path

from .config import Config, resolve_path


class WorkPaths:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.work_dir = resolve_path(cfg, cfg.paths.work_dir)

    # ---- 通用 ----
    @property
    def splits(self) -> Path:
        return self.work_dir / "splits"

    # ---- CT ----
    @property
    def ct(self) -> Path:
        return self.work_dir / "ct"

    def ct_native(self, sid: str) -> Path:
        return self.ct / "native" / f"{sid}.nii.gz"

    def ct_native_meta(self, sid: str) -> Path:
        return self.ct / "native" / f"{sid}.json"

    def ct_native_thin(self, sid: str) -> Path:
        return self.ct / "native_thin" / f"{sid}.nii.gz"

    def ct_native_thin_meta(self, sid: str) -> Path:
        return self.ct / "native_thin" / f"{sid}.json"

    def ct_resampled(self, sid: str) -> Path:
        return self.ct / "resampled" / f"{sid}.nii.gz"

    def ct_resampled_mask(self, sid: str) -> Path:
        return self.ct / "resampled_mask" / f"{sid}.nii.gz"

    def ct_stomach(self, sid: str) -> Path:
        d = self.cfg.get_path("ct.stomach_mask_dir")
        base = resolve_path(self.cfg, d) if d else self.ct / "stomach"
        return base / f"{sid}.nii.gz"

    def ct_crop(self, spec: str, sid: str, kind: str = "img") -> Path:
        return self.ct / "crops" / spec / f"{sid}_{kind}.nii.gz"

    def ct_qc(self, spec: str, sid: str) -> Path:
        return self.ct / "qc" / spec / f"{sid}.png"

    # ---- WSI ----
    @property
    def wsi(self) -> Path:
        return self.work_dir / "wsi"

    def wsi_tiles(self, spec: str, sid: str) -> Path:
        return self.wsi / "tiles" / spec / f"{sid}.h5"

    def wsi_qc(self, spec: str, sid: str) -> Path:
        return self.wsi / "qc" / spec / f"{sid}.png"

    def wsi_features(self, encoder: str, sid: str) -> Path:
        return self.wsi / "features" / encoder / f"{sid}.h5"
