"""CT 编码器注册表（冻结提特征）。所有权重从配置中的**本地路径**加载，绝不联网。

两类编码器：
- **2.5D**（kind="2d"）：ROI 内中心层 ±k 层，每层取相邻 3 层（或同一层复制 3 次）作为 RGB，缩放到 224，
  逐层用 2D 编码器提特征 → [n_层, D]。要求所用裁剪规格 normalize=minmax（已按窗宽缩放到 [0,1]）。
  - ``random_small_2d``：随机小网络（测试用）
  - ``imagenet_r50``：ImageNet ResNet50（torchvision 权重 .pth，可与病理的 resnet50_trunc 共用文件），2048 维
  - ``radimagenet_r50``：RadImageNet ResNet50（PyTorch 权重），2048 维；输入标准化见配置 mean/std
  - ``dinov2_vitb14``：DINOv2 ViT-B/14（官方 dinov2_vitb14_pretrain.pth 或 timm 权重），CLS 768 维
- **3D**（kind="3d"）：整块 ROI → [1, D]。要求裁剪规格 normalize=none（保留 HU，编码器内部按官方方式缩放）。
  - ``random_small_3d``：随机小网络（测试用，可用 minmax 规格）
  - ``ctfm``：CT-FM（MONAI SegResEncoder，SPL 方向，HU −1024–2048 → [0,1]），512 维
  - ``suprem``：SuPreM（MONAI SwinUNETR 的 swinViT 编码器，RAS 方向，HU −175–250 → [0,1]），768 维
  - ``merlin``：Merlin 图像编码器（需安装官方 merlin 包；RAS 方向，HU −1000–1000 → [0,1]）

**注意：** 3D 基础模型与 RadImageNet 的结构、权重键名和预处理按公开资料实现，**必须在服务器上用 3–5 例
与官方示例代码的输出核对一次**（同一输入的特征应几乎相同），核对通过后再全量提取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..config import Config, resolve_path
from ..utils.errors import UserError
from ..wsi.extractors import IMAGENET_MEAN, IMAGENET_STD, _load_strict, _timm, load_state_dict, sha256_file


@dataclass
class CTEncoder:
    name: str
    kind: str                          # 2d / 3d
    model: object                      # torch.nn.Module
    feature_dim: int
    forward: Callable                  # 2d: [B,3,H,W] → [B,D]；3d: [B,1,Z,Y,X]（numpy 顺序 z,y,x）→ [B,D]
    rois: Dict[str, str]               # ROI 角色 → 裁剪规格名
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD
    prepare3d: Optional[Callable] = None   # 3D：HU 数组 (z,y,x) → numpy 数组（编码器输入顺序）
    require_normalize: Optional[str] = None  # 对裁剪规格 normalize 的要求
    require_window: Optional[Sequence[float]] = None  # 对窗宽（截断范围 [lo, hi] HU）的要求
    weights_sha256: Optional[str] = None
    notes: Dict[str, str] = field(default_factory=dict)


def _wpath(cfg: Config, name: str, ec: dict) -> Path:
    p = resolve_path(cfg, ec.get("weights_path"))
    if p is None:
        raise UserError(f"CT 编码器 {name} 没有配置权重路径",
                        f"请在配置文件 v2.features.encoders.{name}.weights_path 中填写本地权重文件的路径。")
    if not p.exists():
        raise UserError(f"CT 编码器 {name} 的权重文件不存在",
                        f"请确认 v2.features.encoders.{name}.weights_path 填写正确（程序不会联网下载权重）。")
    return p


def _hash(cfg: Config, p: Path) -> Optional[str]:
    return sha256_file(p) if cfg.v2.features.get("hash_weights", True) else None


def _rois(ec: dict, default: Dict[str, str]) -> Dict[str, str]:
    """配置中某个角色写 null 表示不提取该 ROI（配置继承时用来去掉默认角色）。"""
    r = {k: v for k, v in dict(ec.get("rois") or default).items() if v}
    if not r:
        raise UserError("CT 编码器没有配置 rois（ROI 角色 → 裁剪规格）")
    return r


def _load(p: Path) -> dict:
    """读取权重，并剥掉常见的外层包装（{"net": ...}、{"state_dict": ...} 等）。"""
    sd = load_state_dict(p)
    while isinstance(sd, dict) and len(sd) <= 3:
        inner = next((sd[k] for k in ("net", "state_dict", "model", "teacher", "student") if isinstance(sd.get(k), dict)),
                     None)
        if inner is None:
            break
        sd = inner
    return sd


def _strip(sd: dict, prefixes: Sequence[str]) -> dict:
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def _scale(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def zyx_to_ras_xyz(arr: np.ndarray) -> np.ndarray:
    """numpy (z, y, x)，轴方向 S/P/L（本流程 LPS）→ (x, y, z)，轴方向 R/A/S（MONAI 读入 RAS 后的顺序）。"""
    return np.ascontiguousarray(arr.transpose(2, 1, 0)[::-1, ::-1, :])


# ---------------- 2.5D ----------------
def _build_random_small_2d(cfg: Config, ec: dict) -> CTEncoder:
    import torch
    from torch import nn

    dim = int(ec.get("feature_dim", 32))
    g = torch.Generator().manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.ReLU(), nn.Conv2d(16, dim, 3, stride=2, padding=1),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return CTEncoder("random_small_2d", "2d", m, dim, lambda mm, x: mm(x),
                     _rois(ec, {"tumor": "tumor_96mm", "regional": "regional_160mm"}), require_normalize="minmax")


def _resnet50(cfg: Config, name: str, ec: dict, remap: bool) -> CTEncoder:
    timm = _timm()
    p = _wpath(cfg, name, ec)
    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    sd = _strip(_load(p), ("module.", "model.", "backbone.", "encoder."))
    if remap and any(k.split(".")[0].isdigit() for k in sd):
        # nn.Sequential(*list(resnet.children())[:-1]) 形式保存的权重：0=conv1, 1=bn1, 4–7=layer1–4
        idx = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
        sd = {(idx[k.split(".", 1)[0]] + "." + k.split(".", 1)[1]) if k.split(".", 1)[0] in idx else k: v
              for k, v in sd.items()}
    _load_strict(model, sd, name, ignore_prefixes=("fc.",))
    return CTEncoder(name, "2d", model, 2048, lambda m, x: m(x),
                     _rois(ec, {"tumor": "tumor_96mm", "regional": "regional_160mm"}),
                     mean=tuple(ec.get("mean", IMAGENET_MEAN)), std=tuple(ec.get("std", IMAGENET_STD)),
                     require_normalize="minmax", weights_sha256=_hash(cfg, p))


def _build_imagenet_r50(cfg: Config, ec: dict) -> CTEncoder:
    return _resnet50(cfg, "imagenet_r50", ec, remap=False)


def _build_radimagenet_r50(cfg: Config, ec: dict) -> CTEncoder:
    e = _resnet50(cfg, "radimagenet_r50", ec, remap=True)
    e.notes["verify"] = "RadImageNet 的输入标准化（mean/std）请按官方说明核对"
    return e


def _build_dinov2_vitb14(cfg: Config, ec: dict) -> CTEncoder:
    timm = _timm()
    p = _wpath(cfg, "dinov2_vitb14", ec)
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=518,
                              dynamic_img_size=True)
    sd = _strip(_load(p), ("module.", "backbone.", "model."))
    if any(k.startswith("embeddings.") for k in sd):
        raise UserError("dinov2_vitb14 的权重是 Hugging Face transformers 格式，本程序不支持",
                        "请改用官方 dinov2_vitb14_pretrain.pth，或 timm 的 vit_base_patch14_dinov2 权重文件。")
    sd = {k: v for k, v in sd.items() if k != "mask_token"}
    _load_strict(model, sd, "dinov2_vitb14")
    return CTEncoder("dinov2_vitb14", "2d", model, 768, lambda m, x: m(x),
                     _rois(ec, {"tumor": "tumor_96mm", "regional": "regional_160mm"}), require_normalize="minmax",
                     weights_sha256=_hash(cfg, p))


# ---------------- 3D ----------------
def _build_random_small_3d(cfg: Config, ec: dict) -> CTEncoder:
    import torch
    from torch import nn

    dim = int(ec.get("feature_dim", 32))
    g = torch.Generator().manual_seed(0)
    m = nn.Sequential(nn.Conv3d(1, 8, 3, stride=(1, 2, 2), padding=1), nn.ReLU(),
                      nn.Conv3d(8, dim, 3, stride=2, padding=1), nn.AdaptiveAvgPool3d(1), nn.Flatten())
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return CTEncoder("random_small_3d", "3d", m, dim, lambda mm, x: mm(x),
                     _rois(ec, {"tumor": "tumor_96mm", "regional": "regional_160mm"}),
                     prepare3d=lambda a: a.astype(np.float32))


def _monai():
    try:
        import monai  # noqa: F401
        return monai
    except ImportError:
        raise UserError("该 CT 编码器需要 MONAI", "请离线安装 monai（纯 Python 包，见 docs/v2_offline_setup.md）。") from None


def _pad_to_multiple(x, k: int):
    import torch.nn.functional as F

    pads = []
    for s in reversed(x.shape[2:]):
        extra = (-s) % k
        pads += [extra // 2, extra - extra // 2]
    return F.pad(x, pads, value=0.0) if any(pads) else x


def _build_ctfm(cfg: Config, ec: dict) -> CTEncoder:
    import torch

    _monai()
    from monai.networks.nets.segresnet_ds import SegResEncoder

    p = _wpath(cfg, "ctfm", ec)
    model = SegResEncoder(spatial_dims=3, in_channels=1, init_filters=int(ec.get("init_filters", 32)),
                          blocks_down=tuple(ec.get("blocks_down", (1, 2, 2, 4, 4))))
    sd = _strip(_load(p), ("module.", "model.", "encoder.", "trunk."))
    _load_strict(model, sd, "ctfm")

    def fwd(m, x):
        feats = m(_pad_to_multiple(x, 16))
        return torch.flatten(torch.nn.functional.adaptive_avg_pool3d(feats[-1], 1), 1)

    # 本流程数组 (z, y, x) 的轴方向即 S/P/L，与 CT-FM 的 SPL 方向一致，无需转置
    return CTEncoder("ctfm", "3d", model, 512, fwd,
                     _rois(ec, {"tumor": "hu_tumor_96mm", "regional": "hu_regional_160mm"}),
                     prepare3d=lambda a: _scale(a, -1024.0, 2048.0), require_normalize="none",
                     require_window=(-1024.0, 2048.0), weights_sha256=_hash(cfg, p),
                     notes={"verify": "CT-FM 的预处理与输出层请按官方示例核对"})


def _build_suprem(cfg: Config, ec: dict) -> CTEncoder:
    import torch

    _monai()
    from monai.networks.nets import SwinUNETR

    arch = str(ec.get("arch", "swinunetr"))
    if arch != "swinunetr":
        raise UserError(f"SuPreM 的 {arch} 结构暂不支持", "请使用 swinunetr 权重（supervised_suprem_swinunetr_2100.pth）。")
    p = _wpath(cfg, "suprem", ec)
    kw = dict(in_channels=1, out_channels=int(ec.get("out_channels", 32)), feature_size=48, use_checkpoint=False)
    try:
        model = SwinUNETR(**kw)
    except TypeError:   # 旧版 MONAI 需要 img_size
        model = SwinUNETR(img_size=(96, 96, 96), **kw)
    sd = _strip(_load(p), ("module.", "backbone.", "net.", "model."))
    sub = {k[len("swinViT."):]: v for k, v in sd.items() if k.startswith("swinViT.")}
    if not sub:
        raise UserError("SuPreM 权重中没有找到 swinViT 编码器参数", "请确认权重文件是 SuPreM 的 SwinUNETR 版本。")
    _load_strict(model.swinViT, sub, "suprem")
    enc = model.swinViT

    def fwd(m, x):
        hs = m(x, True)
        return torch.flatten(torch.nn.functional.adaptive_avg_pool3d(hs[-1], 1), 1)

    return CTEncoder("suprem", "3d", enc, 768, fwd, _rois(ec, {"tumor": "suprem_144mm"}),
                     prepare3d=lambda a: zyx_to_ras_xyz(_scale(a, -175.0, 250.0)), require_normalize="none",
                     require_window=(-175.0, 250.0), weights_sha256=_hash(cfg, p),
                     notes={"verify": "SuPreM 的方向与预处理请按官方示例核对"})


def _build_merlin(cfg: Config, ec: dict) -> CTEncoder:
    import torch

    p = _wpath(cfg, "merlin", ec)
    try:
        from merlin.models.build import ImageEncoder
    except ImportError:
        raise UserError("Merlin 编码器需要官方 merlin 包（merlin-vlm）",
                        "请在联网电脑上下载 merlin-vlm 及其依赖的 wheel 后离线安装（见 docs/v2_offline_setup.md）。") from None
    model = ImageEncoder()
    sd = _strip(_load(p), ("module.", "model."))
    pref = next((q for q in ("encode_image.", "image_encoder.") if any(k.startswith(q) for k in sd)), None)
    if pref is None:
        raise UserError("Merlin 权重中没有找到图像编码器参数（encode_image.*）", "请确认权重文件是 Merlin 官方 checkpoint。")
    _load_strict(model, {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}, "merlin")
    out_idx = int(ec.get("output_index", 0))

    def fwd(m, x):
        out = m(x)
        if isinstance(out, (tuple, list)):
            out = out[out_idx]
        return torch.flatten(out, 1)

    dim = int(ec.get("feature_dim", 2048))
    try:
        with torch.no_grad():
            dim = int(fwd(model.eval(), torch.zeros(1, 1, 64, 64, 32)).shape[-1])
    except Exception:
        pass
    return CTEncoder("merlin", "3d", model, dim, fwd, _rois(ec, {"regional": "merlin_abdomen"}),
                     prepare3d=lambda a: zyx_to_ras_xyz(_scale(a, -1000.0, 1000.0)), require_normalize="none",
                     require_window=(-1000.0, 1000.0), weights_sha256=_hash(cfg, p),
                     notes={"verify": "Merlin 的图像编码器输出与预处理请按官方 demo 核对"})


REGISTRY: Dict[str, Callable[[Config, dict], CTEncoder]] = {
    "random_small_2d": _build_random_small_2d,
    "imagenet_r50": _build_imagenet_r50,
    "radimagenet_r50": _build_radimagenet_r50,
    "dinov2_vitb14": _build_dinov2_vitb14,
    "random_small_3d": _build_random_small_3d,
    "ctfm": _build_ctfm,
    "suprem": _build_suprem,
    "merlin": _build_merlin,
}


def build_ct_encoder(name: str, cfg: Config) -> CTEncoder:
    if name not in REGISTRY:
        raise UserError(f"未知的 CT 编码器：{name}", f"可用：{', '.join(REGISTRY)}")
    ec = dict(cfg.get_path(f"v2.features.encoders.{name}", {}) or {})
    enc = REGISTRY[name](cfg, ec)
    enc.model.eval()
    for p in enc.model.parameters():
        p.requires_grad_(False)
    return enc


def check_specs(enc: CTEncoder, specs: Dict[str, dict]) -> None:
    """检查编码器使用的裁剪规格是否满足其预处理要求。"""
    for role, s in specs.items():
        norm = s.get("normalize", "minmax")
        if enc.require_normalize and norm != enc.require_normalize:
            raise UserError(f"编码器 {enc.name} 的 ROI {role} 使用的裁剪规格 {s['name']} 的 normalize={norm}，"
                            f"应为 {enc.require_normalize}",
                            "2.5D 编码器用 minmax 规格（如 tumor_96mm）；3D 基础模型用 normalize=none 的 HU 规格。")
        if enc.require_window:
            lo = float(s["window"]["level"]) - float(s["window"]["width"]) / 2
            hi = float(s["window"]["level"]) + float(s["window"]["width"]) / 2
            if lo > enc.require_window[0] + 1 or hi < enc.require_window[1] - 1:
                raise UserError(f"裁剪规格 {s['name']} 的窗宽截断范围 [{lo:g}, {hi:g}] HU 小于编码器 {enc.name} "
                                f"需要的 [{enc.require_window[0]:g}, {enc.require_window[1]:g}] HU",
                                "请加宽该规格的 window（截断范围应覆盖编码器需要的 HU 范围）。")
