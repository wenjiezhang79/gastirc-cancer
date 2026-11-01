"""可插拔的病理切块编码器注册表。所有权重从配置中的**本地路径**加载，绝不联网。

已注册：
- ``random_small``   随机初始化的小 CNN（测试用，不需要权重）
- ``resnet50_trunc`` ImageNet ResNet50 截到 layer3 + 全局平均池化（1024 维），权重为 torchvision 的 .pth
- ``uni2_h``         UNI2-h（timm ViT-H/14，1536 维）
- ``virchow2``       Virchow2（timm ViT-H/14，CLS + patch 均值 = 2560 维）
- ``gigapath``       Prov-GigaPath 切块编码器（timm ViT-g/14，1536 维）
- ``trident``        通过服务器上已安装的 TRIDENT 调用其编码器（CONCH v1.5、MUSK、CHIEF 等），
                     配置 features.extractors.trident.model_name

注意：UNI2-h / Virchow2 / GigaPath 的结构参数按官方模型卡填写，**需要在服务器上用 3–5 例与
TRIDENT 提取的特征核对一次**（同一切块的特征应几乎相同）。7 个编码器的特征已用 TRIDENT 提取，
日常直接用 TRIDENT 导入（import_trident）即可；本模块用于新增病例或做染色增强时重新提取。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..config import Config, resolve_path
from ..utils.errors import UserError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Extractor:
    name: str
    model: object                     # torch.nn.Module
    feature_dim: int
    input_px: int
    transform: Callable               # PIL.Image -> torch.Tensor[3,H,W]
    forward: Callable                 # (model, batch) -> [B, D]
    weights_sha256: Optional[str] = None
    precision: Optional[str] = None


class ImageTransform:
    """缩放到 input_px → [0,1] → 标准化。不依赖 torchvision；可被 DataLoader 子进程序列化。"""

    def __init__(self, input_px: int, mean: Sequence[float], std: Sequence[float]):
        self.input_px = int(input_px)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img):
        import torch
        from PIL import Image

        if img.size != (self.input_px, self.input_px):
            img = img.resize((self.input_px, self.input_px), Image.BILINEAR)
        a = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(((a - self.mean) / self.std).astype(np.float32))


def make_transform(input_px: int, mean: Sequence[float], std: Sequence[float]) -> ImageTransform:
    return ImageTransform(input_px, mean, std)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_path(cfg: Config, name: str) -> Path:
    ec = cfg.get_path(f"features.extractors.{name}", {}) or {}
    wp = ec.get("weights_path")
    if not wp:
        raise UserError(f"编码器 {name} 没有配置权重路径",
                        f"请在配置文件 features.extractors.{name}.weights_path 中填写本地权重文件的路径。")
    p = resolve_path(cfg, wp)
    if not p.is_file():
        raise UserError(f"编码器 {name} 的权重文件不存在：{p}",
                        f"请把权重文件拷到该位置，或修改 features.extractors.{name}.weights_path。"
                        "（程序不会联网下载权重）")
    return p


def load_state_dict(path: Path) -> dict:
    import torch

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise UserError("读取 .safetensors 权重需要 safetensors 包", "请离线安装 safetensors，或改用 .bin/.pth 权重。")
        sd = load_file(str(path))
    else:
        try:
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # 旧版 torch 没有 weights_only
            sd = torch.load(str(path), map_location="cpu")
    for key in ("state_dict", "model", "teacher"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def _load_strict(model, sd: dict, name: str, ignore_prefixes: Sequence[str] = ()) -> None:
    sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in ignore_prefixes)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not any(k.startswith(p) for p in ignore_prefixes)]
    if missing or unexpected:
        raise UserError(
            f"{name} 权重与模型结构不匹配（缺少 {len(missing)} 个参数，多出 {len(unexpected)} 个），"
            f"例如：{(missing + unexpected)[:5]}",
            "请确认权重文件对应的是该编码器（版本是否一致），并把本条报错贴回。",
        )


# ---------------- 各编码器 ----------------
def _build_random_small(cfg: Config, ec: dict) -> Extractor:
    import torch
    from torch import nn

    dim = int(ec.get("feature_dim", 64))
    g = torch.Generator().manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, dim, 3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    px = int(ec.get("input_px", cfg.wsi.tiling.tile_px))
    return Extractor("random_small", model, dim, px, make_transform(px, IMAGENET_MEAN, IMAGENET_STD),
                     lambda m, x: m(x))


def _timm():
    try:
        import timm
        return timm
    except ImportError:
        raise UserError("该编码器需要 timm 包", "请离线安装 requirements.txt 中的 timm。") from None


def _build_resnet50_trunc(cfg: Config, ec: dict) -> Extractor:
    import torch

    timm = _timm()
    p = weights_path(cfg, "resnet50_trunc")
    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    _load_strict(model, load_state_dict(p), "resnet50_trunc", ignore_prefixes=("fc.",))

    def fwd(m, x):
        x = m.maxpool(m.act1(m.bn1(m.conv1(x))))
        x = m.layer3(m.layer2(m.layer1(x)))
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)

    return Extractor("resnet50_trunc", model, 1024, 224, make_transform(224, IMAGENET_MEAN, IMAGENET_STD), fwd,
                     sha256_file(p) if cfg.features.get("hash_weights", True) else None)


def _build_uni2_h(cfg: Config, ec: dict) -> Extractor:
    import torch

    timm = _timm()
    p = weights_path(cfg, "uni2_h")
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    _load_strict(model, load_state_dict(p), "uni2_h")
    return Extractor("uni2_h", model, 1536, 224, make_transform(224, IMAGENET_MEAN, IMAGENET_STD),
                     lambda m, x: m(x), sha256_file(p) if cfg.features.get("hash_weights", True) else None)


def _build_virchow2(cfg: Config, ec: dict) -> Extractor:
    import torch

    timm = _timm()
    p = weights_path(cfg, "virchow2")
    kw = dict(img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375, reg_tokens=4, global_pool="",
              dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, **kw)
    _load_strict(model, load_state_dict(p), "virchow2")

    def fwd(m, x):
        tok = m(x)                      # [B, 1 + 4 + 256, 1280]
        cls = tok[:, 0]
        patch = tok[:, 5:]
        return torch.cat([cls, patch.mean(1)], dim=-1)

    return Extractor("virchow2", model, 2560, 224, make_transform(224, IMAGENET_MEAN, IMAGENET_STD), fwd,
                     sha256_file(p) if cfg.features.get("hash_weights", True) else None)


def _build_gigapath(cfg: Config, ec: dict) -> Extractor:
    timm = _timm()
    p = weights_path(cfg, "gigapath")
    kw = dict(img_size=224, in_chans=3, patch_size=16, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5,
              mlp_ratio=5.33334, num_classes=0)
    model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **kw)
    _load_strict(model, load_state_dict(p), "gigapath")
    return Extractor("gigapath", model, 1536, 224, make_transform(224, IMAGENET_MEAN, IMAGENET_STD),
                     lambda m, x: m(x), sha256_file(p) if cfg.features.get("hash_weights", True) else None)


def _build_trident(cfg: Config, ec: dict) -> Extractor:
    import torch

    name = ec.get("model_name")
    if not name:
        raise UserError("使用 TRIDENT 编码器时需要填写 features.extractors.trident.model_name",
                        "例如 conch_v15 / musk / ctranspath（与 TRIDENT 中的编码器名一致）。")
    try:
        from trident.patch_encoder_models import encoder_factory
    except ImportError:
        raise UserError("当前 Python 环境中没有安装 TRIDENT", "请在安装了 TRIDENT 的环境中运行本命令。") from None
    enc = encoder_factory(name, **dict(ec.get("kwargs", {}) or {}))
    tf = enc.eval_transforms
    model = enc
    with torch.no_grad():
        from PIL import Image
        dummy = tf(Image.new("RGB", (224, 224), (255, 255, 255))).unsqueeze(0)
        dim = int(model(dummy).shape[-1])
    px = int(dummy.shape[-1])
    return Extractor(f"trident_{name}", model, dim, px, tf, lambda m, x: m(x),
                     precision=str(getattr(enc, "precision", "")))


REGISTRY: Dict[str, Callable[[Config, dict], Extractor]] = {
    "random_small": _build_random_small,
    "resnet50_trunc": _build_resnet50_trunc,
    "uni2_h": _build_uni2_h,
    "virchow2": _build_virchow2,
    "gigapath": _build_gigapath,
    "trident": _build_trident,
}


def build_extractor(name: str, cfg: Config) -> Extractor:
    if name not in REGISTRY:
        raise UserError(f"未知的编码器：{name}", f"可用：{', '.join(REGISTRY)}")
    ec = dict(cfg.get_path(f"features.extractors.{name}", {}) or {})
    ext = REGISTRY[name](cfg, ec)
    ext.model.eval()
    for p in ext.model.parameters():
        p.requires_grad_(False)
    return ext


def output_name(ext: Extractor) -> str:
    """特征输出目录名：与 TRIDENT 导入后的编码器名一致（trident_conch_v15 → conch_v15 等）。"""
    n = ext.name
    if n.startswith("trident_"):
        n = n[len("trident_"):]
    alias = {"uni_v2": "uni2_h", "ctranspath": "chief_ctranspath", "chief": "chief_ctranspath",
             "resnet50": "resnet50_trunc"}
    return alias.get(n, n)
