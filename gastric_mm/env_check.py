"""离线环境检查：汇总操作系统、Python、PyTorch/CUDA、GPU、各依赖包版本，以及权重文件是否就绪。

命令行：
    python -m gastric_mm.env_check --config configs/server.yaml [--hash] [--gpu-test]

输出（运行目录下）：env_report.md（贴回给开发者，用于固定依赖版本）、env_report.json、packages_all.txt。
不联网、不读取任何患者数据。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config, resolve_path
from .utils.cli import enforce_offline, run_main
from .utils.runio import Run

# (导入名, 发行包名, 用途)
CHECK_MODULES = [
    ("numpy", "numpy", "必需"), ("scipy", "scipy", "必需"), ("pandas", "pandas", "必需"),
    ("yaml", "PyYAML", "必需"), ("pydicom", "pydicom", "必需：DICOM"), ("SimpleITK", "SimpleITK", "必需：CT"),
    ("h5py", "h5py", "必需：特征"), ("tifffile", "tifffile", "必需：切片备用读取"), ("PIL", "Pillow", "必需"),
    ("matplotlib", "matplotlib", "必需：质控图"), ("openslide", "openslide-python", "必需：切片读取"),
    ("torch", "torch", "必需：特征提取/训练"), ("timm", "timm", "病理编码器"),
    ("imagecodecs", "imagecodecs", "可选：tifffile 读 JPEG 压缩切片"),
    ("trident", "trident", "可选：调用 TRIDENT 编码器"), ("monai", "monai", "v2：CT 编码器"),
    ("totalsegmentator", "TotalSegmentator", "v2：胃分割（建议单独环境）"), ("nnunetv2", "nnunetv2", "v2：肿瘤分割"),
    ("radiomics", "pyradiomics", "v2：影像组学"), ("torchvision", "torchvision", "可选"),
    ("transformers", "transformers", "可选：部分编码器依赖"), ("huggingface_hub", "huggingface_hub", "可选"),
    ("torchscale", "torchscale", "可选：MUSK 依赖"), ("musk", "musk", "可选：MUSK"),
    ("conch", "conch", "可选：CONCH"), ("sklearn", "scikit-learn", "v1：评估"), ("lifelines", "lifelines", "v4：生存统计"),
    ("einops", "einops", "v2：SuPreM（SwinUNETR）"), ("safetensors", "safetensors", "v2：CT-FM / DINOv2 的 .safetensors 权重"),
    ("merlin", "merlin-vlm", "v2：Merlin（可选）"),
]


def _version(dist: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(dist)
    except Exception:
        return None


def check_modules() -> List[dict]:
    rows = []
    for mod, dist, use in CHECK_MODULES:
        r = {"module": mod, "dist": dist, "use": use, "version": _version(dist), "import": ""}
        try:
            m = importlib.import_module(mod)
            r["import"] = "OK"
            if r["version"] is None:
                r["version"] = getattr(m, "__version__", "?")
        except Exception as e:
            r["import"] = f"失败：{type(e).__name__}: {str(e)[:120]}"
        rows.append(r)
    return rows


def check_torch(gpu_test: bool) -> dict:
    info: Dict[str, object] = {}
    try:
        import torch
    except Exception as e:
        return {"error": f"无法导入 torch：{e}"}
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn"] = None
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            g = {"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1024 ** 3, 1),
                 "capability": f"{p.major}.{p.minor}"}
            try:
                g["bf16"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                g["bf16"] = None
            if gpu_test:
                try:
                    x = torch.randn(1024, 1024, device=f"cuda:{i}")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = (x @ x).float().sum().item()
                    g["matmul_test"] = "OK" if y == y else "NaN"
                except Exception as e:
                    g["matmul_test"] = f"失败：{e}"
            gpus.append(g)
    info["gpus"] = gpus
    return info


def check_openslide() -> dict:
    try:
        import openslide
        return {"openslide_python": openslide.__version__, "openslide_library": openslide.__library_version__}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_weights(cfg, do_hash: bool) -> List[dict]:
    rows = []
    exts = dict(cfg.get_path("features.extractors", {}) or {})
    for name, ec in (cfg.get_path("v2.features.encoders", {}) or {}).items():
        exts[f"CT:{name}"] = ec
    for name, ec in exts.items():
        if not isinstance(ec, dict) or "weights_path" not in ec:
            continue
        wp = ec.get("weights_path")
        r = {"extractor": name, "configured": bool(wp), "exists": None, "size_mb": None, "sha256_prefix": None}
        if wp:
            p = resolve_path(cfg, wp)
            r["exists"] = p.is_file()
            if p.is_file():
                r["size_mb"] = round(p.stat().st_size / 1024 ** 2, 1)
                if do_hash:
                    r["sha256_prefix"] = sha256_file(p)[:16]
        rows.append(r)
    return rows


def system_info() -> dict:
    mem = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem = round(int(line.split()[1]) / 1024 ** 2, 1)
    except Exception:
        pass
    os_release = None
    try:
        with open("/etc/os-release") as f:
            d = dict(l.strip().split("=", 1) for l in f if "=" in l)
            os_release = d.get("PRETTY_NAME", "").strip('"')
    except Exception:
        pass
    return {"platform": platform.platform(), "os": os_release, "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "memory_gb": mem}


def all_packages() -> List[str]:
    try:
        from importlib import metadata
        return sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]},
                      key=str.lower)
    except Exception:
        return []


def build_report(sysi, torchi, osl, mods, weights) -> str:
    L = ["# 环境检查报告", "", "## 系统", ""]
    for k, v in sysi.items():
        L.append(f"- {k}: {v}")
    L += ["", "## PyTorch / GPU", ""]
    for k, v in torchi.items():
        if k != "gpus":
            L.append(f"- {k}: {v}")
    for g in torchi.get("gpus", []):
        L.append(f"- GPU {g['index']}: {g['name']}，{g['memory_gb']} GB，算力 {g['capability']}，bf16={g.get('bf16')}"
                 + (f"，测试={g['matmul_test']}" if "matmul_test" in g else ""))
    L += ["", "## OpenSlide", ""] + [f"- {k}: {v}" for k, v in osl.items()]
    L += ["", "## 依赖包", "", "| 模块 | 包名 | 版本 | 导入 | 用途 |", "|---|---|---|---|---|"]
    for r in mods:
        L.append(f"| {r['module']} | {r['dist']} | {r['version'] or '未安装'} | {r['import']} | {r['use']} |")
    L += ["", "## 编码器权重", ""]
    if weights:
        L += ["| 编码器 | 已配置 | 文件存在 | 大小 MB | SHA256 前 16 位 |", "|---|---|---|---|---|"]
        for r in weights:
            L.append(f"| {r['extractor']} | {r['configured']} | {r['exists']} | {r['size_mb']} | {r['sha256_prefix'] or ''} |")
    else:
        L.append("配置中没有填写权重路径。")
    L += ["", "完整包列表见 packages_all.txt。", ""]
    return "\n".join(L)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "env_check", args.run_name)
    log = run.logger
    log.info("检查系统与依赖（不联网）……")
    sysi = system_info()
    torchi = check_torch(args.gpu_test)
    osl = check_openslide()
    mods = check_modules()
    weights = check_weights(cfg, args.hash)
    run.write_json({"system": sysi, "torch": torchi, "openslide": osl, "modules": mods, "weights": weights},
                   "env_report.json")
    run.write_text("\n".join(all_packages()) + "\n", "packages_all.txt")
    run.write_text(build_report(sysi, torchi, osl, mods, weights), "env_report.md")
    missing = [r["module"] for r in mods if r["use"].startswith("必需") and r["import"] != "OK"]
    if missing:
        log.warning("缺少必需的包：%s。安装方法见 README 的“离线安装”一节。", ", ".join(missing))
    for key, what in (("v2.external.totalseg.home_dir", "TotalSegmentator 权重目录"),
                      ("v2.external.nnunet.bin_dir", "nnU-Net 环境 bin 目录")):
        v = cfg.get_path(key)
        if v:
            log.info("%s（%s）：%s", what, key, "存在" if resolve_path(cfg, v).is_dir() else "不存在")
    log.info("环境检查完成，请把 %s 的内容贴回。", run.path("env_report.md"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="离线环境检查（不读取患者数据）。")
    p.add_argument("--config", default="configs/default.yaml", help="配置文件（用于检查权重路径），默认 configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="键=值", help="临时覆盖配置项")
    p.add_argument("--run-name", default=None)
    p.add_argument("--hash", action="store_true", help="计算权重文件 SHA256（大文件需要几十秒）")
    p.add_argument("--gpu-test", action="store_true", help="在每张 GPU 上做一次 bf16 矩阵乘法测试")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
