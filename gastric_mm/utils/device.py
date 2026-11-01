"""GPU / CPU 设备选择。"""

from __future__ import annotations

from .errors import UserError


def resolve_device(gpu):
    """gpu 为 None 时使用 CPU；否则检查该 GPU 是否存在。"""
    import torch

    if gpu is None:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise UserError(f"指定了 --gpu {gpu}，但当前环境检测不到可用的 GPU",
                        "请运行 nvidia-smi 确认 GPU 状态，或去掉 --gpu 改用 CPU。")
    if gpu >= torch.cuda.device_count():
        raise UserError(f"没有第 {gpu} 张 GPU（共 {torch.cuda.device_count()} 张）", "请用 --gpu 0 或 --gpu 1。")
    return torch.device(f"cuda:{gpu}")
