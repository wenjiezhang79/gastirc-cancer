"""病理多任务模型 = 多示例聚合器 + 多任务输出头。

模型文件（model.pt）中保存结构参数、任务定义、编码器、生存时间段等，加载时不依赖原配置，
v3 融合可直接加载其聚合器作为病理分支的初始化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from ..training.task_spec import TaskSpec
from .heads import MultiTaskHeads
from .mil import build_aggregator


class PathModel(nn.Module):
    def __init__(self, in_dim: int, model_cfg: dict, spec: TaskSpec, n_bins: int):
        super().__init__()
        self.in_dim = in_dim
        self.model_cfg = dict(model_cfg)
        self.spec = spec
        self.n_bins = n_bins
        self.aggregator = build_aggregator(self.model_cfg.get("aggregator", "abmil"), in_dim, self.model_cfg)
        self.heads = MultiTaskHeads(self.aggregator.out_dim, self.aggregator.n_tokens, spec, n_bins,
                                    float(self.model_cfg.get("dropout", 0.25)))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, attn = self.aggregator(feats, mask)
        out = self.heads(tokens)
        out["tokens"] = tokens
        out["attention"] = attn
        return out


def save_model(model: PathModel, path: Path, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "format": "gastric_mm.path_model.v1",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "in_dim": model.in_dim,
        "model_cfg": model.model_cfg,
        "spec": model.spec.to_dict(),
        "n_bins": model.n_bins,
        "extra": extra or {},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_model(path: Path, map_location="cpu"):
    """返回 (模型, extra)。extra 中有编码器、生存时间段、正样本权重等。"""
    try:
        obj = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:  # 旧版 torch
        obj = torch.load(str(path), map_location=map_location)
    if obj.get("format") != "gastric_mm.path_model.v1":
        raise ValueError(f"{path.name} 不是本项目 v1 病理模型文件")
    model = PathModel(obj["in_dim"], obj["model_cfg"], TaskSpec.from_dict(obj["spec"]), obj["n_bins"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj.get("extra", {})
