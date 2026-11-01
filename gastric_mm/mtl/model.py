"""token 多任务模型：每个角色（ROI / 模态）一个分支 → 1 个 token；token → 多任务输出头。

- 分支：门控注意力 MIL（v1 的 GatedABMIL，K=1，用于 2.5D 多层特征；3D 编码器只有 1 行时退化为投影）或均值池化；
- 某例缺少某个角色时，用该角色的**可学习占位 token** 代替（训练时 roi_dropout 随机丢弃一个角色，学会处理缺失）；
- 输出头：v1 的 MultiTaskHeads（多分类 / 二分类 / 回归 / 生存风险评分 + 分人群基线），token 拼接后进入。

模型文件 model.pt 中保存结构参数、角色与维数、任务定义、生存时间段，加载时不依赖原配置；
v3 融合可直接加载其 CT 分支作为初始化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from ..models.heads import MultiTaskHeads
from ..models.mil import build_aggregator
from ..training.task_spec import TaskSpec

FORMAT = "gastric_mm.token_model.v2"


class TokenMTLModel(nn.Module):
    def __init__(self, in_dims: Dict[str, int], model_cfg: dict, spec: TaskSpec, n_bins: int):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.roles = list(in_dims)
        self.model_cfg = dict(model_cfg)
        self.spec = spec
        self.n_bins = n_bins
        agg = str(self.model_cfg.get("aggregator", "abmil"))
        if agg not in ("abmil", "meanpool"):
            raise ValueError(f"CT 分支的聚合方式应为 abmil 或 meanpool，收到 {agg}")
        mc = dict(self.model_cfg, n_branches=1)
        self.branches = nn.ModuleDict({r: build_aggregator(agg, d, mc) for r, d in self.in_dims.items()})
        h = int(self.model_cfg.get("hidden_dim", 256))
        self.hidden = h
        self.missing = nn.ParameterDict({r: nn.Parameter(torch.zeros(1, 1, h)) for r in self.roles})
        for p in self.missing.values():
            nn.init.normal_(p, std=0.02)
        self.heads = MultiTaskHeads(h, len(self.roles), spec, n_bins, float(self.model_cfg.get("dropout", 0.25)))

    def encode(self, batch: dict):
        toks, attn = [], {}
        for r in self.roles:
            x, m, pres = batch["x"][r], batch["mask"][r], batch["present"][r]
            t, a = self.branches[r](x, m)                                  # [B,1,H], [B,1,N]
            fill = self.missing[r].expand(t.shape[0], 1, t.shape[2]).to(t.dtype)
            toks.append(torch.where(pres[:, None, None], t, fill))
            attn[r] = a[:, 0]
        return torch.cat(toks, 1), attn

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        tokens, attn = self.encode(batch)
        out = self.heads(tokens)
        out["tokens"] = tokens
        out["attention"] = attn
        return out


def save_token_model(model: TokenMTLModel, path: Path, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"format": FORMAT, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "in_dims": model.in_dims, "model_cfg": model.model_cfg, "spec": model.spec.to_dict(),
           "n_bins": model.n_bins, "extra": extra or {}}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_token_model(path: Path, map_location="cpu"):
    try:
        obj = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location=map_location)
    if obj.get("format") != FORMAT:
        raise ValueError(f"{path.name} 不是本项目 v2 token 模型文件")
    model = TokenMTLModel(obj["in_dims"], obj["model_cfg"], TaskSpec.from_dict(obj["spec"]), obj["n_bins"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj.get("extra", {})
