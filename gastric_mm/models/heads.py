"""多任务输出头。

病理 token [B, K, H] → 拼接 → 共享层 → 各任务输出：
- multiclass：每个任务一个 C 类 logits（Lauren 等）
- binary：每个二分类任务一个 logit（HER2、MSI、EBV、PD-L1、Claudin18.2 …）
- regression：每个回归任务一个值（Ki-67 比例，可选）
- survival：每个终点一个**共享风险评分** r(x)；另有该终点的基线参数 baseline[人群, 时间段]，
  hazard logit = baseline[p, k] + r(x)。基线按训练数据的经验风险初始化。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..training.task_spec import TaskSpec


class MultiTaskHeads(nn.Module):
    def __init__(self, token_dim: int, n_tokens: int, spec: TaskSpec, n_bins: int, dropout: float = 0.25):
        super().__init__()
        self.spec = spec
        d_in = token_dim * n_tokens
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, token_dim), nn.GELU(), nn.Dropout(dropout))
        self.multiclass = nn.ModuleDict({t: nn.Linear(token_dim, c) for t, c in spec.multiclass.items()})
        self.binary = nn.Linear(token_dim, len(spec.binary)) if spec.binary else None
        self.regression = nn.Linear(token_dim, len(spec.regression)) if spec.regression else None
        self.risk = nn.Linear(token_dim, len(spec.survival)) if spec.survival else None
        self.baseline = nn.ParameterDict({
            ep: nn.Parameter(torch.zeros(spec.n_pops(ep), n_bins)) for ep in spec.survival
        })
        if self.risk is not None:
            nn.init.zeros_(self.risk.bias)
            nn.init.normal_(self.risk.weight, std=0.01)

    def init_baseline(self, ep: str, logits: torch.Tensor) -> None:
        with torch.no_grad():
            self.baseline[ep].copy_(logits)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(tokens.flatten(1))
        out: Dict[str, torch.Tensor] = {"embedding": z}
        for t, head in self.multiclass.items():
            out[f"logits:{t}"] = head(z)
        if self.binary is not None:
            b = self.binary(z)
            for i, t in enumerate(self.spec.binary):
                out[f"logit:{t}"] = b[:, i]
        if self.regression is not None:
            r = self.regression(z)
            for i, t in enumerate(self.spec.regression):
                out[f"value:{t}"] = r[:, i]
        if self.risk is not None:
            r = self.risk(z)
            for i, ep in enumerate(self.spec.survival):
                out[f"risk:{ep}"] = r[:, i]
        return out
