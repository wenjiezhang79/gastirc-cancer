"""多示例（MIL）聚合器：把一个病人的全部切块特征 [N, D] 汇总成 K 个病理 token [K, H]。

输入统一为补零后的批次：x [B, N, D]，mask [B, N]（True = 真实切块）。补零部分不影响结果。
输出：tokens [B, K, H]、attention [B, K, N]（补零位置为 0，每行和为 1）。

- MeanPool：均值池化（简单基线），K = 1。
- GatedABMIL：门控注意力 MIL（Ilse et al., 2018），K 个注意力分支 → K 个 token。主方案。
- TransMIL：类别 token + 2 层 Transformer + 卷积位置编码 PPEG（Shao et al., 2021）。
  原文用 Nyström 近似注意力以处理上万切块；活检切块通常只有几百到几千，这里用标准注意力（更准确）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class InputProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MeanPool(nn.Module):
    n_tokens = 1

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.25, **_):
        super().__init__()
        self.proj = InputProjection(in_dim, hidden, dropout)
        self.out_dim = hidden

    def forward(self, x, mask):
        h = self.proj(x)
        w = mask.float()
        attn = w / w.sum(1, keepdim=True).clamp(min=1)
        tokens = torch.einsum("bn,bnh->bh", attn, h)[:, None, :]
        return tokens, attn[:, None, :]


class GatedABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, attn_dim: int = 128, n_branches: int = 2,
                 dropout: float = 0.25, **_):
        super().__init__()
        self.proj = InputProjection(in_dim, hidden, dropout)
        self.attn_v = nn.Linear(hidden, attn_dim)
        self.attn_u = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, n_branches)
        self.attn_drop = nn.Dropout(dropout)
        self.n_tokens = n_branches
        self.out_dim = hidden

    def forward(self, x, mask):
        h = self.proj(x)                                                   # [B, N, H]
        a = self.attn_w(self.attn_drop(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))))  # [B, N, K]
        a = a.transpose(1, 2).float()                                      # [B, K, N]
        a = a.masked_fill(~mask[:, None, :], float("-inf"))
        attn = torch.softmax(a, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                             # 全部补零的样本（不应出现）
        tokens = torch.einsum("bkn,bnh->bkh", attn.to(h.dtype), h)
        return tokens, attn


class PPEG(nn.Module):
    """金字塔位置编码：把切块序列排成方阵，用 3 种卷积核的深度卷积编码位置。逐样本处理以排除补零影响。"""

    def __init__(self, dim: int):
        super().__init__()
        self.c7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.c5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.c3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, tokens, lengths):
        out = tokens.clone()
        for b in range(tokens.shape[0]):
            n = int(lengths[b])
            if n == 0:
                continue
            side = int(math.ceil(math.sqrt(n)))
            seq = tokens[b, :n]
            if side * side > n:                         # 与原文一致：重复开头的切块补成方阵
                reps = int(math.ceil((side * side - n) / n))
                seq = torch.cat([seq] + [tokens[b, :n]] * reps, 0)[: side * side]
            g = seq.transpose(0, 1).reshape(1, -1, side, side)
            g = g + self.c7(g) + self.c5(g) + self.c3(g)
            out[b, :n] = g.reshape(-1, side * side).transpose(0, 1)[:n]
        return out


class TransLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x, key_padding_mask, need_weights=False):
        y = self.norm1(x)
        a, w = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=need_weights,
                         average_attn_weights=True)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, w


class TransMIL(nn.Module):
    n_tokens = 1

    def __init__(self, in_dim: int, hidden: int = 256, heads: int = 4, dropout: float = 0.25, **_):
        super().__init__()
        self.proj = InputProjection(in_dim, hidden, dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.cls, std=0.02)
        self.layer1 = TransLayer(hidden, heads, dropout)
        self.ppeg = PPEG(hidden)
        self.layer2 = TransLayer(hidden, heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x, mask):
        B = x.shape[0]
        h = self.proj(x)
        lengths = mask.sum(1)
        seq = torch.cat([self.cls.expand(B, -1, -1).to(h.dtype), h], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], 1)
        seq, _ = self.layer1(seq, kpm)
        body = self.ppeg(seq[:, 1:], lengths)
        seq = torch.cat([seq[:, :1], body], 1)
        seq, w = self.layer2(seq, kpm, need_weights=True)
        tokens = self.norm(seq[:, :1])
        attn = w[:, :1, 1:].float() * mask[:, None, :].float()            # 类别 token 对各切块的注意力
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-12)
        return tokens, attn


AGGREGATORS = {"meanpool": MeanPool, "abmil": GatedABMIL, "transmil": TransMIL}


def build_aggregator(name: str, in_dim: int, mc) -> nn.Module:
    if name not in AGGREGATORS:
        raise ValueError(f"未知的聚合器 {name}（可选 {', '.join(AGGREGATORS)}）")
    return AGGREGATORS[name](
        in_dim, hidden=int(mc.get("hidden_dim", 256)), attn_dim=int(mc.get("attn_dim", 128)),
        n_branches=int(mc.get("n_branches", 2)), dropout=float(mc.get("dropout", 0.25)),
        heads=int(mc.get("transmil_heads", 4)),
    )
