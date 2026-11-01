"""损失函数。缺失标签（NaN）一律屏蔽；某任务在当前批次没有标签时返回 (0, 0)，由加权器跳过。

离散时间生存（主方案）
    第 k 个时间段的风险（hazard）logit = 该人群该时间段的基线 b[p, k] + 共享风险评分 r(x)。
    事件发生在第 j 段：  Σ_{k<j} log(1−h_k) + log h_j
    在第 j 段删失：      Σ_{k<j} log(1−h_k)            （只计算确定存活过的时间段）
    参见 Tutz & Schmid, Modeling Discrete Time-to-Event Data (2016)；Zadeh & Schmid (2021)。

Cox 部分似然（对照）
    批内 Breslow 近似。小批量下不稳定，这正是主方案选离散时间的原因。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

Pair = Tuple[torch.Tensor, int]


def _zero(ref: torch.Tensor) -> torch.Tensor:
    return ref.sum() * 0.0


def masked_ce(logits: torch.Tensor, target: torch.Tensor, class_weight: Optional[torch.Tensor] = None) -> Pair:
    ok = torch.isfinite(target)
    n = int(ok.sum())
    if n == 0:
        return _zero(logits), 0
    return F.cross_entropy(logits[ok].float(), target[ok].long(), weight=class_weight), n


def masked_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> Pair:
    """logits / target: [B]（单个二分类任务）。"""
    ok = torch.isfinite(target)
    n = int(ok.sum())
    if n == 0:
        return _zero(logits), 0
    return F.binary_cross_entropy_with_logits(logits[ok].float(), target[ok].float(), pos_weight=pos_weight), n


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> Pair:
    ok = torch.isfinite(target)
    n = int(ok.sum())
    if n == 0:
        return _zero(pred), 0
    return F.mse_loss(pred[ok].float(), target[ok].float()), n


def hazard_logits(risk: torch.Tensor, baseline: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
    """risk [B]，baseline [P, K]，pop [B] → 各时间段 hazard logit [B, K]。"""
    return baseline[pop.clamp(min=0)] + risk.float()[:, None]


def discrete_time_nll(risk: torch.Tensor, baseline: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor,
                      pop: torch.Tensor) -> Pair:
    """bin_idx 为 -1 表示该终点缺失。"""
    ok = bin_idx >= 0
    n = int(ok.sum())
    if n == 0:
        return _zero(risk) + _zero(baseline), 0
    L = hazard_logits(risk[ok], baseline, pop[ok])
    j = bin_idx[ok].long()
    e = event[ok].float()
    k = torch.arange(L.shape[1], device=L.device)[None, :]
    before = (k < j[:, None]).float()
    at = (k == j[:, None]).float()
    ll = (before * F.logsigmoid(-L)).sum(1) + e * (at * F.logsigmoid(L)).sum(1)
    return -ll.mean(), n


def survival_curve(risk: torch.Tensor, baseline: torch.Tensor, pop: torch.Tensor) -> torch.Tensor:
    """各时间段结束时的生存概率 S_k = Π_{m≤k}(1 − h_m)，[B, K]。"""
    h = torch.sigmoid(hazard_logits(risk, baseline, pop))
    return torch.cumprod(1 - h, dim=1)


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> Pair:
    ok = torch.isfinite(time) & torch.isfinite(event)
    r, t, e = risk[ok].float(), time[ok].float(), event[ok].float()
    n_ev = int(e.sum())
    if n_ev == 0 or len(r) < 2:
        return _zero(risk), 0
    order = torch.argsort(t, descending=True)
    r, e = r[order], e[order]
    lcse = torch.logcumsumexp(r, dim=0)          # 风险集：时间 ≥ t_i 的所有人
    return -((r - lcse) * e).sum() / n_ev, int(ok.sum())


class LossWeighter(nn.Module):
    """多任务加权。

    - uncertainty：每个任务一个可学习的 s_t，总损失 Σ exp(−s_t)·L_t + s_t（Kendall et al., 2018）；
    - fixed：Σ w_t·L_t（权重来自配置，未列出的为 1）。
    当前批次没有标签的任务整项跳过。
    """

    def __init__(self, tasks, mode: str = "uncertainty", fixed_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        if mode not in ("uncertainty", "fixed"):
            raise ValueError(f"未知的加权方式 {mode}（可选 uncertainty / fixed）")
        self.tasks = list(tasks)
        self.mode = mode
        self.fixed = {t: float((fixed_weights or {}).get(t, 1.0)) for t in self.tasks}
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks))) if mode == "uncertainty" else None

    def forward(self, losses: Dict[str, Pair]) -> torch.Tensor:
        total = None
        for i, t in enumerate(self.tasks):
            if t not in losses:
                continue
            loss, n = losses[t]
            if n == 0:
                continue
            if self.mode == "uncertainty":
                term = torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
            else:
                term = self.fixed[t] * loss
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))[0] if losses else torch.zeros(())
            return _zero(ref)
        return total

    def monitor(self, losses: Dict[str, Pair]) -> float:
        """早停监测用：固定权重下各任务损失的平均（不受可学习权重影响）。"""
        vals = [self.fixed[t] * float(l.detach()) for t, (l, n) in losses.items() if n > 0 and t in self.fixed]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def weights(self) -> Dict[str, float]:
        if self.mode == "uncertainty":
            return {t: float(torch.exp(-self.log_vars[i]).detach()) for i, t in enumerate(self.tasks)}
        return dict(self.fixed)
