"""v3 融合模型：病理分支 + CT 分支 + 融合层 + 多任务输出头（docs/decisions.md 第十四节第 4 小节）。

```
病理切块特征 → 门控 ABMIL（K=2，来自 v1）→ 2 个病理 token ─┐
CT 肿瘤 ROI   → 门控注意力（K=1，来自 v2）→ 1 个 CT token  ─┼→ 各自线性投影到 d_model
CT 区域 ROI   → 门控注意力（K=1，来自 v2）→ 1 个 CT token  ─┘
[CLS] + 各槽位 token + 模态类型嵌入（path / ct）+ 槽位嵌入 → 1–2 层 Transformer（pre-LN）→ CLS → 多任务输出头
```

融合方式（``model.fusion``）：
- ``transformer``（主模型）：缺失槽位按 ``model.missing`` 处理——``mask``（主方案）：不参与注意力（key_padding_mask），
  CLS 只看实际存在的 token；``visible``（对照）：用可学习占位 token 代替并参与注意力；
- ``concat``（简单拼接对照）：各槽位 token（缺失用占位 token）拼接后直接进输出头；
- ``bilinear``（门控双线性对照，Pathomic Fusion 式）：病理向量与 CT 向量各自门控、降维、末尾补 1 后做外积，
  再与两个向量拼接进输出头；缺失模态用占位向量。

单模态辅助头（``model.aux_heads``，消融①）：病理 token、CT token 各自接一套输出头，损失乘 ``aux_weight`` 加到训练损失上，
防止模型只依赖信息最强的模态；不参与早停监测与预测输出。

模型文件 model.pt 保存结构参数、角色与维数、任务定义、生存时间段，加载时不依赖原配置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ..models.heads import MultiTaskHeads
from ..models.mil import TransLayer, build_aggregator
from ..training.task_spec import TaskSpec

FORMAT = "gastric_mm.fusion_model.v3"
PATH_ROLE = "path"
MODALITY_OF = {"path": "path", "tumor": "ct", "regional": "ct"}
FUSIONS = ("transformer", "concat", "bilinear")
MISSING_MODES = ("mask", "visible")


def modality_of(role: str) -> str:
    return MODALITY_OF.get(role, "ct")


class FusionModel(nn.Module):
    """in_dims：{角色: 输入特征维数}，角色为 path / tumor / regional 的子集。

    model_cfg 关键项：fusion、missing、d_model、n_layers、n_heads、dropout、aux_heads、aux_weight、bilinear_dim，
    以及分支结构 ``path``（aggregator / hidden_dim / attn_dim / n_branches / dropout，与 v1 模型一致）
    和 ``ct``（aggregator / hidden_dim / attn_dim / dropout，与 v2 模型一致）。
    """

    def __init__(self, in_dims: Dict[str, int], model_cfg: dict, spec: TaskSpec, n_bins: int):
        super().__init__()
        self.in_dims = dict(in_dims)
        self.roles = list(in_dims)
        self.model_cfg = dict(model_cfg)
        self.spec = spec
        self.n_bins = n_bins
        mc = self.model_cfg
        self.fusion = str(mc.get("fusion", "transformer"))
        self.missing_mode = str(mc.get("missing", "mask"))
        if self.fusion not in FUSIONS:
            raise ValueError(f"v3.model.fusion 应为 {'/'.join(FUSIONS)}，收到 {self.fusion}")
        if self.missing_mode not in MISSING_MODES:
            raise ValueError(f"v3.model.missing 应为 {'/'.join(MISSING_MODES)}，收到 {self.missing_mode}")
        d = int(mc.get("d_model", 128))
        drop = float(mc.get("dropout", 0.25))
        self.d_model = d

        # ---- 分支（名称前缀 branch_ 用于分支学习率 / 冻结） ----
        self.branch_cfg: Dict[str, dict] = {}
        self.branch = nn.ModuleDict()
        self.slots: List[Tuple[str, int]] = []
        for r, din in self.in_dims.items():
            if r == PATH_ROLE:
                bc = dict(mc.get("path", {}) or {})
            else:
                bc = dict(mc.get("ct", {}) or {}, n_branches=1)
            bc.setdefault("aggregator", "abmil")
            self.branch_cfg[r] = bc
            agg = build_aggregator(str(bc["aggregator"]), din, bc)
            self.branch[r] = agg
            self.slots += [(r, k) for k in range(agg.n_tokens)]
        self.branch_prefixes = ("branch.",)
        self._frozen = False
        n_slots = len(self.slots)

        # ---- 投影与嵌入 ----
        self.proj = nn.ModuleDict({r: nn.Sequential(nn.LayerNorm(self.branch[r].out_dim),
                                                    nn.Linear(self.branch[r].out_dim, d))
                                   for r in self.roles})
        self.mods = sorted({modality_of(r) for r in self.roles})
        self.type_emb = nn.Parameter(torch.zeros(len(("path", "ct")), d))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        self.missing_slots = nn.Parameter(torch.zeros(n_slots, d))      # 可学习占位 token（visible / concat / bilinear）
        for p in (self.type_emb, self.slot_emb, self.missing_slots):
            nn.init.normal_(p, std=0.02)
        self.register_buffer("slot_type", torch.tensor([0 if modality_of(r) == "path" else 1 for r, _ in self.slots]),
                             persistent=False)

        # ---- 融合层 ----
        if self.fusion == "transformer":
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.cls, std=0.02)
            n_layers, heads = int(mc.get("n_layers", 1)), int(mc.get("n_heads", 4))
            if d % heads:
                raise ValueError(f"d_model={d} 不能被 n_heads={heads} 整除")
            self.layers = nn.ModuleList([TransLayer(d, heads, drop) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d)
            self.heads = MultiTaskHeads(d, 1, spec, n_bins, drop)
        elif self.fusion == "concat":
            self.heads = MultiTaskHeads(d, n_slots, spec, n_bins, drop)
        else:  # bilinear
            b = int(mc.get("bilinear_dim", 32))
            self.missing_mod = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in ("path", "ct")})
            for p in self.missing_mod.values():
                nn.init.normal_(p, std=0.02)
            self.bil_gate = nn.ModuleDict({m: nn.Bilinear(d, d, b) for m in ("path", "ct")})
            self.bil_red = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, b), nn.ReLU()) for m in ("path", "ct")})
            self.bil_out = nn.Sequential(nn.LayerNorm((b + 1) ** 2), nn.Dropout(drop), nn.Linear((b + 1) ** 2, d),
                                         nn.ReLU())
            self.heads = MultiTaskHeads(d, 3, spec, n_bins, drop)

        # ---- 单模态辅助头（消融①） ----
        self.aux_weight = float(mc.get("aux_weight", 0.3)) if bool(mc.get("aux_heads", False)) else 0.0
        if self.aux_weight > 0:
            self.aux_heads = nn.ModuleDict()
            for m in self.mods:
                n_m = sum(1 for r, _ in self.slots if modality_of(r) == m)
                self.aux_heads[m] = MultiTaskHeads(d, n_m, spec, n_bins, drop)
        else:
            self.aux_heads = None

    # ---------------- 分支冻结 ----------------
    def set_branches_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)
        for p in self.branch.parameters():
            p.requires_grad_(not self._frozen)
        if self._frozen:
            self.branch.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.branch.eval()
        return self

    def dataset_kwargs(self, train_cfg) -> dict:
        ct_roles = [r for r in self.roles if modality_of(r) == "ct"]
        aug = {}
        if PATH_ROLE in self.roles:
            aug[PATH_ROLE] = {"dropout": float(train_cfg.get("tile_dropout", 0.0)),
                              "max_n": train_cfg.get("max_tiles")}
        return {"groups": {r: modality_of(r) for r in self.roles},
                "modality_dropout": float(train_cfg.get("modality_dropout", 0.0)),
                "roi_roles": ct_roles, "role_aug": aug}

    # ---------------- 前向 ----------------
    def encode(self, batch: dict):
        """返回 槽位 token [B,S,d]、槽位是否存在 [B,S]、各角色注意力。"""
        toks, present, attn = [], [], {}
        for r in self.roles:
            x, m, pres = batch["x"][r], batch["mask"][r], batch["present"][r]
            t, a = self.branch[r](x, m)                              # [B,K,H], [B,K,N]
            t = self.proj[r](t)
            toks.append(t)
            present.append(pres[:, None].expand(-1, t.shape[1]))
            attn[r] = a.transpose(1, 2) if a.shape[1] > 1 else a[:, 0]   # 病理 [B,N,K]；CT [B,N]
        tokens = torch.cat(toks, 1)                                    # [B,S,d]
        present = torch.cat(present, 1).bool()
        return tokens, present, attn

    def _fill(self, tokens, present):
        fill = self.missing_slots[None].to(tokens.dtype).expand_as(tokens)
        return torch.where(present[..., None], tokens, fill)

    def _modality_vector(self, tokens, present, m):
        sel = (self.slot_type == (0 if m == "path" else 1))
        t, p = tokens[:, sel], present[:, sel].float()
        n = p.sum(1, keepdim=True)
        v = (t * p[..., None]).sum(1) / n.clamp(min=1)
        return torch.where(n > 0, v, self.missing_mod[m][None].to(v.dtype).expand_as(v))

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        tokens, present, attn = self.encode(batch)
        B = tokens.shape[0]
        emb = self.type_emb[self.slot_type][None] + self.slot_emb[None]
        extras = {}
        if self.fusion == "transformer":
            seq_t = tokens if self.missing_mode == "mask" else self._fill(tokens, present)
            seq = torch.cat([self.cls.expand(B, -1, -1).to(seq_t.dtype), seq_t + emb.to(seq_t.dtype)], 1)
            if self.missing_mode == "mask":
                kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=present.device), ~present], 1)
            else:
                kpm = None
            w = None
            for i, layer in enumerate(self.layers):
                seq, w = layer(seq, kpm, need_weights=(i == len(self.layers) - 1))
            z = self.final_norm(seq[:, 0])
            out = self.heads(z[:, None])
            cls_w = w[:, 0, 1:].float()                                # 最后一层 CLS 对各槽位的注意力（头平均）
            cls_w = cls_w / cls_w.sum(1, keepdim=True).clamp(min=1e-12)  # 归一化为各槽位的占比（不含 CLS 自身）
            for j, (r, k) in enumerate(self.slots):
                extras[f"attn_{r}{k if sum(1 for rr, _ in self.slots if rr == r) > 1 else ''}"] = cls_w[:, j]
            for m in self.mods:
                sel = (self.slot_type == (0 if m == "path" else 1))
                extras[f"attn_mod_{m}"] = cls_w[:, sel].sum(1)
            fused = z[:, None]
        elif self.fusion == "concat":
            filled = self._fill(tokens, present) + emb.to(tokens.dtype)
            out = self.heads(filled)
            fused = out["embedding"][:, None]
        else:
            vp = self._modality_vector(tokens, present, "path")
            vc = self._modality_vector(tokens, present, "ct")
            hp = self.bil_red["path"](vp) * torch.sigmoid(self.bil_gate["path"](vp, vc))
            hc = self.bil_red["ct"](vc) * torch.sigmoid(self.bil_gate["ct"](vc, vp))
            one = torch.ones(B, 1, dtype=hp.dtype, device=hp.device)
            outer = torch.einsum("bi,bj->bij", torch.cat([hp, one], 1), torch.cat([hc, one], 1)).flatten(1)
            fz = self.bil_out(outer)
            out = self.heads(torch.stack([fz, vp, vc], 1))
            fused = out["embedding"][:, None]
        out["tokens"] = torch.cat([fused.to(tokens.dtype), tokens], 1)
        out["attention"] = attn
        out["row_extras"] = extras
        if self.aux_heads is not None and self.training:
            out["aux_tokens"] = (self._fill(tokens, present), present)
        return out

    def aux_losses(self, out: dict, loss_fn) -> torch.Tensor:
        """每个模态只对该模态存在的病例计算辅助头损失（各任务损失的平均），再对模态取平均。

        loss_fn(输出, 输出头, keep)：keep 为布尔掩码，由训练循环据此筛选目标后调用 compute_losses。
        """
        tokens, present = out["aux_tokens"]
        total, n = None, 0
        for m, head in self.aux_heads.items():
            sel = (self.slot_type == (0 if m == "path" else 1))
            keep = present[:, sel].any(1)
            if not bool(keep.any()):
                continue
            o = head(tokens[keep][:, sel])
            vals = [l for l, cnt in loss_fn(o, head, keep).values() if cnt]
            if vals:
                lm = torch.stack([v.float() for v in vals]).mean()
                total = lm if total is None else total + lm
                n += 1
        if total is None:
            return tokens.sum() * 0.0
        return total / n


def save_fusion_model(model: FusionModel, path: Path, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"format": FORMAT, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "in_dims": model.in_dims, "model_cfg": model.model_cfg, "spec": model.spec.to_dict(),
           "n_bins": model.n_bins, "extra": extra or {}}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_fusion_model(path: Path, map_location="cpu"):
    try:
        obj = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location=map_location)
    if obj.get("format") != FORMAT:
        raise ValueError(f"{path.name} 不是本项目 v3 融合模型文件")
    model = FusionModel(obj["in_dims"], obj["model_cfg"], TaskSpec.from_dict(obj["spec"]), obj["n_bins"])
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj.get("extra", {})
