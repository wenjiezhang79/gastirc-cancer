"""token 模型的单次训练与预测（一个折、一个随机种子）。与 v1 训练循环一致：

AdamW + 预热 + 余弦退火；GPU 上 BF16；梯度裁剪与累积；早停监测内层验证集的总损失
（固定权重下各任务损失的平均），恢复最佳 epoch 的参数。生存时间段、正样本权重、基线初始化只用训练部分计算。
损失函数直接复用 v1（training/trainer.compute_losses），不重复实现。
"""

from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..losses import LossWeighter, survival_curve
from ..training.bag_dataset import build_targets, population_index
from ..training.survival_bins import empirical_hazard_logits, fit_bins, survival_at
from ..training.task_spec import TaskSpec
from ..training.trainer import _pos_weights, compute_losses, set_seed
from .data import TokenDataset, batch_to, make_token_collate


def class_weights(labels: pd.DataFrame, spec: TaskSpec, enabled: bool) -> Dict[str, list]:
    out = {}
    if enabled:
        for t, c in spec.multiclass.items():
            v = labels[t].dropna().astype(int).to_numpy()
            cnt = np.bincount(v, minlength=c).astype(float)
            out[t] = np.where(cnt > 0, cnt.sum() / (c * np.maximum(cnt, 1)), 1.0).tolist()
    return out


def make_loader(sids, stores, roles, dims, targets, sec, train: bool, batch_size: int,
                ds_kwargs: Optional[dict] = None) -> DataLoader:
    """ds_kwargs：模型额外的数据增强参数（v3 融合模型的模态丢弃等，由 model.dataset_kwargs 给出），只在训练时使用。"""
    tc = sec.train
    ds = TokenDataset(sids, stores, train=train,
                      slice_dropout=float(tc.get("slice_dropout", 0.0)) if train else 0.0,
                      roi_dropout=float(tc.get("roi_dropout", 0.0)) if train else 0.0,
                      **(ds_kwargs if train and ds_kwargs else {}))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=int(tc.get("num_workers", 0)),
                      collate_fn=make_token_collate(roles, dims, targets), drop_last=False)


def train_one(sec, spec: TaskSpec, stores: Sequence, model_factory: Callable[[int], torch.nn.Module],
              lab_tr: pd.DataFrame, lab_va: pd.DataFrame, device: torch.device, seed: int, logger,
              log_prefix: str = "") -> Tuple[torch.nn.Module, dict, List[dict]]:
    """sec：配置节（如 cfg.v2）；model_factory(n_bins) 返回未训练的模型（须有 .heads 与 .roles）。

    v3 融合模型可选提供以下钩子（v2 模型没有这些属性，行为不变）：
    - ``branch_prefixes``：分支参数名前缀；这些参数的学习率 = lr × sec.train.branch_lr_mult；
      ``sec.train.freeze_branches`` = true 时全程冻结分支，``freeze_epochs`` > 0 时前若干 epoch 冻结；
    - ``set_branches_frozen(bool)``：冻结 / 解冻分支（冻结时分支处于 eval 模式）；
    - ``aux_heads``（ModuleDict）与 ``aux_losses(out, loss_fn)``：单模态辅助头的损失（只加在训练损失上，不参与早停监测）；
    - ``dataset_kwargs(train_cfg)``：训练时的额外数据增强参数（模态丢弃等）。
    """
    tc, lc = sec.train, sec.loss
    set_seed(seed)
    n_bins = int(lc.get("survival_bins", 4))
    survival_loss = str(lc.get("survival_loss", "discrete"))
    raw_tr = build_targets(lab_tr, spec)
    bins = fit_bins({ep: raw_tr[f"time:{ep}"] for ep in spec.survival},
                    {ep: raw_tr[f"event:{ep}"] for ep in spec.survival},
                    {ep: raw_tr[f"pop:{ep}"] for ep in spec.survival},
                    spec.survival, {ep: spec.n_pops(ep) for ep in spec.survival}, n_bins)
    tg_tr = build_targets(lab_tr, spec, bins)
    tg_va = build_targets(lab_va, spec, bins)
    pos_w = _pos_weights(lab_tr, spec, float(lc.get("max_pos_weight", 10)))
    cls_w = class_weights(lab_tr, spec, bool(lc.get("lauren_class_weight", False)))

    model = model_factory(n_bins)
    roles, dims = model.roles, model.in_dims
    aux_heads = getattr(model, "aux_heads", None) or {}
    for ep in spec.survival:
        init = np.stack([empirical_hazard_logits(tg_tr[f"bin:{ep}"][tg_tr[f"pop:{ep}"] == p],
                                                 tg_tr[f"event:{ep}"][tg_tr[f"pop:{ep}"] == p], n_bins)
                         for p in range(spec.n_pops(ep))])
        model.heads.init_baseline(ep, torch.tensor(init, dtype=torch.float32))
        for h in aux_heads.values():
            h.init_baseline(ep, torch.tensor(init, dtype=torch.float32))
    model.to(device)
    weighter = LossWeighter(spec.all_tasks, str(lc.get("weighting", "uncertainty")),
                            dict(lc.get("fixed_weights", {}) or {})).to(device)
    pos_w_t = {t: torch.tensor(w, device=device) for t, w in pos_w.items()}
    cls_w_t = {t: torch.tensor(w, dtype=torch.float32, device=device) for t, w in cls_w.items()}

    prefixes = tuple(getattr(model, "branch_prefixes", ()) or ())
    freeze_all = bool(tc.get("freeze_branches", False)) and bool(prefixes)
    freeze_epochs = int(tc.get("freeze_epochs", 0) or 0) if prefixes else 0
    lr = float(tc.lr)
    if prefixes:
        groups = {(b, d): [] for b in (False, True) for d in (False, True)}
        for n, p in model.named_parameters():
            is_branch = n.startswith(prefixes)
            if is_branch and freeze_all:
                p.requires_grad_(False)
                continue
            nd = p.ndim < 2 or "baseline" in n or "norm" in n.lower() or n.startswith("missing")
            groups[(is_branch, not nd)].append(p)
        mult = float(tc.get("branch_lr_mult", 1.0))
        param_groups = []
        for (is_branch, dec), ps in groups.items():
            extra_ps = list(weighter.parameters()) if (not is_branch and not dec) else []
            if ps or extra_ps:
                param_groups.append({"params": ps + extra_ps, "weight_decay": float(tc.weight_decay) if dec else 0.0,
                                     "lr": lr * (mult if is_branch else 1.0)})
        opt = torch.optim.AdamW(param_groups, lr=lr)
        if freeze_all:
            model.set_branches_frozen(True)
    else:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.ndim < 2 or "baseline" in n or "norm" in n.lower() or n.startswith("missing") else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(tc.weight_decay)},
                                 {"params": no_decay + list(weighter.parameters()), "weight_decay": 0.0}], lr=lr)
    aux_weight = float(getattr(model, "aux_weight", 0.0) or 0.0)

    class _Shim:  # 让 compute_losses 使用辅助头的生存基线
        def __init__(self, heads):
            self.heads = heads

    def aux_loss_fn(o, heads, keep, targets):
        sub = {k: v[keep] for k, v in targets.items()}
        return compute_losses(o, sub, spec, _Shim(heads), pos_w_t, cls_w_t, survival_loss)
    sids_tr, sids_va = list(lab_tr["study_id"]), list(lab_va["study_id"])
    bs = int(tc.batch_size)
    accum = max(1, int(tc.get("grad_accum", 1)))
    steps_per_epoch = max(1, math.ceil(math.ceil(len(sids_tr) / bs) / accum))
    total = steps_per_epoch * int(tc.epochs)
    warm = steps_per_epoch * int(tc.get("warmup_epochs", 2))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_bf16 = device.type == "cuda" and str(tc.get("precision", "bf16")) == "bf16"
    ds_kw = model.dataset_kwargs(tc) if hasattr(model, "dataset_kwargs") else None
    dl_tr = make_loader(sids_tr, stores, roles, dims, tg_tr, sec, True, bs, ds_kw)
    dl_va = make_loader(sids_va, stores, roles, dims, tg_va, sec, False, bs)

    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    patience = int(tc.get("patience", 10))
    history: List[dict] = []
    for epoch in range(int(tc.epochs)):
        t0 = time.time()
        if freeze_epochs and not freeze_all:
            model.set_branches_frozen(epoch < freeze_epochs)
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl_tr):
            batch = batch_to(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(batch)
            losses = compute_losses(out, batch["targets"], spec, model, pos_w_t, cls_w_t, survival_loss)
            loss = weighter(losses)
            if aux_heads and aux_weight > 0:
                loss = loss + aux_weight * model.aux_losses(out, lambda o, h, keep: aux_loss_fn(o, h, keep, batch["targets"]))
            loss = loss / accum
            if loss.requires_grad:
                loss.backward()
            tot += float(loss.detach()) * accum
            nb += 1
            if (bi + 1) % accum == 0 or bi + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad]
                                               + list(weighter.parameters()),
                                               float(tc.get("grad_clip", 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        val = evaluate_loss(model, dl_va, spec, pos_w_t, cls_w_t, survival_loss, device, use_bf16)
        monitor = weighter.monitor({t: (torch.tensor(v[0]), v[1]) for t, v in val.items()})
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot / max(1, nb),
               "val_monitor": monitor, "seconds": round(time.time() - t0, 2)}
        row.update({f"val_{t}": v[0] for t, v in val.items()})
        row.update({f"w_{t}": w for t, w in weighter.weights().items()})
        history.append(row)
        improved = np.isfinite(monitor) and monitor < best - 1e-4
        if improved:
            best, best_epoch, bad = monitor, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if logger is not None and (epoch % 5 == 0 or improved):
            logger.info("%s epoch %d：训练损失 %.4f，内层验证 %.4f%s", log_prefix, epoch, row["train_loss"], monitor,
                        "（最佳）" if improved else "")
        if bad >= patience:
            if logger is not None:
                logger.info("%s 早停：内层验证损失连续 %d 个 epoch 未下降，最佳 epoch %d", log_prefix, patience, best_epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    extra = {"bins": bins, "pos_weight": pos_w, "class_weight": cls_w, "best_epoch": best_epoch,
             "best_val_monitor": best, "seed": seed, "survival_loss": survival_loss, "weights": weighter.weights()}
    return model, extra, history


@torch.no_grad()
def evaluate_loss(model, loader, spec, pos_w, cls_w, survival_loss, device, use_bf16) -> Dict[str, tuple]:
    model.eval()
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for batch in loader:
        batch = batch_to(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(batch)
        for t, (l, n) in compute_losses(out, batch["targets"], spec, model, pos_w, cls_w, survival_loss).items():
            if n:
                sums[t] = sums.get(t, 0.0) + float(l) * n
                counts[t] = counts.get(t, 0) + n
    return {t: (sums[t] / counts[t], counts[t]) for t in sums}


@torch.no_grad()
def predict(model, extra: dict, stores: Sequence, labels: pd.DataFrame, device, batch_size: int, horizons,
            keep_attention: bool = True):
    """返回 (预测表, 注意力 {sid: {角色: [n]}}, token [n_例, R, H])。"""
    spec: TaskSpec = model.spec
    model.eval().to(device)
    sids = list(labels["study_id"])
    tg = build_targets(labels, spec)
    pop_all = population_index(labels)
    dl = DataLoader(TokenDataset(sids, stores, train=False), batch_size=batch_size, shuffle=False,
                    collate_fn=make_token_collate(model.roles, model.in_dims, tg))
    rows: List[dict] = []
    attn_out: Dict[str, Dict[str, np.ndarray]] = {}
    toks = []
    for batch in dl:
        idx = batch["index"].numpy()
        b = batch_to(batch, device)
        out = model(b)
        for k, i in enumerate(idx):
            r = {"study_id": sids[i]}
            for role in model.roles:
                r[f"has_{role}"] = int(bool(batch["present"][role][k]))
            for name, v in (out.get("row_extras") or {}).items():   # v3：CLS 对各 token 的注意力等
                r[name] = float(v[k])
            for t, c in spec.multiclass.items():
                p = torch.softmax(out[f"logits:{t}"][k].float(), -1).cpu().numpy()
                for ci in range(c):
                    r[f"{t}_p{ci}"] = float(p[ci])
            for t in spec.binary:
                r[f"{t}_prob"] = float(torch.sigmoid(out[f"logit:{t}"][k].float()))
            for t in spec.regression:
                r[f"{t}_pred"] = float(out[f"value:{t}"][k])
            for ep in spec.survival:
                risk = out[f"risk:{ep}"][k].float()
                r[f"{ep}_risk"] = float(risk)
                p = int(pop_all[i]) if spec.n_pops(ep) > 1 else 0
                if extra.get("survival_loss", "discrete") == "discrete" and p >= 0:
                    s = survival_curve(risk[None], model.heads.baseline[ep].float(),
                                       torch.tensor([p], device=risk.device)).cpu().numpy()
                    for h, v in zip(horizons, survival_at(s, extra["bins"][ep][p], horizons)[0]):
                        r[f"{ep}_S{int(h)}m"] = float(v)
                else:
                    for h in horizons:
                        r[f"{ep}_S{int(h)}m"] = float("nan")
            rows.append(r)
            if keep_attention:
                attn_out[sids[i]] = {role: out["attention"][role][k, : int(batch["mask"][role][k].sum())]
                                     .float().cpu().numpy() for role in model.roles if bool(batch["present"][role][k])}
        toks.append(out["tokens"].float().cpu().numpy())
    tokens = np.concatenate(toks, 0) if toks else np.zeros((0,))
    return pd.DataFrame(rows), attn_out, tokens
