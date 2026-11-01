"""单个模型的训练与预测（一个折、一个随机种子）。

训练：AdamW + 预热 + 余弦退火；GPU 上 BF16 混合精度；梯度裁剪、梯度累积；
早停监测**内层验证集**的总损失（固定权重下各任务损失的平均），连续 patience 个 epoch 不下降即停止，
恢复最佳 epoch 的参数。生存时间段、正样本权重、基线初始化均只用训练部分计算。
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..losses import (LossWeighter, cox_loss, discrete_time_nll, masked_bce, masked_ce, masked_mse,
                      survival_curve)
from ..models.path_model import PathModel
from .bag_dataset import BagDataset, FeatureStore, build_targets, make_collate, population_index
from .survival_bins import empirical_hazard_logits, fit_bins, survival_at
from .task_spec import TaskSpec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to(batch: dict, device) -> dict:
    out = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["targets"] = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    return out


def compute_losses(out: dict, tg: dict, spec: TaskSpec, model: PathModel, pos_weight: Dict[str, torch.Tensor],
                   class_weight: Dict[str, torch.Tensor], survival_loss: str) -> Dict[str, tuple]:
    losses = {}
    for t in spec.multiclass:
        losses[t] = masked_ce(out[f"logits:{t}"], tg[f"mc:{t}"], class_weight.get(t))
    for t in spec.binary:
        losses[t] = masked_bce(out[f"logit:{t}"], tg[f"bin:{t}"], pos_weight.get(t))
    for t in spec.regression:
        losses[t] = masked_mse(out[f"value:{t}"], tg[f"reg:{t}"])
    for ep in spec.survival:
        risk = out[f"risk:{ep}"]
        if survival_loss == "discrete":
            losses[ep] = discrete_time_nll(risk, model.heads.baseline[ep], tg[f"bin:{ep}"], tg[f"event:{ep}"],
                                           tg[f"pop:{ep}"])
        else:  # Cox，按人群分层
            pop = tg[f"pop:{ep}"]
            tot, n_tot, n_ev = None, 0, 0
            for p in torch.unique(pop):
                m = pop == p
                l, n = cox_loss(risk[m], tg[f"time:{ep}"][m], tg[f"event:{ep}"][m])
                if n:
                    w = float(torch.nan_to_num(tg[f"event:{ep}"][m]).sum())
                    tot = l * w if tot is None else tot + l * w
                    n_tot += n
                    n_ev += w
            losses[ep] = (tot / max(n_ev, 1.0), n_tot) if tot is not None else (risk.sum() * 0.0, 0)
    return losses


def _class_weights(labels: pd.DataFrame, spec: TaskSpec, cfg: Config) -> Dict[str, list]:
    out = {}
    if cfg.v1.loss.get("lauren_class_weight", False):
        for t, c in spec.multiclass.items():
            v = labels[t].dropna().astype(int).to_numpy()
            cnt = np.bincount(v, minlength=c).astype(float)
            w = np.where(cnt > 0, cnt.sum() / (c * np.maximum(cnt, 1)), 1.0)
            out[t] = w.tolist()
    return out


def _pos_weights(labels: pd.DataFrame, spec: TaskSpec, max_w: float) -> Dict[str, float]:
    out = {}
    for t in spec.binary:
        v = labels[t].dropna()
        n1, n0 = float((v == 1).sum()), float((v == 0).sum())
        out[t] = float(np.clip(n0 / n1, 1.0, max_w)) if n1 > 0 else 1.0
    return out


def make_loader(sids, store, targets, cfg: Config, train: bool, batch_size: int) -> DataLoader:
    tc = cfg.v1.train
    ds = BagDataset(sids, store, targets, train=train,
                    max_tiles=tc.get("max_tiles") if train else None,
                    tile_dropout=float(tc.get("tile_dropout", 0.0)) if train else 0.0)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=int(tc.get("num_workers", 0)),
                      collate_fn=make_collate(targets), drop_last=False)


def train_one(cfg: Config, spec: TaskSpec, store: FeatureStore, lab_tr: pd.DataFrame, lab_va: pd.DataFrame,
              device: torch.device, seed: int, logger, log_prefix: str = "") -> Tuple[PathModel, dict, List[dict]]:
    tc, lc, mc = cfg.v1.train, cfg.v1.loss, dict(cfg.v1.model)
    set_seed(seed)
    n_bins = int(lc.get("survival_bins", 4))
    survival_loss = str(lc.get("survival_loss", "discrete"))

    # ---- 只用训练部分计算的量 ----
    raw_tr = build_targets(lab_tr, spec)
    bins = fit_bins({ep: raw_tr[f"time:{ep}"] for ep in spec.survival},
                    {ep: raw_tr[f"event:{ep}"] for ep in spec.survival},
                    {ep: raw_tr[f"pop:{ep}"] for ep in spec.survival},
                    spec.survival, {ep: spec.n_pops(ep) for ep in spec.survival}, n_bins)
    tg_tr = build_targets(lab_tr, spec, bins)
    tg_va = build_targets(lab_va, spec, bins)
    pos_w = _pos_weights(lab_tr, spec, float(lc.get("max_pos_weight", 10)))
    cls_w = _class_weights(lab_tr, spec, cfg)

    sids_tr, sids_va = list(lab_tr["study_id"]), list(lab_va["study_id"])
    store.load(sids_tr[0])
    model = PathModel(store.dim, mc, spec, n_bins)
    for ep in spec.survival:
        init = np.stack([empirical_hazard_logits(tg_tr[f"bin:{ep}"][tg_tr[f"pop:{ep}"] == p],
                                                 tg_tr[f"event:{ep}"][tg_tr[f"pop:{ep}"] == p], n_bins)
                         for p in range(spec.n_pops(ep))])
        model.heads.init_baseline(ep, torch.tensor(init, dtype=torch.float32))
    model.to(device)
    weighter = LossWeighter(spec.all_tasks, str(lc.get("weighting", "uncertainty")),
                            dict(lc.get("fixed_weights", {}) or {})).to(device)
    pos_w_t = {t: torch.tensor(w, device=device) for t, w in pos_w.items()}
    cls_w_t = {t: torch.tensor(w, dtype=torch.float32, device=device) for t, w in cls_w.items()}

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "baseline" in n or "norm" in n.lower() else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": float(tc.weight_decay)},
        {"params": no_decay + list(weighter.parameters()), "weight_decay": 0.0},
    ], lr=float(tc.lr))
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
    dl_tr = make_loader(sids_tr, store, tg_tr, cfg, True, bs)
    dl_va = make_loader(sids_va, store, tg_va, cfg, False, bs)

    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    patience = int(tc.get("patience", 10))
    history: List[dict] = []
    for epoch in range(int(tc.epochs)):
        t0 = time.time()
        model.train()
        tot_loss, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl_tr):
            batch = _to(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(batch["feats"], batch["mask"])
            losses = compute_losses(out, batch["targets"], spec, model, pos_w_t, cls_w_t, survival_loss)
            loss = weighter(losses) / accum
            if loss.requires_grad:
                loss.backward()
            tot_loss += float(loss.detach()) * accum
            n_batches += 1
            if (bi + 1) % accum == 0 or bi + 1 == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(weighter.parameters()),
                                               float(tc.get("grad_clip", 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        val_task = evaluate_loss(model, dl_va, spec, pos_w_t, cls_w_t, survival_loss, device, use_bf16)
        monitor = weighter.monitor({t: (torch.tensor(v[0]), v[1]) for t, v in val_task.items()})
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss / max(1, n_batches),
               "val_monitor": monitor, "seconds": round(time.time() - t0, 2)}
        row.update({f"val_{t}": v[0] for t, v in val_task.items()})
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
    extra = {"encoders": list(store.encoders), "bins": bins, "pos_weight": pos_w, "class_weight": cls_w,
             "best_epoch": best_epoch, "best_val_monitor": best, "seed": seed, "feature_dim": store.dim,
             "survival_loss": survival_loss, "weights": weighter.weights()}
    return model, extra, history


@torch.no_grad()
def evaluate_loss(model, loader, spec, pos_w, cls_w, survival_loss, device, use_bf16) -> Dict[str, tuple]:
    model.eval()
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for batch in loader:
        batch = _to(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(batch["feats"], batch["mask"])
        for t, (l, n) in compute_losses(out, batch["targets"], spec, model, pos_w, cls_w, survival_loss).items():
            if n:
                sums[t] = sums.get(t, 0.0) + float(l) * n
                counts[t] = counts.get(t, 0) + n
    return {t: (sums[t] / counts[t], counts[t]) for t in sums}


@torch.no_grad()
def predict(model: PathModel, extra: dict, store: FeatureStore, labels: pd.DataFrame, device, batch_size: int,
            horizons, keep_attention: bool = True):
    """返回 (预测表, 注意力 {sid: [K, N]}, 病理 token [n, K, H])。"""
    spec = model.spec
    model.eval().to(device)
    sids = list(labels["study_id"])
    tg = build_targets(labels, spec)
    pop_all = population_index(labels)
    dl = DataLoader(BagDataset(sids, store, tg, train=False), batch_size=batch_size, shuffle=False,
                    collate_fn=make_collate(tg))
    rows: List[dict] = []
    attn_out: Dict[str, np.ndarray] = {}
    tokens_out = []
    for batch in dl:
        idx = batch["index"].numpy()
        out = model(batch["feats"].to(device), batch["mask"].to(device))
        lengths = batch["mask"].sum(1).numpy()
        for k, i in enumerate(idx):
            r = {"study_id": sids[i]}
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
                    sa = survival_at(s, extra["bins"][ep][p], horizons)[0]
                    for h, v in zip(horizons, sa):
                        r[f"{ep}_S{int(h)}m"] = float(v)
                else:
                    for h in horizons:
                        r[f"{ep}_S{int(h)}m"] = float("nan")
            rows.append(r)
            if keep_attention:
                attn_out[sids[i]] = out["attention"][k, :, : int(lengths[k])].float().cpu().numpy()
        tokens_out.append(out["tokens"].float().cpu().numpy())
    tokens = np.concatenate(tokens_out, 0) if tokens_out else np.zeros((0,))
    return pd.DataFrame(rows), attn_out, tokens
