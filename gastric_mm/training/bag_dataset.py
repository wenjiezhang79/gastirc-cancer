"""病理特征数据集：每个病人一个"包"（全部切块特征），加上各任务标签。

- 特征来自 ``<work_dir>/wsi/features/<编码器>/<study_id>.h5``；多个编码器时逐切块拼接
  （要求切块坐标一致，TRIDENT 各编码器使用同一套切块）。
- 默认把特征以 float16 缓存在内存中（约 2000 例 × 1000 块 × 1536 维 ≈ 6 GB），v1.feature_cache=false 可关闭。
- 训练时每个包最多随机抽 max_tiles 个切块，并随机丢弃 tile_dropout 比例的切块（特征层面增强）。
- 批处理：不同长度的包补零对齐，mask 标记真实切块。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..paths import WorkPaths
from ..utils.errors import UserError
from ..wsi.features_io import read_features
from .survival_bins import assign_bins
from .task_spec import POPULATIONS, TaskSpec


class FeatureStore:
    def __init__(self, cfg: Config, encoders: Sequence[str], cache: bool = True):
        if not encoders:
            raise UserError("v1.encoders 为空", "请在配置中填写要使用的病理编码器，如 [uni2_h]。")
        self.wp = WorkPaths(cfg)
        self.encoders = list(encoders)
        self.cache_enabled = cache
        self._cache: Dict[str, np.ndarray] = {}
        self._dim: Optional[int] = None

    def path(self, enc: str, sid: str):
        return self.wp.wsi_features(enc, sid)

    def available(self, sids: Sequence[str]) -> List[str]:
        return [s for s in sids if all(self.path(e, s).exists() for e in self.encoders)]

    def load(self, sid: str) -> np.ndarray:
        if sid in self._cache:
            return self._cache[sid]
        parts, ref = [], None
        for e in self.encoders:
            f, c, _ = read_features(self.path(e, sid))
            if ref is None:
                ref = c
            elif c.shape != ref.shape or not np.array_equal(c, ref):
                raise UserError(f"{sid} 在编码器 {self.encoders[0]} 与 {e} 之间的切块坐标不一致，无法拼接",
                                "请检查 TRIDENT 导入报告；拼接要求两个编码器使用同一套切块。")
            parts.append(np.asarray(f, dtype=np.float16))
        feats = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if self._dim is None:
            self._dim = int(feats.shape[1])
        elif feats.shape[1] != self._dim:
            raise UserError(f"{sid} 的特征维数 {feats.shape[1]} 与其他病例 {self._dim} 不一致")
        if len(feats) == 0:
            raise UserError(f"{sid} 的特征文件中没有切块", "请检查该切片的组织检测与特征提取结果。")
        if self.cache_enabled:
            self._cache[sid] = feats
        return feats

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("尚未读取任何特征")
        return self._dim


def population_index(labels: pd.DataFrame) -> np.ndarray:
    pop = labels["population"].map(lambda v: POPULATIONS.index(v) if v in POPULATIONS else -1)
    return pop.to_numpy(dtype=int)


def build_targets(labels: pd.DataFrame, spec: TaskSpec, bins: Optional[Dict[str, list]] = None) -> Dict[str, np.ndarray]:
    """labels：按病例顺序排列的派生标签表（labels.derive_labels 的输出）。"""
    t: Dict[str, np.ndarray] = {}
    for name in spec.multiclass:
        t[f"mc:{name}"] = labels[name].to_numpy(dtype=float)
    for name in spec.binary:
        t[f"bin:{name}"] = labels[name].to_numpy(dtype=float)
    for name in spec.regression:
        t[f"reg:{name}"] = labels[name].to_numpy(dtype=float)
    pop = population_index(labels)
    for ep in spec.survival:
        time = labels[f"{ep}_time"].to_numpy(dtype=float)
        ev = labels[f"{ep}_event"].to_numpy(dtype=float)
        p = pop.copy() if spec.n_pops(ep) > 1 else np.zeros(len(labels), dtype=int)
        valid = np.isfinite(time) & np.isfinite(ev) & (p >= 0)
        time = np.where(valid, time, np.nan)
        ev = np.where(valid, ev, np.nan)
        t[f"time:{ep}"] = time
        t[f"event:{ep}"] = ev
        t[f"pop:{ep}"] = np.where(valid, p, 0).astype(int)
        if bins is not None:
            b = np.full(len(labels), -1, dtype=int)
            for pi, edges in enumerate(bins[ep]):
                m = valid & (p == pi)
                b[m] = assign_bins(time[m], edges)
            t[f"bin:{ep}"] = b
    return t


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, sids: Sequence[str], store: FeatureStore, targets: Dict[str, np.ndarray],
                 train: bool = False, max_tiles: Optional[int] = None, tile_dropout: float = 0.0):
        self.sids = list(sids)
        self.store = store
        self.targets = targets
        self.train = train
        self.max_tiles = max_tiles
        self.tile_dropout = tile_dropout

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        f = self.store.load(self.sids[i])
        n = len(f)
        if self.train:
            keep = n
            if self.tile_dropout > 0 and n > 1:
                keep = max(1, int(round(n * (1 - self.tile_dropout))))
            if self.max_tiles:
                keep = min(keep, int(self.max_tiles))
            if keep < n:
                idx = torch.randperm(n)[:keep].sort().values.numpy()
                f = f[idx]
        return i, torch.from_numpy(np.asarray(f, dtype=np.float32))


def make_collate(targets: Dict[str, np.ndarray]):
    def collate(batch):
        idx = [b[0] for b in batch]
        feats = [b[1] for b in batch]
        n_max = max(x.shape[0] for x in feats)
        d = feats[0].shape[1]
        x = torch.zeros(len(feats), n_max, d)
        mask = torch.zeros(len(feats), n_max, dtype=torch.bool)
        for k, f in enumerate(feats):
            x[k, : f.shape[0]] = f
            mask[k, : f.shape[0]] = True
        tg = {}
        for key, arr in targets.items():
            v = arr[idx]
            tg[key] = torch.as_tensor(v, dtype=torch.long if v.dtype.kind in "iu" else torch.float32)
        return {"index": torch.as_tensor(idx), "feats": x, "mask": mask, "targets": tg}
    return collate
