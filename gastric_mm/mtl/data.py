"""token 模型的数据：特征仓库（按角色读取特征）、数据集（含训练增强）、批处理。

特征仓库接口（v3 的病理仓库也实现同样的接口）：
- ``roles``：该仓库提供的角色（如 CT 的 tumor / regional）；
- ``available(sids)``：有特征文件的病例；
- ``load(sid) -> {角色: [n, D] 数组}``（缺少的角色不在字典中）；
- ``dims -> {角色: D}``。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..ct.features_io import read_ct_features
from ..paths import WorkPaths
from ..utils.errors import UserError


class CTFeatureStore:
    """CT ROI 特征：``<work_dir>/ct/features/<编码器>/<study_id>.h5``。"""

    def __init__(self, cfg: Config, encoder: str, roles: Sequence[str], cache: bool = True):
        if not encoder:
            raise UserError("没有指定 CT 编码器", "请在配置 v2.encoder 中填写（与特征提取时的编码器名一致），或用 --encoder。")
        if not roles:
            raise UserError("v2.rois 为空", "请至少使用一个 ROI，如 [tumor, regional]。")
        self.wp = WorkPaths(cfg)
        self.encoder = encoder
        self.roles = list(roles)
        self.cache_enabled = cache
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._dims: Dict[str, int] = {}
        self.attrs: Optional[dict] = None

    def path(self, sid: str):
        return self.wp.ct_features(self.encoder, sid)

    def available(self, sids: Sequence[str]) -> List[str]:
        out = []
        for s in sids:
            if self.path(s).exists():
                if any(r in self.load(s) for r in self.roles):
                    out.append(s)
        return out

    def load(self, sid: str) -> Dict[str, np.ndarray]:
        if sid in self._cache:
            return self._cache[sid]
        rois, attrs = read_ct_features(self.path(sid))
        if self.attrs is None:
            self.attrs = attrs
        out = {}
        for r in self.roles:
            if r not in rois:
                continue
            f = np.asarray(rois[r], dtype=np.float16)
            d = int(f.shape[1])
            if r in self._dims and self._dims[r] != d:
                raise UserError(f"{sid} 的 {r} 特征维数 {d} 与其他病例 {self._dims[r]} 不一致",
                                "同一实验只能使用同一编码器提取的特征；请检查特征目录是否混入了其他编码器的结果。")
            self._dims[r] = d
            out[r] = f
        if self.cache_enabled:
            self._cache[sid] = out
        return out

    @property
    def dims(self) -> Dict[str, int]:
        missing = [r for r in self.roles if r not in self._dims]
        if missing:
            raise UserError(f"所有病例都没有 ROI {missing} 的特征",
                            f"请检查 v2.rois 与特征提取时编码器的 rois 设置是否一致（编码器 {self.encoder}）。")
        return {r: self._dims[r] for r in self.roles}


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, sids: Sequence[str], stores: Sequence, train: bool = False, slice_dropout: float = 0.0,
                 roi_dropout: float = 0.0):
        self.sids = list(sids)
        self.stores = list(stores)
        self.train = train
        self.slice_dropout = float(slice_dropout)
        self.roi_dropout = float(roi_dropout)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        sid = self.sids[i]
        feats: Dict[str, np.ndarray] = {}
        for st in self.stores:
            feats.update(st.load(sid))
        if self.train:
            if self.slice_dropout > 0:
                for r, f in list(feats.items()):
                    n = len(f)
                    keep = max(1, int(round(n * (1 - self.slice_dropout))))
                    if keep < n:
                        feats[r] = f[np.sort(np.random.permutation(n)[:keep])]
            if self.roi_dropout > 0 and len(feats) >= 2 and np.random.rand() < self.roi_dropout:
                drop = list(feats)[np.random.randint(len(feats))]
                feats = {k: v for k, v in feats.items() if k != drop}
        return i, {r: torch.from_numpy(np.asarray(f, dtype=np.float32)) for r, f in feats.items()}


def make_token_collate(roles: Sequence[str], dims: Dict[str, int], targets: Dict[str, np.ndarray]):
    roles = list(roles)

    def collate(batch):
        idx = [b[0] for b in batch]
        B = len(batch)
        x, mask, present = {}, {}, {}
        for r in roles:
            lens = [b[1][r].shape[0] if r in b[1] else 0 for b in batch]
            n = max(1, max(lens))
            xr = torch.zeros(B, n, dims[r])
            mr = torch.zeros(B, n, dtype=torch.bool)
            for k, b in enumerate(batch):
                if r in b[1]:
                    xr[k, : lens[k]] = b[1][r]
                    mr[k, : lens[k]] = True
            x[r], mask[r], present[r] = xr, mr, torch.tensor([ln > 0 for ln in lens])
        tg = {}
        for key, arr in targets.items():
            v = arr[idx]
            tg[key] = torch.as_tensor(v, dtype=torch.long if v.dtype.kind in "iu" else torch.float32)
        return {"index": torch.as_tensor(idx), "x": x, "mask": mask, "present": present, "targets": tg}

    return collate


def batch_to(batch: dict, device) -> dict:
    out = dict(batch)
    for k in ("x", "mask", "present"):
        out[k] = {r: v.to(device, non_blocking=True) for r, v in batch[k].items()}
    out["targets"] = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
    return out
