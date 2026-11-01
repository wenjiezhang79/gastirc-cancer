"""token 模型的数据：特征仓库（按角色读取特征）、数据集（含训练增强）、批处理。

特征仓库接口（CT：CTFeatureStore；v3 病理：PathFeatureStore；二者接口一致）：
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


class PathFeatureStore:
    """v3：病理切块特征（包装 v1 的 FeatureStore），角色名 ``path``，``load`` 返回 ``{"path": [N_切块, D]}``。

    多个编码器时逐切块拼接（与 v1 相同）。接口与 CTFeatureStore 一致（roles / available / load / dims）。
    """

    ROLE = "path"

    def __init__(self, cfg: Config, encoders: Sequence[str], cache: bool = True):
        from ..training.bag_dataset import FeatureStore

        if not encoders:
            raise UserError("没有指定病理编码器", "请在配置 v3.path_encoders 中填写（与 v1 选定的编码器一致），如 [uni2_h]。")
        self.inner = FeatureStore(cfg, list(encoders), cache=cache)
        self.encoders = list(encoders)
        self.roles = [self.ROLE]

    def available(self, sids: Sequence[str]) -> List[str]:
        return self.inner.available(sids)

    def load(self, sid: str) -> Dict[str, np.ndarray]:
        return {self.ROLE: self.inner.load(sid)}

    @property
    def dims(self) -> Dict[str, int]:
        if self.inner._dim is None:
            raise UserError("尚未读取任何病理特征，无法确定特征维数",
                            "请确认病例池中有病理特征文件（v3.path_encoders 与导入的编码器名一致）。")
        return {self.ROLE: self.inner.dim}


class MaskedStore:
    """包装一个仓库：
    - ``only``：只对这些病例返回特征，其余病例返回空字典（v3：病例池中有的病例缺某种模态，读取时按缺失处理）；
    - ``hide``：把部分角色"藏起来"，用于缺失鲁棒性分析（人为去掉 CT 或病理后再预测）。
    """

    def __init__(self, store, hide: Sequence[str] = (), only: Optional[Sequence[str]] = None):
        self.store = store
        self.hide = set(hide)
        self.only = None if only is None else set(only)
        self.roles = list(store.roles)

    def available(self, sids):
        out = self.store.available(sids)
        return out if self.only is None else [s for s in out if s in self.only]

    def load(self, sid):
        if self.only is not None and sid not in self.only:
            return {}
        if len(self.hide) >= len(self.roles) and self.hide >= set(self.roles):
            return {}
        return {r: v for r, v in self.store.load(sid).items() if r not in self.hide}

    @property
    def dims(self):
        return self.store.dims


class TokenDataset(torch.utils.data.Dataset):
    """训练增强（只在 train=True 时）：

    - slice_dropout：随机丢弃各角色的部分行（2.5D 层）；``role_aug`` 可为个别角色单独指定
      ``{"dropout": 比例, "max_n": 最多保留行数}``（v3 病理切块：tile_dropout 与 max_tiles）；
    - roi_dropout：以该概率随机丢弃一个角色（只在 ``roi_roles`` 范围内，默认全部角色；至少保留一个）；
    - modality_dropout（v3）：``groups`` 为 {角色: 模态}；两种模态都有时，以该概率随机遮住其中一种模态的全部角色，
      永远不会两种都遮。
    """

    def __init__(self, sids: Sequence[str], stores: Sequence, train: bool = False, slice_dropout: float = 0.0,
                 roi_dropout: float = 0.0, groups: Optional[Dict[str, str]] = None, modality_dropout: float = 0.0,
                 roi_roles: Optional[Sequence[str]] = None, role_aug: Optional[Dict[str, dict]] = None):
        self.sids = list(sids)
        self.stores = list(stores)
        self.train = train
        self.slice_dropout = float(slice_dropout)
        self.roi_dropout = float(roi_dropout)
        self.groups = dict(groups or {})
        self.modality_dropout = float(modality_dropout)
        self.roi_roles = None if roi_roles is None else set(roi_roles)
        self.role_aug = dict(role_aug or {})

    def __len__(self):
        return len(self.sids)

    def _subsample(self, r: str, f: np.ndarray) -> np.ndarray:
        aug = self.role_aug.get(r)
        frac = float(aug.get("dropout", 0.0)) if aug is not None else self.slice_dropout
        max_n = aug.get("max_n") if aug is not None else None
        n = len(f)
        keep = n
        if frac > 0 and n > 1:
            keep = max(1, int(round(n * (1 - frac))))
        if max_n:
            keep = min(keep, int(max_n))
        if keep < n:
            f = f[np.sort(np.random.permutation(n)[:keep])]
        return f

    def __getitem__(self, i):
        sid = self.sids[i]
        feats: Dict[str, np.ndarray] = {}
        for st in self.stores:
            feats.update(st.load(sid))
        if self.train:
            if self.slice_dropout > 0 or self.role_aug:
                feats = {r: self._subsample(r, f) for r, f in feats.items()}
            if self.modality_dropout > 0 and self.groups:
                mods = sorted({self.groups[r] for r in feats if r in self.groups})
                if len(mods) >= 2 and np.random.rand() < self.modality_dropout:
                    drop = mods[np.random.randint(len(mods))]
                    feats = {k: v for k, v in feats.items() if self.groups.get(k) != drop}
            cand = [r for r in feats if self.roi_roles is None or r in self.roi_roles]
            if self.roi_dropout > 0 and len(cand) >= 2 and np.random.rand() < self.roi_dropout:
                drop = cand[np.random.randint(len(cand))]
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
