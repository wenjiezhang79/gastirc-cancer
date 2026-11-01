"""三方共用的内层验证名单（早停用的"模拟考"病例）。

规则（docs/decisions.md 第十四节第 5 小节，2026-11-01 与研究者确认）：
- 对每个（随机种子 s, 留出折 k），在**整个内部训练集**中除第 k 折以外的病例里（不看有没有病理 / CT 特征），
  按 ``split.stratify_by`` 分层抽 ``split.inner_val_fraction``（默认 15%）作内层验证；
- v1 病理、v2 CT、v3 融合都读同一份名单（各取自己有特征的那部分）。这样融合模型的内层验证病例，
  初始化用的两个单模态模型都没有训练过，早停判断更可靠，且训练人数不减少；
- 名单第一次使用时生成并保存为 ``<work_dir>/splits/inner_val/split_seed<划分种子>_frac<比例>_s<s>.csv``
  （列：heldout_fold, study_id），之后每次使用都与重新计算的结果核对，不一致（划分文件被重新生成等）时报错。

留出折只用于折外预测，不参与任何训练决策；内层验证名单只影响"训练什么时候停"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from .config import Config
from .paths import WorkPaths
from .split import _stratum_keys, greedy_assign
from .utils.errors import UserError

DEFAULT_FRACTION = 0.15


def inner_val_fraction(cfg: Config) -> float:
    """共用比例 split.inner_val_fraction；旧位置（v1/v2/v3.train.inner_val_fraction）不再使用，写了且不同则报错。"""
    frac = float(cfg.get_path("split.inner_val_fraction", DEFAULT_FRACTION))
    if not 0.0 < frac < 0.5:
        raise UserError(f"split.inner_val_fraction = {frac} 不合理", "内层验证比例应在 0–0.5 之间，默认 0.15。")
    for sec in ("v1", "v2", "v3"):
        old = cfg.get_path(f"{sec}.train.inner_val_fraction")
        if old is not None and abs(float(old) - frac) > 1e-12:
            raise UserError(f"配置 {sec}.train.inner_val_fraction = {old} 与 split.inner_val_fraction = {frac} 不同",
                            "内层验证名单现在由 v1 / v2 / v3 三方共用，比例只在 split.inner_val_fraction 设置；"
                            f"请删除 {sec}.train.inner_val_fraction 这一行。")
    return frac


class SharedInnerVal:
    """train_df：内部训练集全部病例（含 study_id、fold 与分层所需的标签列），即 set == train 的行。"""

    def __init__(self, cfg: Config, train_df: pd.DataFrame, logger=None):
        if "fold" not in train_df or "study_id" not in train_df:
            raise ValueError("train_df 需要 study_id 与 fold 列")
        if (train_df.get("set", pd.Series(["train"] * len(train_df))) != "train").any():
            raise ValueError("SharedInnerVal 只接受内部训练集（set == train）")
        self.cfg = cfg
        self.frac = inner_val_fraction(cfg)
        self.by = list(cfg.split.stratify_by)
        # 按 study_id 排序：结果不依赖清单中的行顺序
        self.df = train_df.sort_values("study_id", kind="mergesort").reset_index(drop=True)
        self.logger = logger
        split_seed = cfg.split.get("seed", cfg.get("seed"))
        self.dir = WorkPaths(cfg).splits / "inner_val"
        self.prefix = f"split_seed{split_seed}_frac{self.frac:g}"
        self._cache: Dict[int, Dict[int, Set[str]]] = {}

    def path(self, seed: int) -> Path:
        return self.dir / f"{self.prefix}_s{int(seed)}.csv"

    def compute(self, seed: int) -> Dict[int, Set[str]]:
        out: Dict[int, Set[str]] = {}
        for k in sorted(int(f) for f in self.df["fold"].unique()):
            rest = self.df[self.df["fold"] != k].reset_index(drop=True)
            keys = _stratum_keys(rest, self.by)
            grp = greedy_assign(keys, [1 - self.frac, self.frac], np.random.RandomState(int(seed) * 100 + k))
            out[k] = set(rest.loc[grp == 1, "study_id"].astype(str))
        return out

    def _load_or_create(self, seed: int) -> Dict[int, Set[str]]:
        if seed in self._cache:
            return self._cache[seed]
        cur = self.compute(seed)
        p = self.path(seed)
        if p.exists():
            old = pd.read_csv(p, dtype={"study_id": str})
            saved = {int(k): set(g["study_id"]) for k, g in old.groupby("heldout_fold")}
            if saved != cur:
                raise UserError(
                    f"共用内层验证名单 {p.name} 与按当前划分重新计算的结果不一致（划分文件或分层设置可能被改动过）",
                    "若确实重新划分了数据，已训练的 v1 / v2 / v3 实验都需要用新实验名重跑；"
                    f"确认后删除 {p.parent.name}/{p.name}，再次运行会生成新名单。")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"heldout_fold": k, "study_id": s} for k in sorted(cur) for s in sorted(cur[k])]
            tmp = p.with_suffix(".tmp")
            pd.DataFrame(rows, columns=["heldout_fold", "study_id"]).to_csv(tmp, index=False)
            tmp.replace(p)
            if self.logger is not None:
                self.logger.info("已生成共用内层验证名单：%s（v1 / v2 / v3 共用）", p.name)
        self._cache[seed] = cur
        return cur

    def ids(self, seed: int, fold: int) -> Set[str]:
        d = self._load_or_create(int(seed))
        if int(fold) not in d:
            raise UserError(f"内层验证名单中没有第 {fold} 折", "请检查划分文件的折数。")
        return d[int(fold)]

    def mask(self, rest: pd.DataFrame, seed: int, fold: int) -> np.ndarray:
        """rest：某模态病例池中除第 fold 折以外的病例。返回布尔数组：True = 内层验证。"""
        return rest["study_id"].astype(str).isin(self.ids(seed, fold)).to_numpy()
