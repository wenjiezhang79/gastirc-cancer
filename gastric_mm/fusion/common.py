"""v3 公共函数：特征仓库、模态可用性、单模态来源实验（v1 / v2）、分支初始化与防泄漏检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config, resolve_path
from ..mtl.data import CTFeatureStore, MaskedStore, PathFeatureStore
from ..utils.errors import UserError

SECTION, NAMESPACE = "v3", "fusion"
GROUPS = ("both", "path_only", "ct_only")
GROUP_NAMES = {"both": "两种都有", "path_only": "只有病理", "ct_only": "只有 CT"}
PATH_BRANCH_KEYS = ("aggregator", "hidden_dim", "attn_dim", "n_branches", "dropout", "transmil_heads")
CT_BRANCH_KEYS = ("aggregator", "hidden_dim", "attn_dim", "dropout")


# ---------------- 特征仓库与可用性 ----------------
def make_stores(cfg: Config):
    sec = cfg[SECTION]
    cache = bool(sec.get("feature_cache", True))
    rois = list(sec.get("ct_rois", ["tumor", "regional"]) or [])
    pstore = PathFeatureStore(cfg, list(sec.get("path_encoders") or []), cache=cache)
    cstore = CTFeatureStore(cfg, str(sec.get("ct_encoder") or ""), rois, cache=cache)
    return pstore, cstore


def availability(sids: Sequence[str], pstore: PathFeatureStore, cstore: CTFeatureStore) -> pd.DataFrame:
    """每例：has_path、has_<ROI>（如 has_tumor、has_regional）、has_ct、group（both / path_only / ct_only / none）。"""
    sids = list(sids)
    hp = set(pstore.available(sids))
    hc = set(cstore.available(sids))
    rows = []
    for s in sids:
        r = {"study_id": s, "has_path": int(s in hp)}
        rois = cstore.load(s) if s in hc else {}
        for role in cstore.roles:
            r[f"has_{role}"] = int(role in rois)
        r["has_ct"] = int(bool(rois))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["group"] = np.select([(df.has_path == 1) & (df.has_ct == 1), df.has_path == 1, df.has_ct == 1],
                            ["both", "path_only", "ct_only"], "none")
    return df


def group_of(df: pd.DataFrame) -> pd.Series:
    """由预测表中的 has_* 列得到模态可用性分组。"""
    has_ct = np.zeros(len(df), dtype=bool)
    for c in df.columns:
        if c.startswith("has_") and c != "has_path":
            has_ct |= df[c].fillna(0).to_numpy() > 0.5
    has_p = df["has_path"].fillna(0).to_numpy() > 0.5 if "has_path" in df else np.zeros(len(df), bool)
    return pd.Series(np.select([has_p & has_ct, has_p, has_ct], ["both", "path_only", "ct_only"], "none"),
                     index=df.index)


def wrap_stores(pstore, cstore, avail: pd.DataFrame, hide: Sequence[str] = ()) -> List:
    """只对有该模态特征的病例读取（其余病例按缺失处理）；hide 用于缺失鲁棒性分析。"""
    p_ids = list(avail.loc[avail.has_path == 1, "study_id"])
    c_ids = list(avail.loc[avail.has_ct == 1, "study_id"])
    return [MaskedStore(pstore, hide=[r for r in hide if r in pstore.roles], only=p_ids),
            MaskedStore(cstore, hide=[r for r in hide if r in cstore.roles], only=c_ids)]


# ---------------- 单模态来源实验 ----------------
class SourceExp:
    """v1（kind=path，命名空间 path_mtl）或 v2（kind=ct，命名空间 ct_mtl）的一个已训练实验。"""

    def __init__(self, cfg: Config, kind: str, name: str):
        self.kind = kind
        self.name = name
        self.ns = "path_mtl" if kind == "path" else "ct_mtl"
        self.label = "v1 病理" if kind == "path" else "v2 CT"
        self.dir = resolve_path(cfg, cfg.paths.output_dir) / self.ns / name
        if not (self.dir / "experiment.json").exists():
            raise UserError(f"找不到{self.label}实验 {name}（{self.ns}/{name}）",
                            f"请确认 v3.init.{kind}_exp 填写的是已训练完成的{self.label}实验名，且使用同一个配置文件（输出目录相同）。")
        self.meta = json.loads((self.dir / "experiment.json").read_text(encoding="utf-8"))
        if kind == "ct" and self.meta.get("kind", "token") != "token":
            raise UserError(f"v2 实验 {name} 是影像组学基线，不能用于初始化 CT 分支", "请填写 CT 深度模型实验名。")

    @property
    def fingerprint(self) -> str:
        return str(self.meta.get("fingerprint"))

    def fold_dir(self, seed: int, fold: int) -> Path:
        d = self.dir / f"seed{seed}" / f"fold{fold}"
        if not (d / "done.json").exists() or not (d / "model.pt").exists():
            raise UserError(f"{self.label}实验 {self.name} 没有种子 {seed}、第 {fold} 折的模型",
                            f"融合模型的种子 {seed} 需要{self.label}实验中同种子、同折的模型做初始化（不会悄悄换种子）。"
                            f"请先用该种子训练{self.label}实验，或把 v3.seeds 改为{self.label}实验已有的种子。")
        return d

    def model_obj(self, seed: int, fold: int) -> dict:
        p = self.fold_dir(seed, fold) / "model.pt"
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")
        want = "gastric_mm.path_model.v1" if self.kind == "path" else "gastric_mm.token_model.v2"
        if obj.get("format") != want:
            raise UserError(f"{self.label}实验 {self.name} 的模型文件格式不对（{obj.get('format')}）",
                            "请确认填写的实验名正确。")
        return obj

    def case_ids(self, seed: int, fold: int) -> dict:
        return json.loads((self.fold_dir(seed, fold) / "case_ids.json").read_text(encoding="utf-8"))

    def branch_cfg(self, seed: int, fold: int) -> dict:
        mc = self.model_obj(seed, fold)["model_cfg"]
        keys = PATH_BRANCH_KEYS if self.kind == "path" else CT_BRANCH_KEYS
        return {k: mc[k] for k in keys if k in mc}


def check_source_fold(src: SourceExp, seed: int, fold: int, heldout: Set[str], fold_ids: Set[str],
                      shared_val: Set[str]) -> None:
    """防泄漏与一致性检查（必须）：
    - 来源模型的 train / inner_val 中不得出现融合模型本折的留出病例；
    - 来源模型的留出病例必须都属于当前划分的第 fold 折（折号一致）；
    - 来源模型的内层验证必须来自三方共用名单（且共用名单中的病例不在其训练部分）。
    """
    c = src.case_ids(seed, fold)
    tr, va, ho = set(c["train"]), set(c["inner_val"]), set(c["heldout"])
    leak = (tr | va) & heldout
    if leak:
        raise UserError(f"防泄漏检查失败：{src.label}实验 {src.name} 种子 {seed} 第 {fold} 折的训练部分包含 {len(leak)} 例"
                        f"融合模型本折的留出病例（如 {sorted(leak)[0]}）",
                        "两者的折号不一致（可能使用了不同的划分文件）。请确认 v1 / v2 / v3 使用同一个划分文件，"
                        "必要时用当前划分重新训练该单模态实验。")
    if not ho <= fold_ids:
        raise UserError(f"{src.label}实验 {src.name} 第 {fold} 折的留出病例与当前划分的第 {fold} 折不一致",
                        "请确认 v1 / v2 / v3 使用同一个划分文件（work/splits/），必要时重新训练该单模态实验。")
    if not va <= shared_val or (tr & shared_val):
        raise UserError(f"{src.label}实验 {src.name} 种子 {seed} 第 {fold} 折的内层验证名单与三方共用名单不一致",
                        "该实验可能是用旧版代码（各自抽内层验证）训练的。请用当前代码重新训练该单模态实验（换一个实验名）。")


def check_source_encoders(src: SourceExp, seed: int, fold: int, cfg: Config, dims: Dict[str, int]) -> None:
    obj = src.model_obj(seed, fold)
    ex = obj.get("extra", {})
    sec = cfg[SECTION]
    if src.kind == "path":
        want = list(sec.path_encoders)
        if list(ex.get("encoders", [])) != want:
            raise UserError(f"v1 实验 {src.name} 使用的病理编码器 {ex.get('encoders')} 与 v3.path_encoders {want} 不一致",
                            "融合模型必须使用与初始化模型相同的病理特征。")
        if int(obj["in_dim"]) != int(dims["path"]):
            raise UserError(f"v1 实验 {src.name} 的病理特征维数 {obj['in_dim']} 与当前特征 {dims['path']} 不一致",
                            "请检查病理特征目录是否被重新导入或混入了其他编码器。")
    else:
        if str(ex.get("encoder")) != str(sec.ct_encoder):
            raise UserError(f"v2 实验 {src.name} 使用的 CT 编码器 {ex.get('encoder')} 与 v3.ct_encoder {sec.ct_encoder} 不一致",
                            "融合模型必须使用与初始化模型相同的 CT 特征。")
        miss = [r for r in sec.ct_rois if r not in obj["in_dims"]]
        if miss:
            raise UserError(f"v2 实验 {src.name} 没有 ROI {miss} 的分支", "请让 v3.ct_rois 是该 v2 实验 ROI 的子集。")
        for r in sec.ct_rois:
            if int(obj["in_dims"][r]) != int(dims[r]):
                raise UserError(f"v2 实验 {src.name} 的 {r} 特征维数 {obj['in_dims'][r]} 与当前特征 {dims[r]} 不一致",
                                "请检查 CT 特征目录是否被重新提取。")


def load_branch_weights(model, src: SourceExp, seed: int, fold: int) -> int:
    """把来源模型的分支参数拷入融合模型；返回拷入的张量数。v1：aggregator.* → branch.path.*；
    v2：branches.<ROI>.* → branch.<ROI>.*。"""
    sd = src.model_obj(seed, fold)["state_dict"]
    own = model.state_dict()
    new = {}
    if src.kind == "path":
        for k, v in sd.items():
            if k.startswith("aggregator."):
                new["branch.path." + k[len("aggregator."):]] = v
        roles = ["path"]
    else:
        roles = [r for r in model.roles if r != "path"]
        for k, v in sd.items():
            if k.startswith("branches."):
                role = k.split(".")[1]
                if role in roles:
                    new["branch." + k[len("branches."):]] = v
    want = [k for k in own if any(k.startswith(f"branch.{r}.") for r in roles)]
    missing = [k for k in want if k not in new]
    bad = [k for k in new if k not in own or own[k].shape != new[k].shape]
    if missing or bad:
        raise UserError(f"{src.label}实验 {src.name} 的分支结构与融合模型不一致（缺 {len(missing)} 个、不匹配 {len(bad)} 个参数）",
                        "这通常是程序问题，请把日志贴回来。")
    model.load_state_dict(new, strict=False)
    return len(new)


def sources_from_cfg(cfg: Config) -> Dict[str, Optional[SourceExp]]:
    init = cfg[SECTION].get("init", {}) or {}
    out = {}
    for kind in ("path", "ct"):
        name = init.get(f"{kind}_exp")
        out[kind] = SourceExp(cfg, kind, str(name)) if name else None
    return out


def branch_config(cfg: Config, sources: Dict[str, Optional[SourceExp]], seed: int) -> Tuple[dict, dict]:
    """融合模型的完整结构配置：分支结构从来源模型文件读取（初始化时），否则用 v3.model.path / ct。"""
    mc = cfg[SECTION].model.to_dict()
    for kind in ("path", "ct"):
        src = sources.get(kind)
        if src is not None:
            mc[kind] = src.branch_cfg(seed, 0)
    return mc, {k: (s.name if s else None) for k, s in sources.items()}
