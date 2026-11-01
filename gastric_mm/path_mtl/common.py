"""v1 实验目录与公共函数。

实验目录 ``<output_dir>/path_mtl/<实验名>/``：同一实验可以分多次、在两张 GPU 上分别运行不同的折，
结果写入同一目录。实验第一次运行时记录配置"指纹"，之后若配置中影响结果的部分被改动，
程序会拒绝继续（避免把不同设置的结果混在一起），请改用新的实验名。
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..config import Config, deep_merge, resolve_path, save_config
from ..labels import derive_labels
from ..manifest import load_manifest
from ..split import load_split
from ..utils.errors import UserError
from ..utils.log import setup_logging
from ..utils.runio import git_commit, package_versions

NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
# 不影响结果、允许在同一实验中修改的配置项
_RUNTIME_KEYS = ("seeds", "feature_cache", "save_attention", "compare", "eval", "profiles")


def apply_profile(cfg: Config, profile: Optional[str]) -> Config:
    if not profile:
        return cfg
    profs = cfg.v1.get("profiles", {}) or {}
    if profile not in profs:
        raise UserError(f"配置中没有预设 v1.profiles.{profile}", f"可用：{', '.join(profs) or '无'}")
    cfg = copy.deepcopy(cfg)
    merged = deep_merge(cfg.v1.to_dict() if hasattr(cfg.v1, "to_dict") else dict(cfg.v1), profs[profile])
    from ..config import _to_config
    cfg["v1"] = _to_config(merged)
    cfg["v1"]["_profile"] = profile
    return cfg


def fingerprint(cfg: Config) -> str:
    v1 = {k: v for k, v in cfg.v1.to_dict().items() if k not in _RUNTIME_KEYS}
    v1.get("train", {}).pop("num_workers", None)
    obj = {"v1": v1, "labels": cfg.labels.to_dict(), "split": cfg.split.to_dict(),
           "internal_center": cfg.manifest.internal_center}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exp_root(cfg: Config) -> Path:
    return resolve_path(cfg, cfg.paths.output_dir) / "path_mtl"


def open_experiment(cfg: Config, name: str, log_tag: str) -> Tuple[Path, object]:
    if not name or not NAME_RE.match(name):
        raise UserError(f"实验名“{name}”不合法", "实验名只能包含英文字母、数字、下划线、点和短横线，如 uni2h_abmil。")
    d = exp_root(cfg) / name
    meta_p = d / "experiment.json"
    fp = fingerprint(cfg)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fp:
            raise UserError(
                f"实验 {name} 已存在，但这次的配置与第一次运行时不同（影响结果的设置被改动）",
                "请换一个新的实验名（--exp），不同设置的结果不能混在同一个实验里。",
            )
    else:
        d.mkdir(parents=True, exist_ok=True)
        save_config(cfg, d / "config.yaml")
        meta = {"name": name, "fingerprint": fp, "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "encoders": list(cfg.v1.encoders), "aggregator": cfg.v1.model.aggregator,
                "profile": cfg.v1.get("_profile"), "git_commit": git_commit(), "packages": package_versions()}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = setup_logging(d / "logs" / f"{ts}_{log_tag}.log")
    logger.info("实验目录：%s", d)
    return d, logger


def load_labels_and_split(cfg: Config, logger) -> pd.DataFrame:
    """派生标签 + 划分信息（study_id, set, fold）。"""
    res = load_manifest(cfg, logger, check_paths=False)
    labels = derive_labels(res.df, cfg)
    split = load_split(cfg)
    df = labels.merge(split[["study_id", "set", "fold"]], on="study_id", how="left")
    missing = df["set"].isna().sum()
    if missing:
        raise UserError(f"有 {missing} 例在清单中但不在划分文件中（清单可能新增了病例）",
                        "请确认后重新运行 01b_split.sh --overwrite（会改变已有划分，已训练的实验需要重跑）。")
    return df


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s == "":
        return None
    return [int(x) for x in str(s).split(",") if x.strip() != ""]
