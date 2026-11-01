"""YAML 配置系统。

- 支持继承：配置文件里写 ``_base_: default.yaml``（相对于本文件所在目录），先加载基础配置再覆盖。
- 支持命令行临时覆盖：``--set ct.target_spacing=[0.8,0.8,5.0] --set seed=1``。
- 相对路径：``paths.root`` 若为相对路径，则相对于**项目根目录**（本仓库根目录）；
  其余路径若为相对路径，则相对于 ``paths.root``。用 :func:`resolve_path` 统一解析。
- 每次运行把合并后的最终配置保存到运行目录（见 ``utils/runio.py``）。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import yaml

from .utils.errors import UserError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class Config(dict):
    """可以用点号访问的字典：``cfg.ct.target_spacing`` 等价于 ``cfg["ct"]["target_spacing"]``。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"配置中缺少 '{key}' 这一项。请对照 configs/default.yaml 检查拼写或补充该项。"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按 ``a.b.c`` 取值，不存在时返回 default。"""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> dict:
        return _to_plain(self)


def _to_config(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, Config):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, Config):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的字典与 base 合并，其他类型直接替换。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise UserError(
            f"找不到配置文件：{path}",
            "请检查 --config 参数。服务器上一般先复制 configs/server_example.yaml 为 configs/server.yaml 再修改。",
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise UserError(
            f"配置文件 {path.name} 的 YAML 格式有误：{e}",
            "常见原因：缩进用了 Tab、冒号后缺空格、中文引号。请按报错的行号修改。",
        ) from None
    if not isinstance(data, dict):
        raise UserError(f"配置文件 {path.name} 的顶层必须是键值对。")
    return data


def _load_with_bases(path: Path, _seen: Optional[set] = None) -> dict:
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise UserError(f"配置文件出现循环继承：{path.name}")
    _seen.add(path)
    data = _read_yaml(path)
    bases = data.pop("_base_", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        bp = Path(b)
        if not bp.is_absolute():
            bp = path.parent / bp
        merged = deep_merge(merged, _load_with_bases(bp, _seen))
    return deep_merge(merged, data)


def parse_override(item: str) -> tuple:
    if "=" not in item:
        raise UserError(f"--set 参数格式应为 键=值，例如 --set seed=1，收到：{item}")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise UserError(f"--set 参数缺少键名：{item}")
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return key, parsed


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, value = parse_override(item)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


REQUIRED_KEYS = ("paths.root", "paths.manifest", "paths.work_dir", "paths.output_dir")


def load_config(path: Union[str, Path], overrides: Optional[Iterable[str]] = None) -> Config:
    """加载配置文件（含继承与覆盖），并做基本校验。"""
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path)
    data = _load_with_bases(path)
    data = apply_overrides(data, overrides)
    cfg = _to_config(data)
    for key in REQUIRED_KEYS:
        if cfg.get_path(key) in (None, ""):
            raise UserError(
                f"配置缺少必填项 {key}",
                f"请在配置文件中填写 {key}（可参考 configs/default.yaml 中的说明）。",
            )
    cfg["_meta"] = Config({"config_path": str(path.resolve())})
    return cfg


def project_root() -> Path:
    return PROJECT_ROOT


def root_dir(cfg: Config) -> Path:
    root = Path(os.path.expanduser(str(cfg.paths.root)))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def resolve_path(cfg: Config, value: Optional[Union[str, Path]]) -> Optional[Path]:
    """把配置或清单中的路径解析为绝对路径（相对路径以 paths.root 为基准）。空值返回 None。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = root_dir(cfg) / p
    return p


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    data = cfg.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def require(cfg: Config, dotted: str) -> Any:
    """取必填配置项，缺失时给出清楚的提示。"""
    v = cfg.get_path(dotted)
    if v is None or v == "":
        raise UserError(f"配置缺少 {dotted}", f"请在配置文件中填写 {dotted}。")
    return v
