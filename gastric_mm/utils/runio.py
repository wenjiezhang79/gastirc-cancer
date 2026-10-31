"""运行目录：每次运行在 <output_dir>/<步骤>/<时间戳>[_名称]/ 下保存
config.yaml（实际使用的配置）、run.log、env.json（环境信息）、status.csv（每例状态）和报告。
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import Config, resolve_path, save_config
from .log import setup_logging

KEY_PACKAGES = (
    "numpy", "pandas", "scipy", "PyYAML", "pydicom", "SimpleITK", "h5py", "tifffile",
    "Pillow", "matplotlib", "openslide-python", "openslide-bin", "torch", "timm",
)

PRIVATE_README = (
    "本目录中的文件可能包含原始文件名（可能带住院号或姓名），只供在服务器上核对，\n"
    "不要拷出服务器、不要贴给任何人。\n"
)


def package_versions(names: Iterable[str] = KEY_PACKAGES) -> dict:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = None
    return out


def git_commit() -> Optional[str]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            commit = r.stdout.strip()
            d = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if d.returncode == 0 and d.stdout.strip():
                commit += "（有未提交的修改）"
            return commit
    except Exception:
        return None
    return None


class Run:
    """一次运行。"""

    def __init__(self, cfg: Config, step: str, run_name: Optional[str] = None):
        self.cfg = cfg
        self.step = step
        base = resolve_path(cfg, cfg.paths.output_dir) / step
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = ts + (f"_{run_name}" if run_name else "")
        run_dir = base / name
        i = 1
        while run_dir.exists():
            i += 1
            run_dir = base / f"{name}-{i}"
        run_dir.mkdir(parents=True)
        self.dir = run_dir
        self.logger = setup_logging(run_dir / "run.log")
        save_config(cfg, run_dir / "config.yaml")
        env = {
            "step": step,
            "time": ts,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "seed": cfg.get("seed"),
            "argv": sys.argv[1:],
            "packages": package_versions(),
        }
        with open(run_dir / "env.json", "w", encoding="utf-8") as f:
            json.dump(env, f, ensure_ascii=False, indent=2)
        self.logger.info("运行目录：%s", run_dir)

    @property
    def private_dir(self) -> Path:
        d = self.dir / "private"
        if not d.exists():
            d.mkdir(parents=True)
            (d / "README_不要外传.txt").write_text(PRIVATE_README, encoding="utf-8")
        return d

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_table(self, rows: List[dict], name: str, columns: Optional[List[str]] = None) -> Path:
        path = self.dir / name
        write_csv(rows, path, columns)
        return path

    def write_json(self, obj, name: str) -> Path:
        path = self.dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        return path

    def write_text(self, text: str, name: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path


def write_csv(rows: List[dict], path: Path, columns: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = [k for k in ("study_id", "status", "message") if any(k in r for r in rows)]
        for r in rows:
            for k in r:
                if k not in columns and not k.startswith("_"):
                    columns.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    return v


def atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)
