"""外部工具调用公共层（TotalSegmentator、nnU-Net 等装在独立 Python 环境中的工具）。

- 命令来自配置：可以是一个可执行文件（建议写独立环境中的绝对路径），也可以是命令列表
  （如 ``[/opt/envs/x/bin/python, -m, some.module]``）；
- 子进程环境：继承当前环境，并设置离线开关（HF_HUB_OFFLINE 等）、权重目录等环境变量；
  ``--gpu`` 通过 CUDA_VISIBLE_DEVICES 传给子进程；
- 子进程的输出可能含有文件路径，失败信息写入日志前统一去除路径（只保留最后若干行）。
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ..utils.errors import CaseError, UserError
from ..utils.privacy import scrub

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def resolve_command(cmd: Union[str, Sequence[str], None], what: str, cfg_key: str) -> List[str]:
    """把配置中的命令解析为参数列表，并确认可执行文件存在。"""
    if cmd is None or cmd == "" or (isinstance(cmd, (list, tuple)) and not cmd):
        raise UserError(f"没有配置 {what} 的命令", f"请在配置文件 {cfg_key} 中填写（独立环境中的可执行文件绝对路径）。")
    parts = list(cmd) if isinstance(cmd, (list, tuple)) else shlex.split(str(cmd))
    parts = [str(p) for p in parts]
    exe = parts[0]
    if os.sep in exe or exe.startswith("."):
        if not Path(exe).exists():
            raise UserError(f"找不到 {what} 的可执行文件（{cfg_key}）",
                            f"请确认 {what} 的独立环境已安装，并把 {cfg_key} 改成正确的绝对路径。")
    else:
        found = shutil.which(exe)
        if not found:
            raise UserError(f"在当前环境中找不到命令 {exe}（{what}）",
                            f"{what} 装在独立的 Python 环境中：请在 {cfg_key} 填写该环境中可执行文件的绝对路径，"
                            f"如 /opt/envs/xxx/bin/{exe}。")
        parts[0] = found
    return parts


def tool_env(extra: Optional[Dict[str, str]] = None, gpu: Optional[int] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(OFFLINE_ENV)
    for k, v in (extra or {}).items():
        if v is not None:
            env[k] = str(v)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_tool(args: Sequence[str], env: Dict[str, str], timeout: Optional[float], what: str,
             sensitive: Iterable[str] = (), cwd: Optional[Path] = None, tail: int = 15) -> str:
    """运行外部命令；失败时抛出 CaseError（信息去除路径）。返回标准输出（已去除路径）。"""
    sens = [str(s) for s in sensitive if s]
    try:
        r = subprocess.run(list(args), env=env, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CaseError(f"{what} 运行超时（{timeout:g} 秒）") from None
    except OSError as e:
        raise CaseError(f"{what} 无法启动：{scrub(str(e), sens)}") from None
    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()[-tail:]
        msg = scrub("\n".join(lines), sens)
        raise CaseError(f"{what} 运行失败（退出码 {r.returncode}）：{msg}")
    return scrub(r.stdout or "", sens)


def describe(args: Sequence[str]) -> str:
    """日志中显示的命令（只显示程序名与参数名，不显示路径）。"""
    out = [Path(args[0]).name]
    for a in args[1:]:
        out.append(a if a.startswith("-") or os.sep not in a else "<路径>")
    return " ".join(out)
