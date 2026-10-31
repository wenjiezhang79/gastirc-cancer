"""命令行公共参数与统一的错误处理。"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, List, Optional, Sequence

from .errors import UserError
from .log import close_logging, get_logger
from .privacy import scrub


def enforce_offline() -> None:
    """运行时禁止联网：让 Hugging Face / timm 等库只用本地文件。"""
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"
    os.environ.setdefault("TORCH_HOME", os.path.join(os.getcwd(), ".torch_offline_home"))


def base_parser(description: str, cases: bool = True, workers: bool = True, gpu: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="配置文件路径，例如 configs/server.yaml")
    p.add_argument(
        "--set", action="append", default=[], metavar="键=值",
        help="临时覆盖配置项，可重复，例如 --set ct.target_spacing=[0.8,0.8,5.0]",
    )
    p.add_argument("--run-name", default=None, help="运行名称（附加在输出目录名后，便于区分）")
    if cases:
        p.add_argument("--limit", type=int, default=None, help="只处理前 N 例（冒烟测试用，如 --limit 5）")
        p.add_argument("--study-ids", default=None, help="只处理指定病例，逗号分隔，如 C1-0001,C1-0002")
        p.add_argument("--overwrite", action="store_true", help="已有结果也重新处理（默认跳过已完成的病例）")
    if workers:
        p.add_argument("--num-workers", type=int, default=None, help="并行进程数（默认取配置 num_workers）")
    if gpu:
        p.add_argument("--gpu", type=int, default=None, help="使用第几张 GPU（0 或 1）；不填则用 CPU")
    return p


def run_main(func: Callable[[argparse.Namespace], int], parser: argparse.ArgumentParser,
             argv: Optional[Sequence[str]] = None) -> int:
    """解析参数并运行；把异常转换成研究者能看懂的提示。返回退出码。"""
    enforce_offline()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logger = get_logger()
    try:
        code = func(args)
        return int(code or 0)
    except UserError as e:
        msg = f"【需要处理】{e.message}"
        if e.hint:
            msg += f"\n  → 怎么办：{e.hint}"
        if logger.handlers:
            logger.error(msg)
        else:
            print(msg, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已手动中断。", file=sys.stderr)
        return 130
    except Exception as e:  # 代码 bug 或未预料的问题
        tb = scrub(traceback.format_exc())
        msg = (
            f"【程序内部错误】{type(e).__name__}: {scrub(str(e))}\n"
            "  → 怎么办：这很可能是代码问题，不是你的操作问题。请把运行目录下 run.log 的最后 80 行"
            "（确认不含姓名/住院号后）贴回来。"
        )
        if logger.handlers:
            logger.debug("调用栈：\n%s", tb)
            logger.error(msg)
        else:
            print(tb, file=sys.stderr)
            print(msg, file=sys.stderr)
        return 2
    finally:
        close_logging()


def parse_id_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]
