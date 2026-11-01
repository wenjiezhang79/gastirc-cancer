"""日志：同时输出到终端和运行目录下的 run.log。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

LOGGER_NAME = "gastric_mm"


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """配置日志。重复调用会先移除旧的处理器（便于测试中多次运行）。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:  # pragma: no cover
            pass
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # 文件中保留更详细的调试信息
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def close_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:  # pragma: no cover
            pass
