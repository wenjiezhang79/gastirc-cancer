"""逐病例处理：单例失败不影响其他病例，失败信息去除路径后记录。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from .errors import CaseError, UserError
from .privacy import scrub

PATH_COLUMNS = ("slide_path", "ct_dir", "tumor_mask_path")


def _sensitive_values(case: dict) -> List[str]:
    out = []
    for k, v in case.items():
        if k in PATH_COLUMNS or k.endswith("_path") or k.endswith("_dir"):
            if v:
                out.append(str(v))
    return out


def _safe_call(func: Callable, case: dict, cfg) -> dict:
    sid = case.get("study_id", "?")
    t0 = time.time()
    try:
        res = func(case, cfg) or {}
        res.setdefault("status", "ok")
    except (CaseError, UserError) as e:
        res = {"status": "failed", "message": scrub(str(e), _sensitive_values(case))}
    except Exception as e:
        res = {
            "status": "failed",
            "message": scrub(f"{type(e).__name__}: {e}", _sensitive_values(case)),
            "_traceback": scrub(traceback.format_exc(), _sensitive_values(case)),
        }
    res["study_id"] = sid
    res["elapsed_s"] = round(time.time() - t0, 2)
    return res


STATUS_ZH = {"ok": "完成", "skipped": "跳过", "failed": "失败", "warning": "完成（有警告）"}


def process_cases(cases: List[dict], func: Callable, cfg, logger: logging.Logger,
                  num_workers: int = 1, desc: str = "处理") -> List[dict]:
    """对每个病例调用 ``func(case, cfg) -> dict``；返回每例结果（含 status/message）。"""
    n = len(cases)
    results: Dict[str, dict] = {}
    logger.info("%s：共 %d 例，并行进程数 %d", desc, n, max(1, num_workers))

    def _log(i: int, res: dict) -> None:
        st = res.get("status", "ok")
        msg = res.get("message", "")
        line = f"[{i}/{n}] {res['study_id']} {STATUS_ZH.get(st, st)}（{res['elapsed_s']}s）"
        if msg:
            line += f"：{msg}"
        if st == "failed":
            logger.warning(line)
            if res.get("_traceback"):
                logger.debug("%s 调用栈：\n%s", res["study_id"], res["_traceback"])
        else:
            logger.info(line)

    if num_workers <= 1 or n <= 1:
        for i, case in enumerate(cases, 1):
            res = _safe_call(func, case, cfg)
            results[res["study_id"]] = res
            _log(i, res)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_safe_call, func, case, cfg) for case in cases]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results[res["study_id"]] = res
                _log(i, res)

    ordered = [results[c.get("study_id", "?")] for c in cases if c.get("study_id", "?") in results]
    for r in ordered:
        r.pop("_traceback", None)
    summarize(ordered, logger, desc)
    return ordered


def summarize(results: List[dict], logger: logging.Logger, desc: str = "处理") -> None:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.get("status", "ok")] = counts.get(r.get("status", "ok"), 0) + 1
    parts = [f"{STATUS_ZH.get(k, k)} {v}" for k, v in counts.items()]
    logger.info("%s结束：%s", desc, "，".join(parts) if parts else "无病例")
    failed = [r["study_id"] for r in results if r.get("status") == "failed"]
    if failed:
        logger.warning(
            "失败病例（%d）：%s。原因见上方日志或 status.csv；修正后可只重跑这些病例：--study-ids %s",
            len(failed), ", ".join(failed[:50]) + (" ..." if len(failed) > 50 else ""),
            ",".join(failed[:50]),
        )


def exit_code(results: List[dict]) -> int:
    """全部成功返回 0；有失败病例返回 3（便于脚本判断）。"""
    return 3 if any(r.get("status") == "failed" for r in results) else 0


def resolve_workers(args, cfg) -> int:
    n: Optional[int] = getattr(args, "num_workers", None)
    if n is None:
        n = int(cfg.get("num_workers", 1) or 1)
    return max(1, n)
