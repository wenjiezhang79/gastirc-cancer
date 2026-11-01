"""由配置中的某一节（如 v2）生成任务定义 TaskSpec（与 v1 的 TaskSpec 相同，只是读取的配置节不同）。"""

from __future__ import annotations

from ..config import Config
from ..labels import task_definitions
from ..training.task_spec import TaskSpec
from ..utils.errors import UserError


def spec_from_section(cfg: Config, section: str) -> TaskSpec:
    sec = cfg[section]
    tc = sec.tasks
    defs = {t.name: t for t in task_definitions(cfg)}

    def check(names, kind, what):
        out = []
        for n in names or []:
            if n not in defs or defs[n].kind != kind:
                valid = [k for k, t in defs.items() if t.kind == kind]
                raise UserError(f"{section}.tasks.{what} 中的“{n}”不是有效任务", f"可用：{', '.join(valid)}")
            out.append(n)
        return out

    mc = check(tc.get("multiclass"), "multiclass", "multiclass")
    spec = TaskSpec(
        multiclass={n: defs[n].n_classes for n in mc},
        binary=check(tc.get("binary"), "binary", "binary"),
        regression=check(tc.get("regression"), "regression", "regression"),
        survival=check(tc.get("survival"), "survival", "survival"),
        population_baselines=list(sec.get("population_baselines", ["os"]) or []),
    )
    if not spec.all_tasks:
        raise UserError(f"{section}.tasks 中没有启用任何任务", "请至少启用一个任务。")
    return spec
