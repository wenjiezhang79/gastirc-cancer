"""任务定义：模型要预测哪些任务、每个任务的类型。

由配置 ``v1.tasks`` 生成，并对照 labels.py 的标签列检查名称是否有效。
保存在模型文件中，预测时不依赖原配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..config import Config
from ..labels import task_definitions
from ..utils.errors import UserError

POPULATIONS = ["resected", "unresectable"]


@dataclass
class TaskSpec:
    multiclass: Dict[str, int] = field(default_factory=dict)   # 名称 → 类别数
    binary: List[str] = field(default_factory=list)
    regression: List[str] = field(default_factory=list)
    survival: List[str] = field(default_factory=list)          # 终点：os / dfs / pfs
    # 这些终点按人群（手术 / 不可切除）分别设置基线风险，共享风险评分
    population_baselines: List[str] = field(default_factory=lambda: ["os"])

    @property
    def all_tasks(self) -> List[str]:
        return list(self.multiclass) + list(self.binary) + list(self.regression) + list(self.survival)

    def n_pops(self, endpoint: str) -> int:
        return len(POPULATIONS) if endpoint in self.population_baselines else 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(**d)

    @classmethod
    def from_config(cls, cfg: Config) -> "TaskSpec":
        tc = cfg.v1.tasks
        defs = {t.name: t for t in task_definitions(cfg)}

        def check(names, kinds, what):
            out = []
            for n in names or []:
                if n not in defs or defs[n].kind not in kinds:
                    valid = [k for k, t in defs.items() if t.kind in kinds]
                    raise UserError(f"v1.tasks.{what} 中的“{n}”不是有效任务", f"可用：{', '.join(valid)}")
                out.append(n)
            return out

        mc = check(tc.get("multiclass"), ("multiclass",), "multiclass")
        spec = cls(
            multiclass={n: defs[n].n_classes for n in mc},
            binary=check(tc.get("binary"), ("binary",), "binary"),
            regression=check(tc.get("regression"), ("regression",), "regression"),
            survival=check(tc.get("survival"), ("survival",), "survival"),
            population_baselines=list(cfg.v1.get("population_baselines", ["os"]) or []),
        )
        if not spec.all_tasks:
            raise UserError("v1.tasks 中没有启用任何任务", "请至少启用一个任务。")
        return spec
