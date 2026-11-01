"""病例清单（manifest）的读取、校验与校验报告。

命令行：
    python -m gastric_mm.manifest --config configs/server.yaml

输出（运行目录下）：
- validation_report.md   中文校验报告（只含 study_id，可直接贴回）
- validation_issues.csv  每条问题一行：级别、study_id、列名、说明
- summary.json           汇总统计
- labels_preview.csv     派生后的训练标签（缺失为空）

其他模块通过 :func:`load_manifest` 读取清单：有"错误"级问题时停止并提示先运行本模块。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config, load_config, resolve_path
from .labels import derive_labels, label_summary
from .utils.cli import base_parser, run_main
from .utils.errors import UserError
from .utils.privacy import looks_like_phi_column
from .utils.runio import Run

# ---------------- 列定义 ----------------
LABEL_SOURCES = ["biopsy", "surgery", "post_neoadj"]
SOURCE_TASKS = ["lauren", "her2", "msi", "ebv", "pdl1", "cldn182", "ki67"]

# kind: id / center / enum / path_file / path_dir / int / float / binary / text
COLUMNS: "OrderedDict[str, dict]" = OrderedDict([
    ("study_id", {"kind": "id", "required": True}),
    ("center", {"kind": "center", "required": True}),
    ("slide_path", {"kind": "path_file"}),
    ("ct_dir", {"kind": "path_dir"}),
    ("ct_point_x", {"kind": "int", "min": 0}),
    ("ct_point_y", {"kind": "int", "min": 0}),
    ("ct_point_z", {"kind": "int", "min": 0}),
    ("tumor_mask_path", {"kind": "path_file"}),
    ("mask_source", {"kind": "enum", "values": ["manual", "auto_reviewed"]}),
    ("lauren", {"kind": "enum", "values": ["intestinal", "diffuse", "mixed"]}),
    ("her2_ihc", {"kind": "enum", "values": ["0", "1+", "2+", "3+"]}),
    ("her2_fish", {"kind": "enum", "values": ["pos", "neg"]}),
    ("msi", {"kind": "enum", "values": ["MSI-H", "MSS"]}),
    ("ebv", {"kind": "enum", "values": ["pos", "neg"]}),
    ("pdl1_cps", {"kind": "float", "min": 0, "warn_max": 100}),
    ("cldn182_pct", {"kind": "float", "min": 0, "max": 100}),
    ("ki67_pct", {"kind": "float", "min": 0, "max": 100}),
    ("label_source", {"kind": "enum", "values": LABEL_SOURCES}),
    ("population", {"kind": "enum", "values": ["resected", "unresectable"]}),
    ("os_months", {"kind": "float", "min": 0}),
    ("os_event", {"kind": "binary"}),
    ("dfs_months", {"kind": "float", "min": 0}),
    ("dfs_event", {"kind": "binary"}),
    ("pfs_months", {"kind": "float", "min": 0}),
    ("pfs_event", {"kind": "binary"}),
    ("neoadj", {"kind": "binary"}),
    ("immuno", {"kind": "binary"}),
    ("ctnm_stage", {"kind": "text"}),
    ("age", {"kind": "float", "min": 0, "warn_min": 18, "warn_max": 100}),
    ("sex", {"kind": "text"}),
    ("ecog", {"kind": "int", "min": 0, "max": 4}),
])

# 可选扩展列
OPTIONAL_COLUMNS: "OrderedDict[str, dict]" = OrderedDict(
    [(f"{t}_source", {"kind": "enum", "values": LABEL_SOURCES}) for t in SOURCE_TASKS]
    + [
        ("tumor_visibility", {"kind": "enum", "values": ["clear", "uncertain", "invisible"]}),
        ("borrmann", {"kind": "enum", "values": ["early", "I", "II", "III", "IV", "V"]}),
        ("ct_series_number", {"kind": "int", "min": 0}),
    ]
)

ALL_COLUMNS = OrderedDict(list(COLUMNS.items()) + list(OPTIONAL_COLUMNS.items()))
PATH_COLUMNS = [c for c, s in ALL_COLUMNS.items() if s["kind"].startswith("path")]
NUMERIC_COLUMNS = [c for c, s in ALL_COLUMNS.items() if s["kind"] in ("int", "float", "binary")]


@dataclass
class Issue:
    level: str        # error / warning
    code: str         # 问题类型，用于汇总
    study_id: str
    column: str
    message: str


@dataclass
class ValidationResult:
    df: pd.DataFrame                       # 规范化后的清单（数值列为 float，缺失为 NaN/None）
    issues: List[Issue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")


# ---------------- 读取 ----------------
def read_manifest_csv(path: Path) -> pd.DataFrame:
    """以字符串读入清单，空单元格为 ""。自动识别 UTF-8 / GBK 编码（Excel 导出的中文 CSV 常为 GBK）。"""
    if not path.exists():
        raise UserError(
            f"找不到病例清单：{path}",
            "请检查配置中的 paths.root 和 paths.manifest；清单应为 CSV 文件（可用 configs/manifest_template.csv 作模板）。",
        )
    last = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last = e
    else:
        raise UserError(f"无法识别清单的文本编码：{last}", "请用 Excel 另存为“CSV UTF-8（逗号分隔）”格式。")
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(lambda v: str(v).strip() if v is not None else "")
    return df


_ENUM_ALIASES = {
    "her2_ihc": {"0": "0", "1+": "1+", "2+": "2+", "3+": "3+", "＋": "+"},
}


def _norm_enum(col: str, v: str, values: Sequence[str]) -> Optional[str]:
    """不区分大小写匹配枚举值；返回规范写法，匹配不到返回 None。"""
    s = v.strip().replace("＋", "+")
    if col == "her2_ihc" and s in ("0.0",):
        s = "0"
    for val in values:
        if s.lower() == val.lower():
            return val
    return None


def _to_float(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x):
        return None
    return x


# ---------------- 校验 ----------------
def validate_manifest(raw: pd.DataFrame, cfg: Config, check_paths: Optional[bool] = None) -> ValidationResult:
    mc = cfg.manifest
    if check_paths is None:
        check_paths = bool(mc.get("check_paths", True))
    strict_paths = bool(mc.get("strict_paths", False))
    issues: List[Issue] = []

    def add(level, code, sid, col, msg):
        issues.append(Issue(level, code, sid or "", col or "", msg))

    # ---- 列名 ----
    cols = list(raw.columns)
    for c in cols:
        if looks_like_phi_column(c) and c not in ALL_COLUMNS:
            add("error", "phi_column", "", c,
                f"列名“{c}”疑似病人身份信息（姓名/住院号等）。清单中只能使用研究编号 study_id，请删除该列。")
    dup_cols = [c for c in set(cols) if cols.count(c) > 1]
    for c in dup_cols:
        add("error", "dup_column", "", c, f"列名“{c}”重复出现。")
    for c, spec in COLUMNS.items():
        if c not in cols:
            if spec.get("required"):
                add("error", "missing_required_column", "", c, f"缺少必需列“{c}”。")
            else:
                add("warning", "missing_column", "", c, f"缺少标准列“{c}”，该列按全部缺失处理。")
    for c in cols:
        if c not in ALL_COLUMNS and not looks_like_phi_column(c):
            add("warning", "unknown_column", "", c,
                f"未知列“{c}”，不会被使用。若含身份信息（姓名、住院号、日期等）请删除。")

    df = raw.copy()
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    if any(i.code == "missing_required_column" for i in issues):
        return ValidationResult(df=_typed(df), issues=issues, summary={})

    # ---- study_id ----
    pattern = re.compile(str(mc.get("study_id_pattern", r"^C\d+-\d+$")))
    sids = df["study_id"]
    for idx, sid in sids.items():
        if not sid:
            add("error", "empty_id", f"第{idx + 2}行", "study_id", "study_id 为空。")
        elif not pattern.match(sid):
            add("error", "bad_id", sid, "study_id",
                f"study_id 不符合格式 {pattern.pattern}（应为研究编号如 C1-0001，不能用住院号）。")
    dups = sids[sids.duplicated(keep=False) & (sids != "")]
    for sid in sorted(set(dups)):
        add("error", "dup_id", sid, "study_id", f"study_id 重复出现 {int((sids == sid).sum())} 次。")

    def rid(idx):
        return df.at[idx, "study_id"] or f"第{idx + 2}行"

    # ---- center ----
    centers = list(mc.centers)
    for idx, v in df["center"].items():
        if v not in centers:
            add("error", "bad_center", rid(idx), "center", f"center 取值“{v}”无效，应为 {centers} 之一。")

    # ---- 逐列取值 ----
    for c, spec in ALL_COLUMNS.items():
        kind = spec["kind"]
        if kind in ("id", "center", "text") or kind.startswith("path"):
            continue
        for idx, v in df[c].items():
            if v == "":
                continue
            if kind == "enum":
                nv = _norm_enum(c, v, spec["values"])
                if nv is None:
                    hint = ""
                    if c == "her2_ihc" and v in ("1", "2", "3"):
                        hint = "（请写成 1+ / 2+ / 3+）"
                    add("error", f"bad_value_{c}", rid(idx), c,
                        f"{c} 取值“{v}”无效，应为 {spec['values']} 之一{hint}。")
                else:
                    df.at[idx, c] = nv
                continue
            x = _to_float(v)
            if x is None:
                add("error", f"not_number_{c}", rid(idx), c, f"{c} 应为数字，收到“{v}”。")
                continue
            if kind in ("int", "binary") and abs(x - round(x)) > 1e-6:
                add("error", f"not_int_{c}", rid(idx), c, f"{c} 应为整数，收到“{v}”。")
                continue
            if kind == "binary" and x not in (0.0, 1.0):
                add("error", f"not_binary_{c}", rid(idx), c, f"{c} 只能是 0 或 1，收到“{v}”。")
                continue
            if "min" in spec and x < spec["min"]:
                add("error", f"below_min_{c}", rid(idx), c, f"{c}={v} 小于允许的最小值 {spec['min']}。")
            if "max" in spec and x > spec["max"]:
                add("error", f"above_max_{c}", rid(idx), c, f"{c}={v} 大于允许的最大值 {spec['max']}。")
            if "warn_max" in spec and x > spec["warn_max"]:
                add("warning", f"high_{c}", rid(idx), c, f"{c}={v} 偏大，请核实。")
            if "warn_min" in spec and x < spec["warn_min"]:
                add("warning", f"low_{c}", rid(idx), c, f"{c}={v} 偏小，请核实。")

    # ---- 一致性 ----
    max_os = float(mc.get("max_os_months", 240))
    for idx in df.index:
        sid = rid(idx)
        pts = [df.at[idx, f"ct_point_{a}"] for a in "xyz"]
        if any(pts) and not all(pts):
            add("error", "partial_point", sid, "ct_point_x/y/z", "ct_point_x/y/z 只填了一部分，应三个都填或都空。")
        if df.at[idx, "tumor_mask_path"] and not df.at[idx, "mask_source"]:
            add("warning", "mask_without_source", sid, "mask_source", "有 tumor_mask_path 但 mask_source 为空。")
        if df.at[idx, "mask_source"] and not df.at[idx, "tumor_mask_path"]:
            add("warning", "source_without_mask", sid, "tumor_mask_path", "有 mask_source 但没有 tumor_mask_path。")
        if df.at[idx, "her2_ihc"] == "2+" and not df.at[idx, "her2_fish"]:
            rule = cfg.labels.get("her2_2plus_without_fish", "missing")
            add("warning", "her2_2plus_no_fish", sid, "her2_fish",
                f"HER2 IHC 2+ 但无 FISH 结果，HER2 标签按{'缺失' if rule == 'missing' else '阴性'}处理。")
        for ep in ("os", "dfs", "pfs"):
            m, e = df.at[idx, f"{ep}_months"], df.at[idx, f"{ep}_event"]
            if bool(m) != bool(e):
                add("warning", f"partial_{ep}", sid, f"{ep}_months/{ep}_event",
                    f"{ep}_months 与 {ep}_event 只填了一项，该终点按缺失处理。")
        om = _to_float(df.at[idx, "os_months"])
        if om is not None and om > max_os:
            add("warning", "os_too_long", sid, "os_months", f"os_months={om:g} 超过 {max_os:g}，请确认单位是“月”。")
        pop = df.at[idx, "population"]
        if pop == "unresectable" and (df.at[idx, "dfs_months"] or df.at[idx, "dfs_event"]):
            add("warning", "dfs_unresectable", sid, "dfs_months", "不可切除患者填写了 DFS（DFS 只用于手术患者）。")
        if pop == "resected" and (df.at[idx, "pfs_months"] or df.at[idx, "pfs_event"]):
            add("warning", "pfs_resected", sid, "pfs_months", "手术患者填写了 PFS（PFS 只用于不可切除/晚期患者）。")
        has_label = any(df.at[idx, c] for c in ("lauren", "her2_ihc", "msi", "ebv", "pdl1_cps", "cldn182_pct", "ki67_pct"))
        if has_label and not df.at[idx, "label_source"]:
            add("warning", "no_label_source", sid, "label_source", "有分型/标志物标签但 label_source 为空（主分析按非新辅助后标本处理）。")
        if df.at[idx, "neoadj"] in ("1", "1.0") and df.at[idx, "label_source"] == "surgery":
            add("warning", "neoadj_surgery_label", sid, "label_source",
                "新辅助患者的标签来源为 surgery：若为新辅助治疗后的手术标本，应填 post_neoadj。")

    # ---- 路径 ----
    for c in PATH_COLUMNS:
        nonempty = df[c][df[c] != ""]
        dup = nonempty[nonempty.duplicated(keep=False)]
        for idx in dup.index:
            add("warning", f"dup_path_{c}", rid(idx), c, f"{c} 与其他病例相同（同一文件被多个病例使用？）。")
    if check_paths:
        level = "error" if strict_paths else "warning"
        for idx in df.index:
            for c in PATH_COLUMNS:
                v = df.at[idx, c]
                if not v:
                    continue
                p = resolve_path(cfg, v)
                kind = ALL_COLUMNS[c]["kind"]
                ok = p.is_file() if kind == "path_file" else (p.is_dir() and any(p.iterdir()))
                if not ok:
                    what = "文件不存在" if kind == "path_file" else "文件夹不存在或为空"
                    add(level, f"missing_path_{c}", rid(idx), c, f"{c} 指向的{what}，该模态按缺失处理。")

    typed = _typed(df)
    res = ValidationResult(df=typed, issues=issues)
    res.summary = _summarize(typed, cfg, issues, check_paths)
    return res


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in NUMERIC_COLUMNS:
            out[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce").astype(float)
        else:
            out[c] = pd.Series([v if v != "" else None for v in df[c]], index=df.index, dtype=object)
    return out


def _summarize(df: pd.DataFrame, cfg: Config, issues: List[Issue], check_paths: bool) -> dict:
    missing_paths = {(i.study_id, i.column) for i in issues if i.code.startswith("missing_path_")}

    def has(col, sid, v):
        return v is not None and (sid, col) not in missing_paths

    centers = [c for c in cfg.manifest.centers]
    by_center = {}
    for c in centers + ["全部"]:
        sub = df if c == "全部" else df[df["center"] == c]
        if c != "全部" and len(sub) == 0:
            continue
        sids = sub["study_id"].tolist()
        pt = sub[["ct_point_x", "ct_point_y", "ct_point_z"]].notna().all(axis=1)
        by_center[c] = {
            "n": int(len(sub)),
            "slide": int(sum(has("slide_path", s, v) for s, v in zip(sids, sub["slide_path"]))),
            "ct": int(sum(has("ct_dir", s, v) for s, v in zip(sids, sub["ct_dir"]))),
            "ct_point": int(pt.sum()),
            "tumor_mask": int(sum(has("tumor_mask_path", s, v) for s, v in zip(sids, sub["tumor_mask_path"]))),
            "resected": int((sub["population"] == "resected").sum()),
            "unresectable": int((sub["population"] == "unresectable").sum()),
            "neoadj": int((sub["neoadj"] == 1).sum()),
            "immuno": int((sub["immuno"] == 1).sum()),
            "post_neoadj_label": int((sub["label_source"] == "post_neoadj").sum()),
        }
    return {
        "n_cases": int(len(df)),
        "n_errors": sum(1 for i in issues if i.level == "error"),
        "n_warnings": sum(1 for i in issues if i.level == "warning"),
        "paths_checked": bool(check_paths),
        "by_center": by_center,
    }


# ---------------- 报告 ----------------
def _fmt_rate(r) -> str:
    return "" if r is None else f"{r * 100:.1f}%"


def build_report(res: ValidationResult, labels: Optional[pd.DataFrame], cfg: Config) -> str:
    s = res.summary
    L = ["# 病例清单校验报告", ""]
    verdict = "通过" if res.n_errors == 0 else "未通过（有错误，必须修正）"
    L += [f"- 结论：**{verdict}**", f"- 病例数：{s.get('n_cases', 0)}",
          f"- 错误：{res.n_errors}　警告：{res.n_warnings}",
          f"- 路径检查：{'已检查' if s.get('paths_checked') else '未检查'}", ""]
    if s.get("by_center"):
        L += ["## 各中心病例与模态", "",
              "| 中心 | 病例 | 切片 | CT | 中心点 | 肿瘤分割 | 手术 | 不可切除 | 新辅助 | 免疫 | 新辅助后标签 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, d in s["by_center"].items():
            L.append(f"| {c} | {d['n']} | {d['slide']} | {d['ct']} | {d['ct_point']} | {d['tumor_mask']} | "
                     f"{d['resected']} | {d['unresectable']} | {d['neoadj']} | {d['immuno']} | {d['post_neoadj_label']} |")
        L.append("")
    if labels is not None and len(labels):
        rows = label_summary(labels, cfg)
        centers = list(OrderedDict.fromkeys(r["center"] for r in rows))
        L += ["## 训练标签（派生后）", "",
              "按派生规则（labels.py）生成；新辅助后标本的标签"
              + ("已启用。" if cfg.labels.get("use_post_neoadj") else "按缺失处理。"),
              "每格为：有标签例数 / 阳性（生存为事件）例数（比例）；多分类为各类例数；Ki-67 为中位数。", "",
              "| 任务 | 主分析 | " + " | ".join(centers) + " |", "|---|---|" + "---|" * len(centers)]
        by_task: "OrderedDict[str, dict]" = OrderedDict()
        for r in rows:
            by_task.setdefault(r["task"], {"main": r["main"]})[r["center"]] = r
        for task, d in by_task.items():
            cells = []
            for c in centers:
                r = d.get(c)
                if r is None:
                    cells.append("")
                elif r["note"] == "中位数":
                    cells.append(f"{r['n_labeled']}（{r['rate']:.2f}）" if r["rate"] is not None else f"{r['n_labeled']}")
                elif r["n_positive"] is None:
                    cells.append(f"{r['n_labeled']}：{r['note'].replace('各类数量 ', '')}")
                else:
                    cells.append(f"{r['n_labeled']} / {r['n_positive']}（{_fmt_rate(r['rate'])}）")
            L.append(f"| {task} | {d['main']} | " + " | ".join(cells) + " |")
        L.append("")
    if res.issues:
        L += ["## 问题汇总", "", "完整列表见 validation_issues.csv。", "",
              "| 级别 | 列 | 问题 | 例数 | 病例（最多 20 个） |", "|---|---|---|---|---|"]
        groups: "OrderedDict[tuple, List[Issue]]" = OrderedDict()
        for i in sorted(res.issues, key=lambda x: (x.level != "error", x.code)):
            groups.setdefault((i.level, i.code, i.column), []).append(i)
        for (level, code, col), items in groups.items():
            ids = [x.study_id for x in items if x.study_id]
            msg = items[0].message
            L.append(f"| {'错误' if level == 'error' else '警告'} | {col} | {msg} | {len(items)} | "
                     f"{', '.join(ids[:20])}{' ...' if len(ids) > 20 else ''} |")
        L.append("")
    else:
        L += ["未发现问题。", ""]
    return "\n".join(L)


# ---------------- 供其他模块使用 ----------------
def load_manifest(cfg: Config, logger=None, check_paths: Optional[bool] = None) -> ValidationResult:
    """读取并校验清单。有错误时抛出 UserError。"""
    path = resolve_path(cfg, cfg.paths.manifest)
    raw = read_manifest_csv(path)
    res = validate_manifest(raw, cfg, check_paths=check_paths)
    if res.n_errors:
        first = [f"{i.study_id or '-'} {i.column}: {i.message}" for i in res.issues if i.level == "error"][:5]
        raise UserError(
            f"病例清单有 {res.n_errors} 个错误，例如：\n    " + "\n    ".join(first),
            "请先运行 bash scripts/01_validate_manifest.sh 查看完整校验报告并修正清单。",
        )
    if logger is not None:
        logger.info("清单：%d 例（警告 %d 条，详见清单校验报告）", len(res.df), res.n_warnings)
    return res


def missing_path_set(res: ValidationResult) -> set:
    """校验中发现路径不存在的 (study_id, 列名) 集合。"""
    return {(i.study_id, i.column) for i in res.issues if i.code.startswith("missing_path_")}


def to_cases(df: pd.DataFrame) -> List[dict]:
    """把清单行转成普通字典（缺失为 None，整数列转 int）。"""
    cases = []
    for _, row in df.iterrows():
        d = {}
        for c, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                v = None
            if v is not None and c in ALL_COLUMNS and ALL_COLUMNS[c]["kind"] in ("int", "binary"):
                v = int(v)
            d[c] = v
        cases.append(d)
    return cases


def select_cases(cases: List[dict], study_ids: Optional[List[str]] = None, limit: Optional[int] = None,
                 require: Sequence[str] = ()) -> List[dict]:
    out = [c for c in cases if all(c.get(r) not in (None, "") for r in require)]
    if study_ids:
        want = set(study_ids)
        unknown = want - {c["study_id"] for c in cases}
        if unknown:
            raise UserError(f"以下 study_id 不在清单中：{', '.join(sorted(unknown))}")
        out = [c for c in out if c["study_id"] in want]
    if limit is not None:
        out = out[: max(0, limit)]
    return out


# ---------------- 命令行 ----------------
def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "manifest", args.run_name)
    log = run.logger
    path = resolve_path(cfg, cfg.paths.manifest)
    log.info("读取清单（配置项 paths.manifest）")
    raw = read_manifest_csv(path)
    res = validate_manifest(raw, cfg, check_paths=False if args.no_path_check else None)
    labels = None
    if not any(i.code == "missing_required_column" for i in res.issues):
        labels = derive_labels(res.df, cfg)
        labels.to_csv(run.path("labels_preview.csv"), index=False, encoding="utf-8-sig")
    run.write_table([i.__dict__ for i in res.issues], "validation_issues.csv",
                    ["level", "code", "study_id", "column", "message"])
    run.write_json(res.summary, "summary.json")
    run.write_text(build_report(res, labels, cfg), "validation_report.md")
    if res.n_errors:
        log.error("校验未通过：%d 个错误、%d 条警告。请查看 %s 修正后重跑。",
                  res.n_errors, res.n_warnings, run.path("validation_report.md").name)
        return 1
    log.info("校验通过：%d 例，%d 条警告。报告：%s", len(res.df), res.n_warnings, run.path("validation_report.md"))
    return 0


def main(argv=None) -> int:
    p = base_parser("读取并校验病例清单，生成校验报告和训练标签预览。", cases=False, workers=False)
    p.add_argument("--no-path-check", action="store_true", help="不检查文件路径是否存在（只校验表格内容）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
