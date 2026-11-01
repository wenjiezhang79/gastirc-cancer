"""v4 公共部分：读取各模型在各队列上的预测、合并协变量、冻结量（截断值 / 标准化）、Cox 设计矩阵。

预测的来源（只读，不重新训练、不再次运行最终评估）：
- ``internal_train``：实验目录下的 ``oof_predictions.csv``（内部训练集折外预测，种子平均）；
- ``test`` / ``ext1`` / ``ext2`` / ``ext3``：实验目录 ``final_evaluations.json`` 中登记的最终评估结果
  （``final_<队列>_<时间>/predictions.csv``；外部中心优先取 ``external`` 那次的结果，同一队列取最后一次）。
没有任何最终评估时为"开发模式"：只分析 internal_train（报告中注明）。

冻结量一律来自内部训练集：二分类阈值与生存风险截断取各实验的 ``postprocess.json``；风险评分的标准化（均值、标准差）、
三分位截断、一线方案合并规则由 internal_train 计算后用于所有队列。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config, resolve_path
from ..labels import derive_labels
from ..manifest import load_manifest
from ..split import load_split
from ..utils.errors import UserError
from .clinical import FIRSTLINE, STAGE_GROUPS, TX_PATTERNS, derive_clinical, firstline_mapping

SECTION = "v4"
COHORTS = ("internal_train", "test", "ext1", "ext2", "ext3")
EXTERNAL = ("ext1", "ext2", "ext3")
COHORT_NAMES = {"internal_train": "内部训练集（折外）", "test": "内部测试集", "ext1": "外部 ext1", "ext2": "外部 ext2",
                "ext3": "外部 ext3", "external": "外部合并", "all": "全部队列合并"}
POP_NAMES = {"resected": "手术患者", "unresectable": "不可切除患者"}
# 图中文字（默认英文，v4.figure_language: zh 时用中文；报告正文始终为中文）
COHORT_EN = {"internal_train": "Internal training (OOF)", "test": "Internal test", "ext1": "External 1", "ext2": "External 2",
             "ext3": "External 3", "external": "External pooled", "all": "All cohorts"}
POP_EN = {"resected": "Resected", "unresectable": "Unresectable"}
LABEL_EN = {"低风险": "Low risk", "中风险": "Intermediate risk", "高风险": "High risk", "低": "Low", "中": "Intermediate",
            "高": "High", "单纯手术": "Surgery only", "手术 + 辅助化疗": "Surgery + adjuvant chemo",
            "新辅助 + 手术": "Neoadjuvant + surgery", "未接受系统治疗": "No systemic therapy", "单纯化疗": "Chemo",
            "化疗 + 抗 HER2": "Chemo + anti-HER2", "化疗 + 免疫": "Chemo + IO", "化疗 + Claudin18.2 靶向": "Chemo + anti-CLDN18.2",
            "其他": "Other", "化疗 + 其他（合并）": "Chemo + other (merged)", "对照": "Control", "免疫": "Immunotherapy",
            "阳性": "Positive", "阴性": "Negative", "肠型": "Intestinal", "弥漫型": "Diffuse", "混合型": "Mixed",
            "未辅助化疗": "No adjuvant chemo", "辅助化疗": "Adjuvant chemo", "列线图": "Nomogram", "单独分期": "Stage only"}


def setup_figures(cfg, logger=None) -> None:
    from ..stats.plots import cjk_font_available, set_language
    lang = str(cfg.get_path("v4.figure_language", "en"))
    if lang.startswith("zh") and not cjk_font_available():
        if logger is not None:
            logger.warning("v4.figure_language = zh，但系统没有中文字体，图中文字会显示为方框；已改用英文。")
        lang = "en"
    set_language(lang)


def ft(text: str) -> str:
    """把图中用到的中文标签换成当前图语言（未登记的原样返回）。"""
    from ..stats.plots import tr as ftr
    return ftr(text, LABEL_EN.get(text, text))


def fc(coh: str) -> str:
    from ..stats.plots import tr as ftr
    return ftr(COHORT_NAMES.get(coh, coh), COHORT_EN.get(coh, coh))


def fp(pop: str) -> str:
    from ..stats.plots import tr as ftr
    return ftr(POP_NAMES.get(pop, pop), POP_EN.get(pop, pop))


def v4(cfg: Config):
    if SECTION not in cfg:
        raise UserError("配置中没有 v4 一节", "请确认配置继承了 configs/default.yaml（_base_: default.yaml）。")
    return cfg[SECTION]


# ---------------- 模型与预测 ----------------
class ModelPreds:
    """一个已训练实验在各队列上的预测。ref 写作 <命名空间>/<实验名>，如 fusion/fusion_main、path_mtl/uni2h。"""

    def __init__(self, cfg: Config, ref: str):
        if "/" not in ref:
            raise UserError(f"v4.models 中的“{ref}”格式不对", "请写成 <命名空间>/<实验名>，如 fusion/fusion_main、path_mtl/uni2h_abmil。")
        self.ref = ref
        self.ns, self.name = ref.split("/", 1)
        self.dir = resolve_path(cfg, cfg.paths.output_dir) / self.ns / self.name
        oof = self.dir / "oof_predictions.csv"
        if not oof.exists():
            raise UserError(f"实验 {ref} 没有 oof_predictions.csv（交叉验证尚未汇总，或实验名写错）",
                            "请确认 v4.models 中的实验已跑完全部折，且配置文件（输出目录）与训练时相同。")
        pp = self.dir / "postprocess.json"
        self.post = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
        self.sources: Dict[str, str] = {"internal_train": "oof_predictions.csv"}
        frames = [pd.read_csv(oof, dtype={"study_id": str}).assign(cohort="internal_train")]
        reg_p = self.dir / "final_evaluations.json"
        reg = json.loads(reg_p.read_text(encoding="utf-8")) if reg_p.exists() else []
        latest: Dict[str, dict] = {}
        for r in reg:
            latest[r["cohort"]] = r
        used = set()
        for coh in ("external", "test", "ext1", "ext2", "ext3"):
            r = latest.get(coh)
            if r is None:
                continue
            f = self.dir / r["dir"] / "predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, dtype={"study_id": str})
            d = d[~d.study_id.isin(used)]
            used |= set(d.study_id)
            frames.append(d.assign(cohort=coh))
            self.sources[coh] = f"{r['dir']}（{r['time']}{'，重跑' if r.get('rerun') else ''}）"
        self.pred = pd.concat(frames, ignore_index=True)

    @property
    def final_cohorts(self) -> List[str]:
        return [c for c in self.sources if c != "internal_train"]

    def threshold(self, task: str) -> float:
        return float(self.post.get("binary_thresholds_youden", {}).get(task, float("nan")))

    def risk_cutoff(self, ep: str, pop: str) -> float:
        d = self.post.get("survival_risk_cutoffs_median", {}).get(ep, {})
        return float(d.get(pop, d.get("all", float("nan"))))


def load_covariates(cfg: Config, logger=None) -> pd.DataFrame:
    """标签（含生存）+ 临床派生变量 + 划分信息（set / fold / 队列名）。"""
    res = load_manifest(cfg, logger, check_paths=False)
    lab = derive_labels(res.df, cfg)
    clin = derive_clinical(res.df)
    split = load_split(cfg)[["study_id", "set", "fold"]]
    df = lab.merge(clin.drop(columns=["population"]), on="study_id", how="left").merge(split, on="study_id", how="left")
    internal = cfg.manifest.internal_center
    df["cohort"] = np.select([(df.set == "train"), (df.set == "test") & (df.center == internal)],
                             ["internal_train", "test"], default=df["center"].astype(str))
    df.loc[df.set.isna(), "cohort"] = "unsplit"
    return df


def merged_table(cov: pd.DataFrame, mp: ModelPreds) -> pd.DataFrame:
    """协变量 + 某模型的预测（只保留该模型有预测的病例；内部训练集只用折外预测）。"""
    p = mp.pred.drop(columns=[c for c in ("fold",) if c in mp.pred.columns]).rename(columns={"cohort": "pred_source"})
    d = p.merge(cov, on="study_id", how="left", suffixes=("", "_label"))      # 队列名一律以划分文件为准
    bad = d[(d.pred_source == "internal_train") & (d.set != "train")]
    if len(bad):
        raise UserError(f"模型 {mp.ref} 的折外预测中有 {len(bad)} 例不属于内部训练集",
                        "请确认 v1 / v2 / v3 与 v4 使用同一个划分文件与清单。")
    return d


def cohort_groups(df: pd.DataFrame, extra: Sequence[str] = ("external",)) -> List[Tuple[str, pd.DataFrame]]:
    """按评价顺序给出各队列：内部训练集（折外）、内部测试集、外部各中心、外部合并（有外部结果时）。"""
    out = []
    for c in COHORTS:
        g = df[df.cohort == c]
        if len(g):
            out.append((c, g))
    ext = df[df.cohort.isin(EXTERNAL)]
    if "external" in extra and ext.cohort.nunique() >= 1 and len(ext):
        out.append(("external", ext))
    return out


# ---------------- 冻结量 ----------------
def frozen_risk_stats(df: pd.DataFrame, ep: str = "os") -> Dict[str, Dict[str, float]]:
    """各人群风险评分在内部训练集折外预测中的均值、标准差与三分位截断（冻结后用于所有队列）。"""
    out = {}
    tr = df[df.cohort == "internal_train"]
    for pop in ("resected", "unresectable"):
        r = tr.loc[tr.population == pop, f"{ep}_risk"].dropna()
        if len(r) >= 10:
            out[pop] = {"mean": float(r.mean()), "sd": float(r.std(ddof=1)) or 1.0,
                        "t1": float(r.quantile(1 / 3)), "t2": float(r.quantile(2 / 3)), "n": int(len(r))}
    return out


def add_risk_vars(df: pd.DataFrame, mp: ModelPreds, stats: Dict[str, dict], ep: str = "os") -> pd.DataFrame:
    """加入 risk_z（按人群用冻结的均值 / 标准差标准化）、risk_high（冻结中位数截断）、risk_tertile。"""
    df = df.copy()
    df["risk_z"] = np.nan
    df["risk_high"] = np.nan
    df["risk_tertile"] = None
    for pop, st in stats.items():
        m = df.population == pop
        r = df.loc[m, f"{ep}_risk"]
        df.loc[m, "risk_z"] = (r - st["mean"]) / st["sd"]
        cut = mp.risk_cutoff(ep, pop)
        df.loc[m, "risk_high"] = (r > cut).astype(float).where(r.notna())
        df.loc[m, "risk_tertile"] = np.where(r.isna(), None, np.where(r <= st["t1"], "低", np.where(r <= st["t2"], "中", "高")))
    return df


def add_firstline(df: pd.DataFrame, min_n: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    tr = df[(df.cohort == "internal_train") & (df.population == "unresectable")]
    mp = firstline_mapping(tr["first_line"], min_n)
    df = df.copy()
    df["tx_firstline"] = df["first_line"].map(lambda v: mp.get(v) if isinstance(v, str) else None)
    return df, mp


# ---------------- Cox 设计矩阵 ----------------
NUMERIC_COVARS = {"age10": ("age", 10.0), "ecog": ("ecog", 1.0), "risk_z": ("risk_z", 1.0)}
CATEGORICAL = {"stage_group": STAGE_GROUPS, "tx_pattern": TX_PATTERNS, "tx_firstline": FIRSTLINE, "sex": ["M", "F"],
               "immuno": [0.0, 1.0]}


def design(df: pd.DataFrame, covars: Sequence[str], missing: str = "complete", min_level_n: int = 5):
    """构造 Cox 设计矩阵。covars 可含 risk_z、age10（每 10 岁）、ecog、stage_group、tx_pattern、tx_firstline、sex、immuno。

    分类变量哑编码，参照组 = 该变量取值列表中第一个**实际出现**的水平；例数 < min_level_n 的水平并入参照组（并注明）。
    missing = complete：任一协变量缺失的病例去掉；missing = category：分类变量缺失作为"未记录"一类，数值变量用中位数填补并加缺失指示。
    返回 (X, 列名, 保留病例的布尔掩码, 说明列表)。
    """
    cols, names, notes = [], [], []
    keep = np.ones(len(df), bool)
    for cv in covars:
        if cv in NUMERIC_COVARS:
            src, div = NUMERIC_COVARS[cv]
            v = pd.to_numeric(df[src], errors="coerce").to_numpy(float) / div
            if missing == "category" and cv != "risk_z" and np.isnan(v).any():
                ind = np.isnan(v).astype(float)
                v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0.0, v)
                if ind.sum() >= min_level_n:
                    cols.append(ind)
                    names.append(f"{cv}[未记录]")
            keep &= np.isfinite(v)
            cols.append(v)
            names.append(cv)
        elif cv in CATEGORICAL:
            raw = df[cv]
            v = raw.where(raw.notna(), None).to_numpy(object)
            if missing == "category":
                v = np.array(["未记录" if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in v], object)
            else:
                keep &= np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) for x in v])
            levels = [lv for lv in CATEGORICAL[cv] if (v == lv).sum() > 0] + (["未记录"] if (v == "未记录").sum() else [])
            if len(levels) < 2:
                notes.append(f"{cv} 只有一个水平，未纳入")
                continue
            ref = levels[0]
            for lv in levels[1:]:
                ind = (v == lv).astype(float)
                if ind[keep].sum() < min_level_n:
                    notes.append(f"{cv}={lv} 例数 < {min_level_n}，并入参照组 {ref}")
                    continue
                cols.append(ind)
                names.append(f"{cv}[{lv} vs {ref}]")
        else:
            raise ValueError(f"未知协变量 {cv}")
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    # 去掉在保留病例中恒定的列
    if X.shape[1]:
        var = X[keep].std(0) if keep.sum() else np.zeros(X.shape[1])
        good = var > 0
        for nm, g in zip(names, good):
            if not g:
                notes.append(f"{nm} 在分析病例中不变，未纳入")
        X = X[:, good]
        names = [n for n, g in zip(names, good) if g]
    return X, names, keep, notes


def surv_arrays(df: pd.DataFrame, ep: str):
    t = pd.to_numeric(df[f"{ep}_time"], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df[f"{ep}_event"], errors="coerce").to_numpy(float)
    return t, e, np.isfinite(t) & np.isfinite(e)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt(v, d: int = 3) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def p_text(p) -> str:
    """图中用："p<0.001" 或 "p=0.023"。"""
    if p is None or not np.isfinite(p):
        return "p=—"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
