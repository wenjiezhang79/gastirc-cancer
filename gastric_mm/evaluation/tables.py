"""v4 论文用表：表 1（各队列基线特征）、病例流程图人数、TRIPOD+AI 清单对照（docs/decisions.md 第十五节第 9 小节）。

表 1 覆盖清单中划分过的全部病例（不只是有预测的病例）；模态可用性取主模型预测中的 has_* 列（无预测 = 两种模态都没有）。
只做描述，不做组间检验。

命令行：
    python -m gastric_mm.evaluation.tables --config configs/server.yaml

输出：<output_dir>/v4/tables/<时间>/table1.md、table1.csv、flow.md、tripod_ai.md
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import load_config
from ..stats.survival import median_followup
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from .clinical import FIRSTLINE_NAMES, TX_PATTERN_NAMES
from .common import COHORT_NAMES, COHORTS, ModelPreds, load_covariates, v4

COLS = list(COHORTS)


def _n_pct(m: pd.Series, denom: int) -> str:
    k = int(m.sum())
    return f"{k} ({100 * k / denom:.1f}%)" if denom else "0"


def table1(df: pd.DataFrame) -> pd.DataFrame:
    groups = [(c, df[df.cohort == c]) for c in COLS if (df.cohort == c).any()] + [("合计", df)]
    rows: List[dict] = []

    def add(label, fn):
        r = {"特征": label}
        for c, g in groups:
            r["内部训练集" if c == "internal_train" else COHORT_NAMES.get(c, c)] = fn(g)
        rows.append(r)

    add("例数", lambda g: str(len(g)))
    add("年龄，均值 ± SD", lambda g: f"{g.age.mean():.1f} ± {g.age.std():.1f}" if g.age.notna().any() else "—")
    add("  年龄缺失", lambda g: str(int(g.age.isna().sum())))
    add("性别：男", lambda g: _n_pct(g.sex == "M", len(g)))
    for lv, nm in ((0, "0"), (1, "1"), (2, "≥2")):
        add(f"ECOG {nm}", lambda g, lv=lv: _n_pct((g.ecog >= 2) if lv == 2 else (g.ecog == lv), len(g)))
    add("  ECOG 缺失", lambda g: str(int(g.ecog.isna().sum())))
    add("人群：手术", lambda g: _n_pct(g.population == "resected", len(g)))
    add("人群：不可切除", lambda g: _n_pct(g.population == "unresectable", len(g)))
    for st in ("I", "II", "III", "IV"):
        add(f"cTNM {st} 期", lambda g, st=st: _n_pct(g.stage_group == st, len(g)))
    add("  分期缺失", lambda g: str(int(g.stage_group.isna().sum())))
    for k, nm in enumerate(("肠型", "弥漫型", "混合型")):
        add(f"Lauren {nm}", lambda g, k=k: _n_pct(g.lauren == k, len(g)))
    for t, nm in (("her2_pos", "HER2 阳性"), ("msi_h", "MSI-H"), ("ebv_pos", "EBV 阳性"), ("pdl1_cps5", "PD-L1 CPS≥5"),
                  ("cldn182_pos", "Claudin18.2 阳性")):
        if t in df:
            add(f"{nm}（阳性 / 有结果）", lambda g, t=t: f"{int((g[t] == 1).sum())} / {int(g[t].notna().sum())}")
    for tp in ("surgery_only", "surgery_adjuvant", "neoadj_surgery"):
        add(f"手术者：{TX_PATTERN_NAMES[tp]}", lambda g, tp=tp: _n_pct(g.tx_pattern == tp, int((g.population == "resected").sum())))
    for fl in ("none", "chemo", "chemo_her2", "chemo_io", "chemo_cldn", "other"):
        if (df.first_line == fl).any():
            add(f"不可切除者一线：{FIRSTLINE_NAMES[fl]}",
                lambda g, fl=fl: _n_pct(g.first_line == fl, int((g.population == "unresectable").sum())))
    add("免疫治疗", lambda g: _n_pct(pd.to_numeric(g.immuno, errors="coerce") == 1, len(g)))
    if "has_path" in df:
        add("有病理特征", lambda g: _n_pct(g.has_path == 1, len(g)))
        add("有 CT 特征", lambda g: _n_pct(g.has_ct == 1, len(g)))

    def fu(g):
        t, e = pd.to_numeric(g.os_time, errors="coerce"), pd.to_numeric(g.os_event, errors="coerce")
        ok = t.notna() & e.notna()
        return f"{median_followup(t[ok], e[ok]):.1f}" if ok.sum() >= 5 else "—"
    add("中位随访（月，反向 KM）", fu)
    add("OS 事件（死亡）", lambda g: str(int((pd.to_numeric(g.os_event, errors="coerce") == 1).sum())))
    return pd.DataFrame(rows)


def to_md(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(L)


TRIPOD = [
    ("1", "标题", "注明为多模态深度学习模型的开发与外部验证、目标人群（治疗前胃癌）与预测目标（分型、标志物、预后）"),
    ("2", "摘要", "按 TRIPOD+AI 摘要清单；报告外部合并队列的主要指标（见各 v4 报告）"),
    ("3", "背景与目标", "docs/design.md 第一节；预先规定的统计分析计划：docs/decisions.md 第十五节"),
    ("4–5", "数据来源与参与者", "4 中心回顾性队列；纳入 / 排除标准见 docs/design.md 第二节；病例流程见 flow.md"),
    ("6", "数据预处理", "病理：TRIDENT 20× 切块特征；CT：静脉期 5 mm，ROI 规格见配置 ct.crop.specs；docs/decisions.md 第二至四节"),
    ("7", "结局", "Lauren 分型、6 项标志物（派生规则 labels.py）、OS / DFS / PFS（起点见第十五节第 3 小节）、pCR / TRG"),
    ("8", "预测因子", "治疗前 HE 切片与 CT；治疗与临床变量不作模型输入，只用于统计分析"),
    ("9", "样本量", "内部 2000 例（训练约 1600 / 测试约 400），外部 500 / 500 / 200；各分析的事件数见 v4 报告"),
    ("10", "缺失数据", "标签缺失屏蔽损失；缺失模态屏蔽；多因素 Cox 完整病例 + “未记录”敏感性分析；模态可用性见 fusion_check 报告"),
    ("11", "分析方法", "ABMIL / 2.5D CT 编码器 / 跨模态 Transformer；5 折交叉验证与三方共用内层验证早停；判定规则与比较见 v3 比较报告"),
    ("12", "类别不平衡", "正样本加权；MSI、EBV 报告 PR-AUC 与外部合并结果"),
    ("13", "公平性", "按中心、人群、模态可用性分层报告（v3 / v4 报告）"),
    ("14", "模型输出", "各任务概率、风险评分、各时间点生存概率；阈值与截断值由内部训练集折外预测冻结（postprocess.json）"),
    ("15", "训练与评估的区分", "内部测试集与外部队列训练代码不读取；最终评估加 --final 且登记防重复（final_evaluations.json）"),
    ("16", "伦理", "研究团队填写伦理审批号；仓库与日志中无患者信息，只用 study_id"),
    ("17", "开放科学", "代码仓库；模型权重与数据共享按协议"),
    ("18–19", "患者与公众参与 / 人群", "研究团队填写"),
    ("20", "参与者结果", "表 1（table1.md）与流程图（flow.md）"),
    ("21", "模型开发", "各实验 cv_report.md 与 experiment.json（配置指纹）"),
    ("22", "模型性能", "v4/classification、v4/prognosis、v4/nomogram 报告（含 95% CI、校准、决策曲线）"),
    ("23", "模型更新", "外部队列只运行一次；校准截距调整仅作敏感性分析"),
    ("24–27", "讨论（解释、局限、可用性）", "局限：回顾性、治疗交互只能“提示”、阳性少的标志物置信区间宽；早停与“背答案”诊断见 docs/v3_diagnosis.md"),
]


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    run = Run(cfg, "v4/tables", args.run_name)
    log = run.logger
    cov = load_covariates(cfg, log)
    try:
        mp = ModelPreds(cfg, sec.models.main)
        has = mp.pred.copy()
        hc = [c for c in has.columns if c.startswith("has_") and c != "has_path"]
        has["has_ct"] = (has[hc].fillna(0).max(1) > 0.5).astype(int) if hc else 0
        if "has_path" not in has:
            has["has_path"] = 0
        cov = cov.merge(has[["study_id", "has_path", "has_ct"]].drop_duplicates("study_id"), on="study_id", how="left")
        cov[["has_path", "has_ct"]] = cov[["has_path", "has_ct"]].fillna(0)
    except Exception as ex:  # 没有主模型时仍可出表 1
        log.warning("读取主模型预测失败（%s），表 1 不含模态可用性", ex)
    df = cov[cov.cohort != "unsplit"]
    t1 = table1(df)
    t1.to_csv(run.path("table1.csv"), index=False, encoding="utf-8-sig")
    run.write_text("# 表 1 各队列基线特征\n\n只做描述，不做组间检验。\n\n" + to_md(t1) + "\n", "table1.md")
    F = ["# 病例流程（流程图人数）", "", f"- 清单病例：{len(cov)}（未进入划分 {int((cov.cohort == 'unsplit').sum())}）"]
    for c in COLS:
        g = df[df.cohort == c]
        if not len(g):
            continue
        line = f"- {COHORT_NAMES[c]}：{len(g)}"
        if "has_path" in g:
            both = int(((g.has_path == 1) & (g.has_ct == 1)).sum())
            anyv = int(((g.has_path == 1) | (g.has_ct == 1)).sum())
            line += f"；至少一种模态 {anyv}（两种都有 {both}）"
        os_ok = int((pd.to_numeric(g.os_time, errors="coerce").notna() & pd.to_numeric(g.os_event, errors="coerce").notna()).sum())
        line += f"；有 OS 随访 {os_ok}（手术 {int((g.population == 'resected').sum())}，不可切除 {int((g.population == 'unresectable').sum())}）"
        F.append(line)
    run.write_text("\n".join(F) + "\n", "flow.md")
    T = ["# TRIPOD+AI 清单对照（模板，投稿前逐项核对）", "", "| 条目 | 内容 | 本项目对应位置 / 说明 |", "|---|---|---|"]
    T += [f"| {a} | {b} | {c} |" for a, b, c in TRIPOD]
    run.write_text("\n".join(T) + "\n", "tripod_ai.md")
    log.info("论文用表：%s", run.dir)
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 论文用表：表 1、病例流程图人数、TRIPOD+AI 清单对照。", cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
