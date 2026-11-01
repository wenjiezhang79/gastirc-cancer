"""v4 评估统计：临床派生规则、开发模式（无最终评估）、完整流程（含 --final）、冻结量、交互作用、隐私。

使用含已知治疗效应的多模态假队列（fake_data --mm-cohort，v4_effects=true），全部在 CPU 上运行。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gastric_mm.evaluation import (classification, explain, immuno, neoadj, nomogram, prognosis, table, tables)

REPO = Path(__file__).resolve().parent.parent


class V4Env:
    def __init__(self, base: Path):
        self.base = base
        self.config = base / "v4.yaml"
        self.out = base / "outputs"

    def run(self, main, *args):
        return main(["--config", str(self.config), *args])

    def latest(self, step: str) -> Path:
        return sorted((self.out / "v4" / step).iterdir())[-1]


@pytest.fixture(scope="session")
def v4_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v4")
    cfg = {
        "_base_": str(REPO / "configs" / "fake_v4.yaml"),
        "paths": {"root": str(base / "data"), "output_dir": str(base / "outputs")},
        "split": {"n_folds": 3},
        "fake_data": {"mm_cohort": {"n_internal": 420, "n_external": 150, "v4_effects": True, "signal": 2.0}},
        "v1": {"train": {"epochs": 12, "lr": 5.0e-4}, "eval": {"n_boot": 30}},
        "v2": {"train": {"epochs": 12, "lr": 5.0e-4}, "eval": {"n_boot": 30}},
        "v3": {"train": {"epochs": 12, "lr": 5.0e-4}, "eval": {"n_boot": 30}},
        "v4": {"n_boot": 60, "min_events": 15, "explain": {"n_cases": 2}},
    }
    (base / "v4.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    env = V4Env(base)
    from gastric_mm import fake_data
    from gastric_mm.ct_mtl import train as ct_train
    from gastric_mm.fusion import late, train as f_train
    from gastric_mm.path_mtl import train as p_train
    assert env.run(fake_data.main, "--mm-cohort") == 0
    assert env.run(p_train.main, "--exp", "demo_path") == 0
    assert env.run(ct_train.main, "--exp", "demo_ct") == 0
    assert env.run(f_train.main, "--exp", "demo_fusion") == 0
    assert env.run(late.main, "--exp", "late_stack", "--method", "stack") == 0
    return env


@pytest.fixture(scope="session")
def v4_final(v4_env):
    """开发模式检查之后，再对各模型运行一次 --final。"""
    from gastric_mm.ct_mtl import predict as ct_pred
    from gastric_mm.fusion import predict as f_pred
    from gastric_mm.path_mtl import predict as p_pred
    for c in ("test", "external"):
        assert v4_env.run(p_pred.main, "--exp", "demo_path", "--cohort", c, "--final") == 0
        assert v4_env.run(ct_pred.main, "--exp", "demo_ct", "--cohort", c, "--final") == 0
        assert v4_env.run(f_pred.main, "--exp", "demo_fusion", "--cohort", c, "--final") == 0
        assert v4_env.run(f_pred.main, "--exp", "late_stack", "--cohort", c, "--final") == 0
    return v4_env


# ---------------- 派生规则 ----------------
def test_clinical_rules():
    from gastric_mm.evaluation.clinical import firstline_mapping, stage_from_text, stage_from_tnm, tx_pattern
    assert [stage_from_text(x) for x in ("cIIIA", "IIB", "IVA", "IVB", "4", "I", "")] == ["III", "II", "III", "IV", "IV", "I", None]
    assert stage_from_tnm("T2", "N0", "M0") == "I" and stage_from_tnm("T2", "N1", "M0") == "II"
    assert stage_from_tnm("T3", "N0", "M0") == "II" and stage_from_tnm("cT4a", "cN+", "M0") == "III"
    assert stage_from_tnm("T4b", "N0", "M0") == "III" and stage_from_tnm("T1", "N0", "M1") == "IV"
    assert stage_from_tnm("TX", "N0", "M0") is None
    assert tx_pattern(1.0, np.nan) == "neoadj_surgery" and tx_pattern(0.0, 1.0) == "surgery_adjuvant"
    assert tx_pattern(0.0, 0.0) == "surgery_only" and tx_pattern(0.0, np.nan) is None and tx_pattern(np.nan, 1.0) is None
    mp = firstline_mapping(pd.Series(["chemo"] * 50 + ["chemo_io"] * 40 + ["chemo_her2"] * 5 + ["none"] * 3), 30)
    assert mp["chemo_io"] == "chemo_io" and mp["chemo_her2"] == "chemo_other" and mp["none"] == "none"


def test_interaction_detects_known_effect():
    class Ctx:
        min_ev = 20
    rng = np.random.RandomState(0)
    n = 1500
    z = rng.normal(size=n)
    tr = rng.binomial(1, 0.7, n).astype(float)
    T = rng.exponential(1 / (0.02 * np.exp(0.8 * z + tr * (-0.2 - 0.5 * (z > 0)))))
    C = rng.uniform(6, 90, n)
    g = pd.DataFrame({"os_time": np.minimum(T, C), "os_event": (T <= C).astype(float), "stage_group": "I", "age": 60.0,
                      "ecog": 0.0, "center": "a"})
    r = prognosis.interaction_cox(g, "os", tr, z, "z × 治疗", Ctx())
    assert r["interaction_hr"] < 1 and r["p"] < 0.05


def test_neoadj_fit_recovers_signal():
    from gastric_mm.metrics import auc_binary
    rng = np.random.RandomState(1)
    n = 300
    X = rng.normal(size=(n, 6))
    y = (rng.rand(n) < 1 / (1 + np.exp(-(-1.0 + 1.5 * X[:, 0] - 1.0 * X[:, 1])))).astype(float)
    m = neoadj.fit(X[:200], y[:200], {})
    p = neoadj.predict(m, X[200:])
    assert auc_binary(y[200:], p) > 0.75
    assert abs(m["W"][0]) > abs(m["W"][3])                # 有信号的特征系数更大


# ---------------- 开发模式 ----------------
def test_development_mode(v4_env):
    assert v4_env.run(table.main) == 0
    d = v4_env.latest("table")
    assert "开发模式" in (d / "sources.md").read_text(encoding="utf-8")
    t = pd.read_csv(d / "analysis_table.csv")
    assert set(t.cohort) == {"internal_train"}
    tr = t[t.population == "resected"]
    assert abs(tr.risk_z.mean()) < 1e-6 and abs(tr.risk_z.std() - 1) < 1e-6      # 按内部训练集冻结的标准化
    assert v4_env.run(prognosis.main) == 0
    r = json.loads((v4_env.latest("prognosis") / "results.json").read_text(encoding="utf-8"))
    assert r["resected_os"]["success"]["evaluated"] is False


# ---------------- 完整流程 ----------------
def test_full_pipeline(v4_final):
    env = v4_final
    for mod in (table, classification, prognosis, nomogram, neoadj, immuno, explain, tables):
        assert env.run(mod.main) == 0, mod.__name__
    # 分析表：各队列齐全，不含任何路径
    d = env.latest("table")
    t = pd.read_csv(d / "analysis_table.csv")
    assert {"internal_train", "test", "ext1", "ext2", "ext3"} <= set(t.cohort)
    text = (d / "analysis_table.csv").read_text(encoding="utf-8-sig")
    assert "raw/" not in text and ".svs" not in text
    fz = json.loads((d / "frozen.json").read_text(encoding="utf-8"))
    assert "resected" in fz["risk_standardization_and_tertiles"]
    # 预后：外部合并队列判定成功标准；风险评分在内部训练集多因素 Cox 中显著
    r = json.loads((env.latest("prognosis") / "results.json").read_text(encoding="utf-8"))
    l1 = r["resected_os"]["layer1"]
    assert {"internal_train", "test", "external"} <= set(l1)
    assert r["resected_os"]["success"]["evaluated"] is True
    rz = [x for x in l1["internal_train"]["cox"]["table"] if x["term"] == "risk_z"][0]
    assert rz["hr"] > 1 and rz["p"] < 0.05
    assert l1["internal_train"]["cindex"][0] > 0.6
    assert "surgery_adjuvant" in r["resected_os"]["layer2"]["internal_train"]["treatment"]
    assert "landmark" in r["resected_os"]["layer2"]["internal_train"]["treatment"].get("surgery_only", {"landmark": 1})
    assert r["resected_os"]["layer3"]["centers"]
    assert any(row["term"].startswith("风险评分 × 辅助化疗") for row in r["resected_os"]["layer4"]["rows"])
    figs = list((env.latest("prognosis") / "figs").glob("km_os_resected_*.png"))
    assert len(figs) >= 3
    # 分类：主模型与对照的 DeLong 比较、外部截距重新校准
    c = json.loads((env.latest("classification") / "results.json").read_text(encoding="utf-8"))
    assert c["external"]["msi_h"]["delong"]
    assert "recalibration" in c["external"]["msi_h"]["calibration"]["fusion/demo_fusion"]
    assert c["internal_train"]["lauren"]["models"]["fusion/demo_fusion"]["auc"] > 0.7
    # 列线图：系数冻结并在各队列验证
    nj = json.loads((env.latest("nomogram") / "nomogram.json").read_text(encoding="utf-8"))
    assert "risk_z" in nj["resected"]["nomogram"]["terms"]
    assert (env.latest("nomogram") / "figs" / "nomogram_resected.png").exists()
    # 新辅助：三个模型在各队列都有结果（测试队列中新辅助病例很少，数值本身不作要求；拟合正确性见 test_neoadj_fit_recovers_signal）
    nr = json.loads((env.latest("neoadj") / "results.json").read_text(encoding="utf-8"))
    assert nr["endpoint"] in ("pcr", "mpr")
    assert {"clinical", "model", "combined"} <= set(nr["internal_train"])
    assert (env.latest("neoadj") / "neoadj_models.json").exists()
    # 免疫亚组：加权后协变量更均衡
    im = json.loads((env.latest("immuno") / "results.json").read_text(encoding="utf-8"))
    bal = im["first_line"].get("balance") or im["neoadj"].get("balance")
    if bal:
        assert max(abs(v) for v in bal["after"].values()) <= max(abs(v) for v in bal["before"].values()) + 1e-9
    # 可解释性图与论文用表
    cases = pd.read_csv(env.latest("explain") / "cases.csv")
    assert (cases.figures.str.contains("病理")).any()
    tb = env.latest("tables")
    for f in ("table1.md", "table1.csv", "flow.md", "tripod_ai.md"):
        assert (tb / f).exists()
    assert "外部 ext1" in (tb / "table1.md").read_text(encoding="utf-8")


def test_figure_language_fallback(v4_env):
    from gastric_mm.config import load_config
    from gastric_mm.evaluation.common import setup_figures
    from gastric_mm.stats import plots
    cfg = load_config(v4_env.config, ["v4.figure_language=zh"])
    setup_figures(cfg)
    assert plots.tr("中", "en") in (("中", "en") if plots.cjk_font_available() else ("en",))
    plots.set_language("en")
