"""v1 病理多任务模型：损失、指标、聚合器、交叉验证训练、防泄漏、保存加载、最终评估、编码器比较。

使用埋入已知信号的假特征队列（fake_data.generate_feature_cohort），全部在 CPU 上运行。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from gastric_mm import metrics as M
from gastric_mm.losses import LossWeighter, cox_loss, discrete_time_nll, masked_bce, survival_curve
from gastric_mm.models.mil import build_aggregator
from gastric_mm.path_mtl import compare, predict, train
from gastric_mm.training.survival_bins import assign_bins, make_edges, survival_at
from gastric_mm.utils.errors import UserError

REPO = Path(__file__).resolve().parent.parent


# ---------------- 夹具：假特征队列 ----------------
class V1Env:
    def __init__(self, base: Path):
        self.base = base
        self.config = base / "v1.yaml"
        self.out = base / "outputs"

    def run(self, main, *args):
        return main(["--config", str(self.config), *args])

    def exp(self, name):
        return self.out / "path_mtl" / name

    @property
    def cfg(self):
        from gastric_mm.config import load_config
        return load_config(self.config)


@pytest.fixture(scope="session")
def v1_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v1")
    cfg = {
        "_base_": str(REPO / "configs" / "default.yaml"),
        "paths": {"root": str(base / "data"), "manifest": "manifest.csv", "work_dir": "work",
                  "output_dir": str(base / "outputs")},
        "split": {"n_folds": 3},
        "fake_data": {"feature_cohort": {"n_internal": 200, "n_external": 30, "tiles": [20, 80], "signal": 2.0}},
        "v1": {"train": {"epochs": 40, "batch_size": 16, "patience": 8, "lr": 5.0e-4}, "eval": {"n_boot": 100}},
    }
    (base / "v1.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    env = V1Env(base)
    from gastric_mm import fake_data
    assert env.run(fake_data.main, "--features-only") == 0
    return env


@pytest.fixture(scope="session")
def v1_trained(v1_env):
    """同一实验分两次运行不同的折（模拟两张 GPU），第二次结束后自动汇总。"""
    assert v1_env.run(train.main, "--exp", "mt", "--folds", "0", "--no-aggregate") == 0
    assert not (v1_env.exp("mt") / "cv_report.md").exists()
    assert v1_env.run(train.main, "--exp", "mt", "--folds", "1,2") == 0
    return v1_env


# ---------------- 损失 ----------------
def test_discrete_time_nll_hand_values():
    base = torch.zeros(1, 3)
    # 风险 0 → 各段 h = 0.5；事件在第 2 段：log(.5)*3
    l, n = discrete_time_nll(torch.zeros(1), base, torch.tensor([2]), torch.tensor([1.0]), torch.tensor([0]))
    assert n == 1 and float(l) == pytest.approx(-3 * np.log(0.5), rel=1e-5)
    # 在第 2 段删失：只计前 2 段存活
    l, _ = discrete_time_nll(torch.zeros(1), base, torch.tensor([2]), torch.tensor([0.0]), torch.tensor([0]))
    assert float(l) == pytest.approx(-2 * np.log(0.5), rel=1e-5)
    # 人群基线：第 1 个人群基线更高 → 同样的事件似然不同
    b2 = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    l0, _ = discrete_time_nll(torch.zeros(1), b2, torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0]))
    l1, _ = discrete_time_nll(torch.zeros(1), b2, torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1]))
    assert float(l1) < float(l0)
    # 全部缺失：损失为 0，n=0，反向传播不报错
    r = torch.zeros(2, requires_grad=True)
    l, n = discrete_time_nll(r, base, torch.tensor([-1, -1]), torch.tensor([1.0, 0.0]), torch.tensor([0, 0]))
    assert n == 0
    l.backward()
    s = survival_curve(torch.zeros(1), base, torch.tensor([0]))
    assert torch.allclose(s, torch.tensor([[0.5, 0.25, 0.125]]))


def test_cox_loss_matches_bruteforce():
    rng = np.random.RandomState(0)
    r, t, e = rng.normal(size=12), rng.exponential(size=12), (rng.rand(12) < 0.6).astype(float)
    l, _ = cox_loss(torch.tensor(r), torch.tensor(t), torch.tensor(e))
    ref = -sum(r[i] - np.log(np.exp(r[t >= t[i]]).sum()) for i in range(12) if e[i] == 1) / e.sum()
    assert float(l) == pytest.approx(ref, rel=1e-5)


def test_masked_bce_and_weighter():
    l, n = masked_bce(torch.tensor([0.0, 1.0]), torch.tensor([float("nan"), 1.0]))
    assert n == 1
    w = LossWeighter(["a", "b"], "uncertainty")
    tot = w({"a": (torch.tensor(1.0, requires_grad=True), 3), "b": (torch.tensor(5.0), 0)})
    assert float(tot.detach()) == pytest.approx(1.0)          # b 没有标签，整项跳过
    wf = LossWeighter(["a", "b"], "fixed", {"a": 2.0})
    assert float(wf({"a": (torch.tensor(1.0), 1), "b": (torch.tensor(3.0), 1)})) == pytest.approx(5.0)
    assert wf.monitor({"a": (torch.tensor(1.0), 1), "b": (torch.tensor(3.0), 1)}) == pytest.approx(2.5)


# ---------------- 指标 ----------------
def test_metrics_known_values():
    assert M.auc_binary([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert M.auc_binary([0, 1, np.nan], [0.2, 0.9, 0.5]) == 1.0
    assert np.isnan(M.auc_binary([1, 1], [0.1, 0.2]))
    assert M.c_index([1, 2, 3, 4], [1, 1, 1, 0], [4, 3, 2, 1]) == pytest.approx(1.0)
    assert M.c_index([1, 2, 3, 4], [1, 1, 1, 0], [1, 2, 3, 4]) == pytest.approx(0.0)
    assert M.c_index([1, 2], [1, 1], [1, 1]) == pytest.approx(0.5)
    assert M.pr_auc([0, 1, 1], [0.1, 0.9, 0.8]) == pytest.approx(1.0)
    macro, per = M.multiclass_auc([0, 1, 2, 2], np.eye(3)[[0, 1, 2, 2]])
    assert macro == 1.0 and per == [1.0, 1.0, 1.0]
    y = np.array([0, 1] * 30)
    s = y + np.random.RandomState(0).normal(0, 1, 60)
    a = M.bootstrap_ci(M.auc_binary, [y, s], 200, seed=1, strata=y)
    b = M.bootstrap_ci(M.auc_binary, [y, s], 200, seed=1, strata=y)
    assert a == b and a[1] < a[0] < a[2]
    d = M.paired_bootstrap_diff(M.auc_binary, [y, s], [y, np.zeros(60)], 200)
    assert d["diff"] > 0 and d["lo"] > 0


def test_metrics_empty_groups(v1_env):
    """某组某任务一个标签都没有（如外部某中心没有不可切除患者 → 无 PFS）时，指标为 NaN 而不是报错。"""
    from gastric_mm.path_mtl.evaluate import evaluate
    from gastric_mm.training.task_spec import TaskSpec
    e = np.array([])
    for fn, args in ((M.auc_binary, (e, e)), (M.pr_auc, (e, e)), (M.c_index, (e, e, e)), (M.spearman, (e, e)),
                     (M.youden_threshold, (e, e))):
        assert np.isnan(fn(*args))
    assert np.isnan(M.multiclass_auc(e, np.zeros((0, 3)))[0])
    assert np.isnan(M.bootstrap_ci(M.auc_binary, [e, e], 50)[1])
    assert np.isnan(M.paired_bootstrap_diff(M.auc_binary, [e, e], [e, e], 50)["lo"])
    spec = TaskSpec(multiclass={"lauren": 3}, binary=["msi_h"], survival=["os", "pfs"], population_baselines=["os"])
    rng = np.random.RandomState(0)
    n = 30
    df = pd.DataFrame({"study_id": [f"C2-{i:04d}" for i in range(n)], "lauren": rng.randint(0, 3, n).astype(float),
                       "lauren_intestinal": np.nan, "msi_h": np.nan, "msi_h_prob": rng.rand(n),
                       "os_time": rng.exponential(30, n), "os_event": (rng.rand(n) < 0.5).astype(float),
                       "os_risk": rng.normal(size=n), "pfs_time": np.nan, "pfs_event": np.nan, "pfs_risk": rng.normal(size=n),
                       "population": "resected"})
    for k in range(3):
        df[f"lauren_p{k}"] = rng.rand(n)
    rows = evaluate(df, spec, v1_env.cfg, n_boot=50, thresholds={"msi_h": 0.5})
    got = {(r["task"], r["metric"]): r for r in rows}
    assert np.isnan(got[("msi_h", "AUC")]["value"]) and got[("msi_h", "AUC")]["n"] == 0
    assert np.isnan(got[("pfs", "C-index")]["value"]) and got[("pfs", "C-index")]["n"] == 0
    assert np.isfinite(got[("os", "C-index")]["value"])


# ---------------- 聚合器 ----------------
@pytest.mark.parametrize("name", ["abmil", "transmil", "meanpool"])
def test_aggregator_padding_invariance_and_grad(name):
    torch.manual_seed(0)
    agg = build_aggregator(name, 12, {"hidden_dim": 16, "n_branches": 3, "dropout": 0.0}).eval()
    x = torch.randn(2, 25, 12)
    mask = torch.ones(2, 25, dtype=torch.bool)
    mask[1, 15:] = False
    tok, attn = agg(x, mask)
    tok1, _ = agg(x[1:2, :15], torch.ones(1, 15, dtype=torch.bool))
    assert torch.allclose(tok[1], tok1[0], atol=1e-5)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    assert float(attn[1, :, 15:].detach().abs().sum()) == 0
    agg.train()
    agg(x, mask)[0].sum().backward()
    assert any(p.grad is not None for p in agg.parameters())


# ---------------- 生存时间段 ----------------
def test_survival_bins():
    t = np.array([1, 2, 3, 5, 8, 13, 21, 34, np.nan])
    e = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1])
    edges = make_edges(t, e, 4)
    assert edges[0] == 0 and np.isinf(edges[-1]) and np.all(np.diff(edges[:-1]) > 0)
    b = assign_bins(t, edges)
    assert b[-1] == -1 and b.max() == 3 and b[:-1].min() == 0
    surv = np.array([[0.9, 0.7, 0.5, 0.3]])
    s = survival_at(surv, [0, 10, 20, 30, np.inf], [5, 15, 25, 60])
    assert np.all(np.diff(s[0]) < 0) and s[0, 0] < 1 and s[0, -1] < 0.3


def test_task_spec_validation(v1_env):
    from gastric_mm.training.task_spec import TaskSpec
    cfg = v1_env.cfg
    cfg.v1.tasks["binary"] = ["msi_h", "not_a_task"]
    with pytest.raises(UserError):
        TaskSpec.from_config(cfg)


# ---------------- 交叉验证训练 ----------------
def test_cv_outputs_and_learning(v1_trained):
    exp = v1_trained.exp("mt")
    for f in ("oof_predictions.csv", "cv_metrics.json", "cv_report.md", "postprocess.json", "oof_attention.h5",
              "oof_embeddings.h5"):
        assert (exp / f).exists(), f
    m = json.loads((exp / "cv_metrics.json").read_text(encoding="utf-8"))
    main = {(r["task"], r["metric"]): r["value"] for r in m["oof"]}
    assert main[("lauren", "宏平均AUC")] > 0.75
    assert main[("msi_h", "AUC")] > 0.75
    assert main[("os", "C-index")] > 0.6
    oof = pd.read_csv(exp / "oof_predictions.csv")
    s = oof[["os_S12m", "os_S36m", "os_S60m"]].dropna().to_numpy()
    assert np.all(np.diff(s, axis=1) <= 1e-9)
    post = json.loads((exp / "postprocess.json").read_text(encoding="utf-8"))
    assert "msi_h" in post["binary_thresholds_youden"]


def test_no_leakage_case_ids(v1_trained):
    exp = v1_trained.exp("mt")
    split = pd.read_csv(v1_trained.base / "data" / "work" / "splits" / "split_seed2026.csv")
    train_ids = set(split[split.set == "train"].study_id)
    oof_ids = set()
    for k in range(3):
        c = json.loads((exp / "seed0" / f"fold{k}" / "case_ids.json").read_text(encoding="utf-8"))
        tr, va, ho = set(c["train"]), set(c["inner_val"]), set(c["heldout"])
        assert (tr | va | ho) <= train_ids                 # 只用内部训练集
        assert not (tr & ho) and not (va & ho) and not (tr & va)
        assert 0.08 < len(va) / (len(tr) + len(va)) < 0.22
        oof_ids |= ho
    assert set(pd.read_csv(exp / "oof_predictions.csv").study_id) == oof_ids


def test_training_never_reads_test_or_external(v1_env, monkeypatch):
    import gastric_mm.training.bag_dataset as bd
    seen = []
    orig = bd.read_features

    def spy(path):
        seen.append(Path(path).stem)
        return orig(path)
    monkeypatch.setattr(bd, "read_features", spy)
    assert v1_env.run(train.main, "--exp", "spy", "--folds", "0", "--no-aggregate",
                      "--set", "v1.train.epochs=2") == 0
    split = pd.read_csv(v1_env.base / "data" / "work" / "splits" / "split_seed2026.csv")
    forbidden = set(split[split.set != "train"].study_id)
    assert seen and not (set(seen) & forbidden)


def test_save_load_reproducible(v1_trained):
    from gastric_mm.config import load_config
    from gastric_mm.models.path_model import load_model
    from gastric_mm.path_mtl.common import load_labels_and_split
    from gastric_mm.training.bag_dataset import FeatureStore
    from gastric_mm.training.trainer import predict as do_predict
    exp = v1_trained.exp("mt")
    cfg = load_config(v1_trained.config)
    model, extra = load_model(exp / "seed0" / "fold1" / "model.pt")
    df = load_labels_and_split(cfg, None)
    held = json.loads((exp / "seed0" / "fold1" / "case_ids.json").read_text(encoding="utf-8"))["heldout"][:5]
    lab = df[df.study_id.isin(held)].reset_index(drop=True)
    p, attn, tok = do_predict(model, extra, FeatureStore(cfg, extra["encoders"]), lab, torch.device("cpu"), 4,
                              [12, 36, 60])
    oof = pd.read_csv(exp / "seed0" / "fold1" / "oof.csv").set_index("study_id")
    for _, r in p.iterrows():
        assert r["msi_h_prob"] == pytest.approx(oof.loc[r.study_id, "msi_h_prob"], abs=1e-6)
        assert r["os_risk"] == pytest.approx(oof.loc[r.study_id, "os_risk"], abs=1e-6)
    assert all(a.shape[0] == 2 for a in attn.values())   # 2 个注意力分支


def test_experiment_config_guard(v1_trained):
    code = v1_trained.run(train.main, "--exp", "mt", "--folds", "0", "--set", "v1.train.lr=0.1")
    assert code == 1


# ---------------- 最终评估 ----------------
def test_predict_final_guard_and_reports(v1_trained):
    env = v1_trained
    assert env.run(predict.main, "--exp", "mt", "--cohort", "test") == 1          # 没有 --final
    assert env.run(predict.main, "--exp", "mt", "--cohort", "test", "--final") == 0
    assert env.run(predict.main, "--exp", "mt", "--cohort", "test", "--final") == 1   # 重跑被拒绝
    assert env.run(predict.main, "--exp", "mt", "--cohort", "test", "--final", "--allow-rerun") == 0
    reg = json.loads((env.exp("mt") / "final_evaluations.json").read_text(encoding="utf-8"))
    assert [r["rerun"] for r in reg] == [False, True]
    assert env.run(predict.main, "--exp", "mt", "--cohort", "external", "--final") == 0
    d = sorted(env.exp("mt").glob("final_external_*"))[-1]
    rep = (d / "report.md").read_text(encoding="utf-8")
    assert "ext1" in rep and "外部合并" in rep and "postprocess.json" in rep
    pred = pd.read_csv(d / "predictions.csv")
    split = pd.read_csv(env.base / "data" / "work" / "splits" / "split_seed2026.csv")
    assert set(pred.study_id) <= set(split[split.set == "external"].study_id)
    assert "msi_h_pred" in pred and "os_high_risk" in pred


# ---------------- 编码器比较、拼接与各种变体 ----------------
def test_screening_profile_and_compare(v1_env):
    for e in ("uni2_h", "conch_v15"):
        assert v1_env.run(train.main, "--exp", f"screen_{e}", "--profile", "screening", "--encoders", e,
                          "--seeds", "0", "--set", "v1.train.epochs=6") == 0
    rep = (v1_env.exp("screen_uni2_h") / "cv_report.md").read_text(encoding="utf-8")
    assert "msi_h" in rep and "her2_pos" not in rep
    assert v1_env.run(compare.main, "--pattern", "screen_*", "--n-boot", "50") == 0
    d = sorted((v1_env.out / "path_mtl" / "_compare").iterdir())[-1]
    df = pd.read_csv(d / "compare.csv")
    assert list(df["rank"]) == [1, 2] and np.isnan(df.loc[0, "diff_lo"]) and np.isfinite(df.loc[1, "diff_lo"])


def test_concat_encoders_and_coord_mismatch(v1_env, tmp_path):
    from gastric_mm.models.path_model import load_model
    assert v1_env.run(train.main, "--exp", "concat", "--encoders", "uni2_h,conch_v15", "--folds", "0",
                      "--no-aggregate", "--set", "v1.train.epochs=2") == 0
    m, extra = load_model(v1_env.exp("concat") / "seed0" / "fold0" / "model.pt")
    assert m.in_dim == 32 + 16 and extra["encoders"] == ["uni2_h", "conch_v15"]
    # 坐标不一致时报错
    import shutil
    import h5py
    work2 = tmp_path / "work"
    shutil.copytree(v1_env.base / "data" / "work", work2)
    f = sorted((work2 / "wsi" / "features" / "conch_v15").glob("C1-*.h5"))[0]
    with h5py.File(f, "a") as h:
        c = h["coords"][()]
        c[0, 0] += 1
        del h["coords"]
        h.create_dataset("coords", data=c)
    code = v1_env.run(train.main, "--exp", "concat_bad", "--encoders", "uni2_h,conch_v15", "--folds", "0",
                      "--no-aggregate", "--set", f"paths.work_dir={work2}", "--set", "v1.train.epochs=1")
    assert code == 1


@pytest.mark.parametrize("variant", [
    ["--set", "v1.model.aggregator=transmil"],
    ["--set", "v1.model.aggregator=meanpool"],
    ["--set", "v1.loss.survival_loss=cox"],
    ["--set", "v1.loss.weighting=fixed", "--set", "v1.loss.lauren_class_weight=true"],
    ["--set", "v1.tasks.multiclass=[]", "--set", "v1.tasks.binary=[msi_h]", "--set", "v1.tasks.survival=[]"],
    ["--set", "v1.tasks.regression=[ki67_frac]", "--set", "v1.tasks.multiclass=[lauren,her2_3cls]"],
])
def test_variants_run(v1_env, variant):
    name = "var_" + str(abs(hash(tuple(variant))) % 10 ** 8)
    assert v1_env.run(train.main, "--exp", name, "--folds", "0", "--no-aggregate",
                      "--set", "v1.train.epochs=2", *variant) == 0
    assert (v1_env.exp(name) / "seed0" / "fold0" / "oof.csv").exists()
