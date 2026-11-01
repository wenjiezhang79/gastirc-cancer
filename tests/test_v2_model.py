"""v2 CT 单模态多任务模型与影像组学基线：模型单元、交叉验证训练、防泄漏、保存加载、最终评估、编码器比较、LASSO 求解器。

使用多模态假特征队列（fake_data.generate_mm_cohort，埋入已知信号），全部在 CPU 上运行。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from gastric_mm.ct_mtl import compare, predict, radiomics_model, train
from gastric_mm.utils.errors import UserError

REPO = Path(__file__).resolve().parent.parent


class V2Env:
    def __init__(self, base: Path):
        self.base = base
        self.config = base / "v2.yaml"
        self.out = base / "outputs"

    def run(self, main, *args):
        return main(["--config", str(self.config), *args])

    def exp(self, name):
        return self.out / "ct_mtl" / name

    @property
    def cfg(self):
        from gastric_mm.config import load_config
        return load_config(self.config)

    @property
    def split(self):
        return pd.read_csv(self.base / "data" / "work" / "splits" / "split_seed2026.csv")


@pytest.fixture(scope="session")
def v2_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2m")
    cfg = {
        "_base_": str(REPO / "configs" / "fake_v2.yaml"),
        "paths": {"root": str(base / "data"), "output_dir": str(base / "outputs")},
        "split": {"n_folds": 3},
        "fake_data": {"mm_cohort": {"n_internal": 240, "n_external": 30, "signal": 2.0}},
        "v2": {"train": {"epochs": 25, "lr": 5.0e-4}, "eval": {"n_boot": 100}},
    }
    (base / "v2.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    env = V2Env(base)
    from gastric_mm import fake_data
    assert env.run(fake_data.main, "--mm-cohort") == 0
    return env


@pytest.fixture(scope="session")
def v2_trained(v2_env):
    """同一实验分两次运行不同的折（模拟两张 GPU），第二次结束后自动汇总。"""
    assert v2_env.run(train.main, "--exp", "ct", "--folds", "0", "--no-aggregate") == 0
    assert not (v2_env.exp("ct") / "cv_report.md").exists()
    assert v2_env.run(train.main, "--exp", "ct", "--folds", "1,2") == 0
    return v2_env


# ---------------- 模型单元 ----------------
def _spec():
    from gastric_mm.training.task_spec import TaskSpec
    return TaskSpec(multiclass={"lauren": 3}, binary=["msi_h"], survival=["os"])


@pytest.mark.parametrize("agg", ["abmil", "meanpool"])
def test_token_model_padding_missing_and_grad(agg):
    from gastric_mm.mtl.model import TokenMTLModel
    torch.manual_seed(0)
    m = TokenMTLModel({"tumor": 8, "regional": 8}, {"aggregator": agg, "hidden_dim": 16, "attn_dim": 8, "dropout": 0.0},
                      _spec(), 4).eval()

    def batch(n_t, pad_t, present_r=True):
        xt = torch.randn(1, n_t, 8)
        xt_p = torch.cat([xt, torch.randn(1, pad_t, 8) * 100], 1)
        mt = torch.tensor([[True] * n_t + [False] * pad_t])
        xr = torch.randn(1, 3, 8) * 50
        return xt, {"x": {"tumor": xt_p, "regional": xr}, "mask": {"tumor": mt, "regional": torch.ones(1, 3, dtype=torch.bool)},
                    "present": {"tumor": torch.tensor([True]), "regional": torch.tensor([present_r])}}

    xt, b1 = batch(5, 0, False)
    b2 = {k: dict(v) for k, v in b1.items()}
    b2["x"]["tumor"] = torch.cat([xt, torch.randn(1, 4, 8) * 100], 1)
    b2["mask"]["tumor"] = torch.tensor([[True] * 5 + [False] * 4])
    b2["x"]["regional"] = torch.randn(1, 3, 8)                      # 缺失角色的内容不影响输出
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    assert torch.allclose(o1["logits:lauren"], o2["logits:lauren"], atol=1e-5)
    assert torch.allclose(o1["risk:os"], o2["risk:os"], atol=1e-5)
    m.train()
    out = m(b1)
    (out["logits:lauren"].sum() + out["risk:os"].sum()).backward()
    assert m.missing["regional"].grad is not None and m.missing["regional"].grad.abs().sum() > 0


def test_roi_dropout_keeps_one(v2_env):
    from gastric_mm.mtl.data import CTFeatureStore, TokenDataset
    cfg = v2_env.cfg
    store = CTFeatureStore(cfg, "fake_ct_a", ["tumor", "regional"])
    sids = store.available(list(v2_env.split.study_id[:40]))
    ds = TokenDataset(sids, [store], train=True, slice_dropout=0.5, roi_dropout=1.0)
    for i in range(len(ds)):
        _, f = ds[i]
        assert len(f) >= 1
        full = store.load(sids[i])
        for r, v in f.items():
            assert 1 <= len(v) <= len(full[r])


# ---------------- 交叉验证训练 ----------------
def test_ct_cv_outputs_and_learning(v2_trained):
    exp = v2_trained.exp("ct")
    for f in ("oof_predictions.csv", "cv_metrics.json", "cv_report.md", "postprocess.json", "oof_embeddings.h5",
              "experiment.json"):
        assert (exp / f).exists(), f
    m = json.loads((exp / "cv_metrics.json").read_text(encoding="utf-8"))
    main = {(r["task"], r["metric"]): r["value"] for r in m["oof"]}
    assert main[("lauren", "宏平均AUC")] > 0.7
    assert main[("os", "C-index")] > 0.6
    oof = pd.read_csv(exp / "oof_predictions.csv")
    assert {"has_tumor", "has_regional"} <= set(oof.columns)
    assert (oof.has_tumor == 0).any() or (oof.has_regional == 0).any()        # 有缺 ROI 的病例
    s = oof[["os_S12m", "os_S36m", "os_S60m"]].dropna().to_numpy()
    assert np.all(np.diff(s, axis=1) <= 1e-9)
    import h5py
    with h5py.File(exp / "seed0" / "fold0" / "attention.h5", "r") as f:
        g = f[list(f.keys())[0]]
        assert set(g.keys()) <= {"tumor", "regional"}
        a = g[list(g.keys())[0]][()]
        assert abs(float(a.sum()) - 1.0) < 1e-2
    with h5py.File(exp / "oof_embeddings.h5", "r") as f:
        assert f[list(f.keys())[0]].shape == (2, 256)                      # 2 个 ROI token，供 v3 使用


def test_ct_no_leakage(v2_trained):
    exp = v2_trained.exp("ct")
    sp = v2_trained.split
    train_ids = set(sp[sp.set == "train"].study_id)
    oof_ids = set()
    for k in range(3):
        c = json.loads((exp / "seed0" / f"fold{k}" / "case_ids.json").read_text(encoding="utf-8"))
        tr, va, ho = set(c["train"]), set(c["inner_val"]), set(c["heldout"])
        assert (tr | va | ho) <= train_ids
        assert not (tr & ho) and not (va & ho) and not (tr & va)
        oof_ids |= ho
    assert set(pd.read_csv(exp / "oof_predictions.csv").study_id) == oof_ids


def test_ct_training_never_reads_test_or_external(v2_env, monkeypatch):
    import gastric_mm.mtl.data as md
    seen = []
    orig = md.read_ct_features

    def spy(path):
        seen.append(Path(path).stem)
        return orig(path)
    monkeypatch.setattr(md, "read_ct_features", spy)
    assert v2_env.run(train.main, "--exp", "spy", "--folds", "0", "--no-aggregate", "--set", "v2.train.epochs=2") == 0
    sp = v2_env.split
    assert seen and not (set(seen) & set(sp[sp.set != "train"].study_id))


def test_ct_save_load_reproducible(v2_trained):
    from gastric_mm.mtl.data import CTFeatureStore
    from gastric_mm.mtl.model import load_token_model
    from gastric_mm.mtl.trainer import predict as do_predict
    from gastric_mm.path_mtl.common import load_labels_and_split
    env = v2_trained
    exp = env.exp("ct")
    cfg = env.cfg
    model, extra = load_token_model(exp / "seed0" / "fold1" / "model.pt")
    assert extra["encoder"] == "fake_ct_a" and model.roles == ["tumor", "regional"]
    held = json.loads((exp / "seed0" / "fold1" / "case_ids.json").read_text(encoding="utf-8"))["heldout"][:6]
    lab = load_labels_and_split(cfg, None)
    lab = lab[lab.study_id.isin(held)].reset_index(drop=True)
    p, attn, tok = do_predict(model, extra, [CTFeatureStore(cfg, "fake_ct_a", ["tumor", "regional"])], lab,
                              torch.device("cpu"), 4, [12, 36, 60])
    oof = pd.read_csv(exp / "seed0" / "fold1" / "oof.csv").set_index("study_id")
    for _, r in p.iterrows():
        assert r["lauren_p0"] == pytest.approx(oof.loc[r.study_id, "lauren_p0"], abs=1e-6)
        assert r["os_risk"] == pytest.approx(oof.loc[r.study_id, "os_risk"], abs=1e-6)
    assert tok.shape[1:] == (2, 256)


def test_ct_experiment_guard(v2_trained):
    assert v2_trained.run(train.main, "--exp", "ct", "--folds", "0", "--set", "v2.train.lr=0.1") == 1
    assert v2_trained.run(train.main, "--exp", "ct", "--folds", "0", "--encoder", "fake_ct_b") == 1
    assert v2_trained.run(train.main, "--exp", "bad", "--encoder", "no_such") == 1      # 没有特征 → 清楚报错


def test_ct_predict_final(v2_trained):
    env = v2_trained
    assert env.run(predict.main, "--exp", "ct", "--cohort", "test") == 1
    assert env.run(predict.main, "--exp", "ct", "--cohort", "test", "--final") == 0
    assert env.run(predict.main, "--exp", "ct", "--cohort", "test", "--final") == 1
    assert env.run(predict.main, "--exp", "ct", "--cohort", "external", "--final") == 0
    d = sorted(env.exp("ct").glob("final_external_*"))[-1]
    rep = (d / "report.md").read_text(encoding="utf-8")
    assert "ext1" in rep and "外部合并" in rep
    pred = pd.read_csv(d / "predictions.csv")
    sp = env.split
    assert set(pred.study_id) <= set(sp[sp.set == "external"].study_id)
    assert "os_high_risk" in pred and "msi_h_pred" in pred


@pytest.mark.parametrize("variant", [
    ["--set", "v2.model.aggregator=meanpool"],
    ["--set", "v2.loss.survival_loss=cox", "--set", "v2.loss.weighting=fixed"],
    ["--set", "v2.rois=[tumor]"],
    ["--set", "v2.tasks.binary=[ct_t34]", "--set", "v2.tasks.survival=[]", "--set", "v2.tasks.multiclass=[]"],
])
def test_ct_variants_run(v2_env, variant):
    name = "var_" + str(abs(hash(tuple(variant))) % 10 ** 8)
    assert v2_env.run(train.main, "--exp", name, "--folds", "0", "--no-aggregate", "--set", "v2.train.epochs=2",
                      *variant) == 0
    assert (v2_env.exp(name) / "seed0" / "fold0" / "oof.csv").exists()


# ---------------- 影像组学基线与比较 ----------------
def test_radiomics_baseline_and_compare(v2_env):
    env = v2_env
    assert env.run(radiomics_model.main, "--exp", "rad") == 0
    exp = env.exp("rad")
    for f in ("oof_predictions.csv", "cv_report.md", "postprocess.json", "radiomics_report.md"):
        assert (exp / f).exists(), f
    m = json.loads((exp / "cv_metrics.json").read_text(encoding="utf-8"))
    main = {(r["task"], r["metric"]): r["value"] for r in m["oof"]}
    assert main[("os", "C-index")] > 0.5
    model = json.loads((exp / "seed0" / "fold0" / "model.json").read_text(encoding="utf-8"))
    w = dict(zip(model["pre"]["names"], (v[0] for v in model["tasks"]["os"]["W"])))
    assert max(w, key=lambda k: abs(w[k])) in ("original_feat00", "original_feat01")   # 埋入的预后特征权重最大
    assert "original_const" not in model["pre"]["names"]                      # 常数特征被剔除
    assert not {"original_feat00", "original_feat01"} <= set(model["pre"]["names"])   # 高度相关只留一个
    rad_ids = set(pd.read_csv(env.base / "data" / "work" / "ct" / "radiomics" / "original_bw25" / "features.csv").study_id)
    assert set(pd.read_csv(exp / "oof_predictions.csv").study_id) <= rad_ids
    assert env.run(predict.main, "--exp", "rad", "--cohort", "test", "--final") == 0
    # 编码器初筛（预设 screening）+ 比较
    for e in ("fake_ct_a", "fake_ct_b"):
        assert env.run(train.main, "--exp", f"screen_{e}", "--profile", "screening", "--encoder", e, "--seeds", "0",
                       "--set", "v2.train.epochs=12") == 0
    rep = (env.exp("screen_fake_ct_a") / "cv_report.md").read_text(encoding="utf-8")
    assert "ct_t34" in rep and "msi_h" not in rep
    assert env.run(compare.main, "--exps", "screen_fake_ct_a,screen_fake_ct_b,rad", "--n-boot", "50") == 0
    d = sorted((env.out / "ct_mtl" / "_compare").iterdir())[-1]
    df = pd.read_csv(d / "compare.csv")
    assert list(df["rank"]) == [1, 2, 3] and "ct_t34" in df.columns
    assert df.loc[0, "experiment"] == "screen_fake_ct_a"                     # 噪声小的假编码器排第一
    assert "辅助参考" in (d / "compare_report.md").read_text(encoding="utf-8")
    assert env.run(compare.main, "--exps", "screen_fake_ct_a") == 1


# ---------------- LASSO 求解器 ----------------
def test_cox_gradient_and_ties():
    from gastric_mm.ct_mtl.linear import _cox
    rng = np.random.RandomState(0)
    n = 50
    lp, t = rng.normal(size=n), np.round(rng.exponential(10, size=n))
    e, s = (rng.rand(n) < 0.6).astype(float), (rng.rand(n) < 0.3).astype(int)
    _, g = _cox(lp, t, e, s)
    num = np.array([(_cox(lp + 1e-6 * np.eye(n)[i], t, e, s)[0] - _cox(lp - 1e-6 * np.eye(n)[i], t, e, s)[0]) / 2e-6
                    for i in range(n)])
    assert np.abs(num - g).max() < 1e-6


def test_l1_models_recover_signal():
    from gastric_mm.ct_mtl.linear import Preprocessor, fit_with_cv
    rng = np.random.RandomState(1)
    n, p = 300, 20
    X = rng.normal(size=(n, p))
    X[:, 5] = 3.0
    y = (X[:, 0] - X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(float)
    pre = Preprocessor(0.9).fit(X, [f"f{i}" for i in range(p)])
    assert "f5" not in pre.names
    Z = pre.transform(X)
    r = fit_with_cv("binary", Z, y, {"w": np.ones(n)}, 1)
    nz = {pre.names[i] for i in np.flatnonzero(np.abs(r["W"][:, 0]) > 0)}
    assert {"f0", "f1"} <= nz and len(nz) < 12
    t = rng.exponential(1 / np.exp(Z[:, 2]))
    ev = np.ones(n)
    r = fit_with_cv("survival", Z, t, {"time": t, "event": ev, "strata": np.zeros(n, int)}, 1)
    assert r["W"][pre.names.index("f2"), 0] > 0.3
