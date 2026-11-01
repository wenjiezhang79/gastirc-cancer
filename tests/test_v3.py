"""v3 融合：共用内层验证名单、模态可用性检查、融合模型单元、单模态初始化与防泄漏、交叉验证、后期融合、比较报告、最终评估。

使用多模态假特征队列（fake_data.generate_mm_cohort，含少量只有病理 / 只有 CT 的病例），全部在 CPU 上运行。
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from gastric_mm.fusion import check, compare, diagnose, late, predict, train
from gastric_mm.utils.errors import UserError

REPO = Path(__file__).resolve().parent.parent


class V3Env:
    def __init__(self, base: Path):
        self.base = base
        self.config = base / "v3.yaml"
        self.out = base / "outputs"
        self.work = base / "data" / "work"

    def run(self, main, *args):
        return main(["--config", str(self.config), *args])

    def exp(self, ns, name):
        return self.out / ns / name

    @property
    def cfg(self):
        from gastric_mm.config import load_config
        return load_config(self.config)

    @property
    def split(self):
        return pd.read_csv(self.work / "splits" / "split_seed2026.csv", dtype={"study_id": str})


@pytest.fixture(scope="session")
def v3_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("v3")
    cfg = {
        "_base_": str(REPO / "configs" / "fake_v3.yaml"),
        "paths": {"root": str(base / "data"), "output_dir": str(base / "outputs")},
        "split": {"n_folds": 3},
        "fake_data": {"mm_cohort": {"n_internal": 300, "n_external": 30, "signal": 2.0, "path_missing": 0.06}},
        "v1": {"train": {"epochs": 20, "lr": 5.0e-4}, "eval": {"n_boot": 50}},
        "v2": {"train": {"epochs": 20, "lr": 5.0e-4}, "eval": {"n_boot": 50}},
        "v3": {"init": {"path_exp": "p", "ct_exp": "c"}, "train": {"epochs": 20, "lr": 5.0e-4},
               "eval": {"n_boot": 50}, "compare": {"n_boot": 100}},
    }
    (base / "v3.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    env = V3Env(base)
    from gastric_mm import fake_data
    from gastric_mm.ct_mtl import train as ct_train
    from gastric_mm.path_mtl import train as path_train
    assert env.run(fake_data.main, "--mm-cohort") == 0
    assert env.run(path_train.main, "--exp", "p") == 0
    assert env.run(ct_train.main, "--exp", "c") == 0
    return env


@pytest.fixture(scope="session")
def v3_trained(v3_env):
    """同一融合实验分两次运行不同的折（模拟两张 GPU），第二次结束后自动汇总；另做后期融合。"""
    assert v3_env.run(train.main, "--exp", "f", "--folds", "0", "--no-aggregate") == 0
    assert not (v3_env.exp("fusion", "f") / "cv_report.md").exists()
    assert v3_env.run(train.main, "--exp", "f", "--folds", "1,2") == 0
    for m in ("mean", "stack"):
        assert v3_env.run(late.main, "--exp", f"late_{m}", "--method", m) == 0
    return v3_env


# ---------------- 共用内层验证名单 ----------------
def test_shared_inner_val_used_by_v1_v2_v3(v3_trained):
    env = v3_trained
    files = sorted((env.work / "splits" / "inner_val").glob("*.csv"))
    assert files, "共用内层验证名单文件未生成"
    shared = pd.read_csv(files[0], dtype={"study_id": str})
    for k in range(3):
        s = set(shared.loc[shared.heldout_fold == k, "study_id"])
        ids = {ns: json.loads((env.exp(ns, n) / "seed0" / f"fold{k}" / "case_ids.json").read_text(encoding="utf-8"))
               for ns, n in (("path_mtl", "p"), ("ct_mtl", "c"), ("fusion", "f"))}
        pools = {ns: set(c["train"]) | set(c["inner_val"]) | set(c["heldout"]) for ns, c in ids.items()}
        for ns, c in ids.items():
            assert set(c["inner_val"]) == s & pools[ns]          # 内层验证 = 共用名单 ∩ 该模态病例池
            assert not (set(c["train"]) & s)
        # 两种模态都有的病例：三方的内层验证完全相同 → 融合的内层验证病例两个分支都没训练过
        both = pools["path_mtl"] & pools["ct_mtl"]
        iv = [set(c["inner_val"]) & both for c in ids.values()]
        assert iv[0] == iv[1] == iv[2] and len(iv[0]) > 0
        assert not (set(ids["fusion"]["inner_val"]) & (set(ids["path_mtl"]["train"]) | set(ids["ct_mtl"]["train"])))


def test_shared_inner_val_mismatch_and_old_key(v3_env, tmp_path):
    from gastric_mm.inner_val import SharedInnerVal, inner_val_fraction
    from gastric_mm.path_mtl.common import load_labels_and_split
    cfg = v3_env.cfg
    df = load_labels_and_split(cfg, None)
    tr = df[df.set == "train"]
    iv = SharedInnerVal(cfg, tr)
    a = iv.ids(0, 0)
    # 行顺序不影响结果
    assert SharedInnerVal(cfg, tr.sample(frac=1.0, random_state=1)).compute(0)[0] == a
    p = iv.path(0)
    bak = p.read_text(encoding="utf-8")
    try:
        lines = bak.splitlines()
        p.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")   # 篡改名单
        with pytest.raises(UserError):
            SharedInnerVal(cfg, tr).ids(0, 0)
    finally:
        p.write_text(bak, encoding="utf-8")
    cfg2 = v3_env.cfg
    cfg2.v1.train["inner_val_fraction"] = 0.2
    with pytest.raises(UserError):
        inner_val_fraction(cfg2)


# ---------------- 模态可用性检查 ----------------
def test_availability_check(v3_env):
    assert v3_env.run(check.main) == 0
    run = sorted((v3_env.out / "fusion_check").iterdir())[-1]
    tab = pd.read_csv(run / "modality_availability.csv", dtype={"study_id": str})
    assert set(tab.group) >= {"both", "path_only", "ct_only"}
    assert tab.loc[tab.has_path == 0, "path_reason"].str.contains("slide_path").all()
    assert tab.loc[tab.has_ct == 0, "ct_reason"].str.contains("ct_dir").all()
    text = (run / "modality_availability.csv").read_text(encoding="utf-8-sig") + \
        (run / "availability_report.md").read_text(encoding="utf-8")
    assert "raw/" not in text and ".svs" not in text                 # 不输出任何路径
    assert "按集合与中心" in (run / "availability_report.md").read_text(encoding="utf-8")


# ---------------- 融合模型单元 ----------------
def _spec():
    from gastric_mm.training.task_spec import TaskSpec
    return TaskSpec(multiclass={"lauren": 3}, binary=["msi_h"], survival=["os"])


def _model(fusion, missing="mask", aux=False, skip=True):
    from gastric_mm.mtl.fusion_model import FusionModel
    torch.manual_seed(0)
    return FusionModel({"path": 8, "tumor": 6, "regional": 6},
                       {"fusion": fusion, "missing": missing, "skip": skip, "d_model": 16, "n_heads": 4, "dropout": 0.0,
                        "aux_heads": aux, "path": {"hidden_dim": 12, "attn_dim": 8, "n_branches": 2},
                        "ct": {"hidden_dim": 10, "attn_dim": 8}}, _spec(), 4)


def _batch(B=3):
    g = torch.Generator().manual_seed(1)
    return {"x": {"path": torch.randn(B, 7, 8, generator=g), "tumor": torch.randn(B, 4, 6, generator=g),
                  "regional": torch.randn(B, 5, 6, generator=g)},
            "mask": {"path": torch.ones(B, 7, dtype=torch.bool), "tumor": torch.ones(B, 4, dtype=torch.bool),
                     "regional": torch.ones(B, 5, dtype=torch.bool)},
            "present": {"path": torch.tensor([True, True, False]), "tumor": torch.tensor([True, False, True]),
                        "regional": torch.tensor([True, False, True])}}


@pytest.mark.parametrize("fusion,missing,skip", [("transformer", "mask", True), ("transformer", "mask", False),
                                                 ("transformer", "visible", True), ("concat", "mask", True),
                                                 ("bilinear", "mask", True)])
def test_fusion_model_missing_and_padding_invariance(fusion, missing, skip):
    m = _model(fusion, missing, skip=skip).eval()
    if fusion == "transformer":
        assert m.heads.trunk[1].in_features == m.d_model * (3 if skip else 1)
    b1 = _batch()
    b2 = {k: dict(v) for k, v in b1.items()}
    b2["x"] = {r: v.clone() for r, v in b1["x"].items()}
    b2["x"]["tumor"][1] *= 100                                      # 第 2 例缺 CT：内容不影响输出
    b2["x"]["regional"][1] += 50
    b2["x"]["path"][2] -= 30                                        # 第 3 例缺病理
    # 第 1 例病理补零：多补 3 个切块且 mask=False，不影响输出
    b2["x"]["path"] = torch.cat([b2["x"]["path"], torch.randn(3, 3, 8) * 100], 1)
    b2["mask"] = dict(b1["mask"], path=torch.cat([b1["mask"]["path"], torch.zeros(3, 3, dtype=torch.bool)], 1))
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    for k in ("logits:lauren", "logit:msi_h", "risk:os"):
        assert torch.allclose(o1[k], o2[k], atol=1e-5), k
    if fusion == "transformer":
        ex = o1["row_extras"]
        tot = ex["attn_mod_path"] + ex["attn_mod_ct"]
        assert torch.allclose(tot, torch.ones_like(tot), atol=1e-5)
        if missing == "mask":                                     # 屏蔽：缺失模态的注意力占比为 0
            assert float(ex["attn_mod_ct"][1]) == 0.0 and float(ex["attn_mod_path"][2]) == 0.0
    m.train()
    out = m(b1)
    (out["logits:lauren"].sum() + out["risk:os"].sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.branch["path"].parameters())


def test_aux_heads_and_freeze():
    m = _model("transformer", aux=True).train()
    out = m(_batch())
    seen = []

    def loss_fn(o, head, keep):
        seen.append(int(keep.sum()))
        return {"lauren": (o["logits:lauren"].sum() * 0 + o["logits:lauren"].pow(2).mean(), int(keep.sum()))}
    loss = m.aux_losses(out, loss_fn)
    loss.backward()
    assert seen == [2, 2]                      # 两种模态各有 2 例存在（病理：第 1、2 例；CT：第 1、3 例）
    assert m.aux_heads["ct"].trunk[1].weight.grad is not None
    m.set_branches_frozen(True)
    m.train()
    assert all(not p.requires_grad for p in m.branch.parameters())
    assert not m.branch.training and m.proj.training


def test_modality_dropout_never_both(v3_env):
    from gastric_mm.fusion.common import availability, make_stores, wrap_stores
    from gastric_mm.mtl.data import TokenDataset
    cfg = v3_env.cfg
    ps, cs = make_stores(cfg)
    sp = v3_env.split
    sids = list(sp.loc[sp.set == "train", "study_id"][:80])
    av = availability(sids, ps, cs)
    sids = list(av.loc[av.group != "none", "study_id"])
    groups = {"path": "path", "tumor": "ct", "regional": "ct"}
    ds = TokenDataset(sids, wrap_stores(ps, cs, av), train=True, roi_dropout=1.0, groups=groups, modality_dropout=1.0,
                      roi_roles=["tumor", "regional"], role_aug={"path": {"dropout": 0.5, "max_n": 10}})
    mods_seen = set()
    for i in range(len(ds)):
        _, f = ds[i]
        assert len(f) >= 1
        mods = {groups[r] for r in f}
        assert len(mods) == 1                    # 概率 1 时，两种都有的病例只剩一种模态；永远不会两种都遮
        mods_seen |= mods
        if "path" in f:
            assert len(f["path"]) <= 10
    assert mods_seen == {"path", "ct"}


# ---------------- 初始化与防泄漏 ----------------
def test_branch_init_loads_exact_weights(v3_env):
    from gastric_mm.fusion.common import SourceExp, load_branch_weights
    from gastric_mm.mtl.fusion_model import FusionModel
    from gastric_mm.mtl.spec import spec_from_section
    cfg = v3_env.cfg
    sp, sc = SourceExp(cfg, "path", "p"), SourceExp(cfg, "ct", "c")
    op, oc = sp.model_obj(0, 1), sc.model_obj(0, 1)
    mc = cfg.v3.model.to_dict()
    mc["path"], mc["ct"] = sp.branch_cfg(0, 1), sc.branch_cfg(0, 1)
    m = FusionModel({"path": op["in_dim"], "tumor": oc["in_dims"]["tumor"], "regional": oc["in_dims"]["regional"]},
                    mc, spec_from_section(cfg, "v3"), 4)
    load_branch_weights(m, sp, 0, 1)
    load_branch_weights(m, sc, 0, 1)
    sd = m.state_dict()
    for k, v in op["state_dict"].items():
        if k.startswith("aggregator."):
            assert torch.equal(sd["branch.path." + k[len("aggregator."):]], v)
    for k, v in oc["state_dict"].items():
        if k.startswith("branches."):
            assert torch.equal(sd["branch." + k[len("branches."):]], v)


def test_frozen_branches_keep_v1_weights(v3_env):
    assert v3_env.run(train.main, "--exp", "frz", "--folds", "0", "--no-aggregate",
                      "--set", "v3.train.freeze_branches=true", "--set", "v3.train.epochs=2") == 0
    from gastric_mm.fusion.common import SourceExp
    from gastric_mm.mtl.fusion_model import load_fusion_model
    m, _ = load_fusion_model(v3_env.exp("fusion", "frz") / "seed0" / "fold0" / "model.pt")
    src = SourceExp(v3_env.cfg, "path", "p").model_obj(0, 0)["state_dict"]
    sd = m.state_dict()
    for k, v in src.items():
        if k.startswith("aggregator."):
            assert torch.equal(sd["branch.path." + k[len("aggregator."):]], v)


def test_init_errors_seed_encoder_leakage(v3_env):
    # 种子不存在：直接报错（不悄悄换种子）
    assert v3_env.run(train.main, "--exp", "bad_seed", "--seeds", "7", "--folds", "0", "--no-aggregate") == 1
    from gastric_mm.fusion.common import SourceExp, check_source_encoders, check_source_fold
    cfg = v3_env.cfg
    cfg.v3["path_encoders"] = ["conch_v15"]
    with pytest.raises(UserError):
        check_source_encoders(SourceExp(cfg, "path", "p"), 0, 0, cfg, {"path": 32})
    # 防泄漏：复制 v1 实验，把融合第 0 折的一个留出病例写进其训练部分
    src = v3_env.exp("path_mtl", "p")
    dst = v3_env.exp("path_mtl", "p_leak")
    if not dst.exists():
        shutil.copytree(src, dst)
    cid = dst / "seed0" / "fold0" / "case_ids.json"
    c = json.loads(cid.read_text(encoding="utf-8"))
    leak = c["heldout"][0]
    c["train"].append(leak)
    cid.write_text(json.dumps(c), encoding="utf-8")
    s = SourceExp(v3_env.cfg, "path", "p_leak")
    sp = v3_env.split
    fold0 = set(sp.loc[(sp.set == "train") & (sp.fold == 0), "study_id"])
    with pytest.raises(UserError, match="防泄漏"):
        check_source_fold(s, 0, 0, {leak}, fold0, set())
    assert v3_env.run(train.main, "--exp", "bad_leak", "--folds", "0", "--no-aggregate",
                      "--set", "v3.init.path_exp=p_leak") == 1


# ---------------- 交叉验证 ----------------
def test_fusion_cv_outputs_and_complementarity(v3_trained):
    env = v3_trained
    exp = env.exp("fusion", "f")
    for f in ("oof_predictions.csv", "cv_metrics.json", "cv_report.md", "postprocess.json", "oof_embeddings.h5",
              "cv_fusion_extra.json", "experiment.json"):
        assert (exp / f).exists(), f
    oof = pd.read_csv(exp / "oof_predictions.csv")
    assert {"has_path", "has_tumor", "has_regional", "attn_mod_path", "attn_mod_ct"} <= set(oof.columns)
    assert (oof.has_path == 0).any() and ((oof.has_tumor == 0) & (oof.has_regional == 0)).any()
    rep = (exp / "cv_report.md").read_text(encoding="utf-8")
    for s in ("按模态可用性分层", "只有病理", "只有 CT", "模态贡献", "缺失鲁棒性"):
        assert s in rep, s

    def main_value(ns, name, task, metric):
        m = json.loads((env.exp(ns, name) / "cv_metrics.json").read_text(encoding="utf-8"))
        return {(r["task"], r["metric"]): r["value"] for r in m["oof"]}[(task, metric)]

    os_f, os_p, os_c = (main_value(ns, n, "os", "C-index") for ns, n in (("fusion", "f"), ("path_mtl", "p"), ("ct_mtl", "c")))
    # 假数据中预后 = 病理部分 + CT 部分：融合应不低于任何单模态（容差 0.02），且明显高于较弱的 CT
    assert os_f > os_c + 0.02 and os_f > os_p - 0.02, (os_f, os_p, os_c)
    assert main_value("fusion", "f", "lauren", "宏平均AUC") > 0.8
    assert main_value("fusion", "f", "msi_h", "AUC") > 0.8         # MSI 只在病理中：融合仍能学到
    extra = json.loads((exp / "cv_fusion_extra.json").read_text(encoding="utf-8"))
    assert extra["attention_share"]["path_only"]["ct"] == 0.0
    rob = extra["missing_robustness"]["msi_h"]
    assert rob["drop_path"] < rob["full"] - 0.1                  # 去掉病理后 MSI 明显变差


def test_training_never_reads_test_or_external(v3_env, monkeypatch):
    import gastric_mm.ct.features_io as cio
    import gastric_mm.mtl.data as mdata
    import gastric_mm.training.bag_dataset as bd
    sp = v3_env.split
    forbidden = set(sp.loc[sp.set != "train", "study_id"])
    seen = []
    orig_p, orig_c = bd.read_features, mdata.read_ct_features

    def spy_p(path):
        seen.append(Path(path).stem)
        return orig_p(path)

    def spy_c(path):
        seen.append(Path(path).stem)
        return orig_c(path)
    monkeypatch.setattr(bd, "read_features", spy_p)
    monkeypatch.setattr(mdata, "read_ct_features", spy_c)
    assert v3_env.run(train.main, "--exp", "spy", "--folds", "0", "--no-aggregate", "--set", "v3.train.epochs=1") == 0
    assert seen and not (set(seen) & forbidden)
    assert cio  # noqa


def test_save_load_reproducible(v3_trained):
    from gastric_mm.fusion.common import availability, make_stores, wrap_stores
    from gastric_mm.mtl.fusion_model import load_fusion_model
    from gastric_mm.mtl.trainer import predict as tpredict
    from gastric_mm.path_mtl.common import load_labels_and_split
    env = v3_trained
    fd = env.exp("fusion", "f") / "seed0" / "fold1"
    m, extra = load_fusion_model(fd / "model.pt")
    cfg = env.cfg
    df = load_labels_and_split(cfg, None)
    held = df[df.study_id.isin(json.loads((fd / "case_ids.json").read_text(encoding="utf-8"))["heldout"])]
    held = held.set_index("study_id").loc[json.loads((fd / "case_ids.json").read_text(encoding="utf-8"))["heldout"]].reset_index()
    ps, cs = make_stores(cfg)
    av = availability(list(held.study_id), ps, cs)
    p, _, _ = tpredict(m, extra, wrap_stores(ps, cs, av), held, torch.device("cpu"), 8, [12, 36, 60])
    ref = pd.read_csv(fd / "oof.csv", dtype={"study_id": str}).set_index("study_id").loc[p.study_id]
    for c in ("msi_h_prob", "os_risk", "lauren_p0"):
        assert np.allclose(p[c].to_numpy(), ref[c].to_numpy(), atol=1e-5)


def test_experiment_config_guard(v3_trained):
    assert v3_trained.run(train.main, "--exp", "f", "--folds", "0", "--set", "v3.model.d_model=64") == 1


@pytest.mark.parametrize("name,args", [
    ("v_concat", ["--set", "v3.model.fusion=concat"]),
    ("v_bilinear", ["--set", "v3.model.fusion=bilinear"]),
    ("v_visible", ["--set", "v3.model.missing=visible"]),
    ("v_noskip", ["--set", "v3.model.skip=false"]),
    ("v_aux", ["--set", "v3.model.aux_heads=true"]),
    ("v_tumoronly", ["--set", "v3.ct_rois=[tumor]"]),
    ("v_scratch", ["--set", "v3.init.path_exp=null", "--set", "v3.init.ct_exp=null"]),
    ("v_st_msi", ["--single-task", "msi_h"]),
    ("v_freeze2", ["--set", "v3.train.freeze_epochs=1"]),
])
def test_variants_run(v3_env, name, args):
    assert v3_env.run(train.main, "--exp", name, "--folds", "0", "--no-aggregate", "--set", "v3.train.epochs=2",
                      *args) == 0
    assert (v3_env.exp("fusion", name) / "seed0" / "fold0" / "oof.csv").exists()


# ---------------- 后期融合 ----------------
def test_late_fusion_crossfit_does_not_use_own_fold(v3_trained):
    from gastric_mm.mtl.spec import spec_from_section
    from gastric_mm.path_mtl.common import load_labels_and_split
    env = v3_trained
    cfg = env.cfg
    a = pd.read_csv(env.exp("path_mtl", "p") / "oof_predictions.csv", dtype={"study_id": str})
    b = pd.read_csv(env.exp("ct_mtl", "c") / "oof_predictions.csv", dtype={"study_id": str})
    spec, _ = late.common_spec(spec_from_section(cfg, "v3"), a, b)
    both = late.side_by_side(a, b)
    df = load_labels_and_split(cfg, None)
    lab = both[["study_id", "has_path", "has_ct"]].merge(df[df.set == "train"], on="study_id", how="left")
    folds = lab.fold.to_numpy(int)
    p1, _ = late.combine("stack", spec, both, lab, folds != 0, folds == 0, 1e-3)
    lab2 = lab.copy()
    m0 = folds == 0
    rng = np.random.RandomState(0)
    for c in ("lauren", "msi_h", "os_time", "os_event"):
        lab2.loc[m0, c] = rng.permutation(lab2.loc[m0, c].to_numpy())   # 打乱本折标签
    p2, _ = late.combine("stack", spec, both, lab2, folds != 0, folds == 0, 1e-3)
    for t in ("lauren", "msi_h", "os"):
        assert np.allclose(p1[t], p2[t], equal_nan=True)
    oof = pd.read_csv(env.exp("fusion", "late_stack") / "oof_predictions.csv")
    assert {"has_path", "has_ct", "os_risk", "msi_h_prob"} <= set(oof.columns)
    assert (env.exp("fusion", "late_stack") / "late_model.json").exists()


# ---------------- 比较报告 ----------------
def test_compare_report_rule(v3_trained):
    env = v3_trained
    assert env.run(train.main, "--exp", "st_os", "--single-task", "os", "--set", "v3.train.epochs=3") == 0
    assert env.run(compare.main, "--main", "f", "--late", "late_mean,late_stack",
                   "--others", "st_os,not_done", "--skip-missing") == 0
    run = sorted((env.out / "fusion_compare").iterdir())[-1]
    out = json.loads((run / "compare.json").read_text(encoding="utf-8"))
    assert out["primary"] == ["lauren", "her2_pos", "pdl1_cps5", "cldn182_pos", "os"]
    assert isinstance(out["passed"], bool)
    assert out["passed"] == (out["diff_vs_best_single"]["lo"] > 0 and out["diff_vs_late"]["lo"] > 0)
    assert {r["task"] for r in out["noninferiority"]} == {"msi_h", "ebv_pos"}
    for t, ref in out["best_single_by_task"].items():               # 每个任务取点估计较好的单模态
        row = [r for r in out["tasks"] if r["task"] == t][0]
        other = "CT 单模态" if ref == "病理单模态" else "病理单模态"
        assert row[ref] >= row[other] or not np.isfinite(row[other])
    rep = (run / "compare_report.md").read_text(encoding="utf-8")
    assert "预先规定的判定规则" in rep and "not_done" in rep
    # 不在 --skip-missing 时，缺失的实验报错；单任务模型不能作为主模型
    assert env.run(compare.main, "--main", "f", "--late", "late_mean", "--others", "not_done") == 1
    assert env.run(compare.main, "--main", "st_os", "--late", "late_mean") == 1


# ---------------- 最终评估 ----------------
def test_predict_final_guard_and_reports(v3_trained):
    env = v3_trained
    assert env.run(predict.main, "--exp", "f", "--cohort", "test") == 1              # 没有 --final
    assert env.run(predict.main, "--exp", "f", "--cohort", "external", "--final") == 0
    assert env.run(predict.main, "--exp", "f", "--cohort", "external", "--final") == 1   # 不允许重复
    d = sorted(env.exp("fusion", "f").glob("final_external_*"))[-1]
    rep = (d / "report.md").read_text(encoding="utf-8")
    assert "外部合并" in rep and "两种都有" in rep and "ext1" in rep
    pr = pd.read_csv(d / "predictions.csv")
    assert {"modality_group", "msi_h_pred", "os_high_risk"} <= set(pr.columns)
    assert env.run(predict.main, "--exp", "late_stack", "--cohort", "test", "--final") == 0
    d2 = sorted(env.exp("fusion", "late_stack").glob("final_test_*"))[-1]
    assert (d2 / "report.md").exists()
    reg = json.loads((env.exp("fusion", "f") / "final_evaluations.json").read_text(encoding="utf-8"))
    assert len(reg) == 1 and reg[0]["cohort"] == "external"


# ---------------- v1 / v2 不受影响 ----------------
def test_v1_v2_fingerprints_unaffected_by_v3(v3_env):
    from gastric_mm.mtl.experiment import fingerprint as fp2
    from gastric_mm.path_mtl.common import fingerprint as fp1
    cfg = v3_env.cfg
    a1, a2 = fp1(cfg), fp2(cfg, "v2", "token")
    cfg.v3.model["d_model"] = 256
    cfg.v3["ct_encoder"] = "other"
    cfg.v3.train["modality_dropout"] = 0.5
    assert fp1(cfg) == a1 and fp2(cfg, "v2", "token") == a2


# ---------------- 早停诊断 ----------------
def test_diagnose_report(v3_trained):
    env = v3_trained
    assert env.run(diagnose.main, "--exps", "path_mtl/p,ct_mtl/c,f") == 0
    run = sorted((env.out / "fusion_diagnose").iterdir())[-1]
    d = pd.read_csv(run / "diagnose.csv")
    assert set(d.experiment) == {"path_mtl/p", "ct_mtl/c", "fusion/f"}
    assert {"lauren", "msi_h", "os"} <= set(d.task)
    assert (d.task_best_epoch >= 0).all() and (d.task_best_epoch < d.n_epochs).all()
    assert (d.excess >= -1e-12).all()                         # 停止时的损失不会低于该任务自己的最低点
    # 与 done.json 中的最佳 epoch 一致
    best = json.loads((env.exp("fusion", "f") / "seed0" / "fold0" / "done.json").read_text(encoding="utf-8"))["best_epoch"]
    assert (d[(d.experiment == "fusion/f") & (d.fold == "fold0")].stop_epoch == best).all()
    assert "早停诊断" in (run / "diagnose_report.md").read_text(encoding="utf-8")
    assert env.run(diagnose.main, "--exps", "no_such_exp") == 1
