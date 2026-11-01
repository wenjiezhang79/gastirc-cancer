"""v2 CT 流程（体模假数据）：cT 列与标签、勾画抽样、胃分割封装、nnU-Net 数据准备/后处理/审核导入/Dice 汇总、
CT 特征提取（随机小网络与各编码器的权重加载）、隐私金丝雀。

外部工具（TotalSegmentator、nnU-Net）用 gastric_mm.fake_tools 中的假命令代替；全部在 CPU 上运行。
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gastric_mm import fake_data, split
from gastric_mm.ct import (convert, extract_features, import_reviewed, nnunet_data, seg_sample, stomach_seg,
                           tumor_post)

from conftest import FakeEnv, write_test_config
from test_privacy_and_env import _scan

PY = sys.executable


def sitk():
    import SimpleITK
    return SimpleITK


@pytest.fixture(scope="session")
def v2_ct(tmp_path_factory):
    """体模假数据（内部 12 例），胃分割真值移到 truth/，由假 TotalSegmentator 读取。"""
    base = tmp_path_factory.mktemp("v2ct")
    truth = base / "truth"
    extra = {
        "fake_data": {"n_per_center": {"internal": 12, "ext1": 2, "ext2": 1, "ext3": 1}},
        "v2": {
            "seg_sample": {"n": 4},
            "external": {"totalseg": {"command": [PY, "-m", "gastric_mm.fake_tools.totalseg", "--truth-dir", str(truth)]},
                         "nnunet": {"predict_command": [PY, "-m", "gastric_mm.fake_tools.nnunet_predict"]}},
            "features": {
                "slices_2d": {"tumor": 2, "regional": 3},
                "roi_specs": [
                    {"name": "hu_small", "center": "point_or_mask", "spacing": [1.5, 1.5, 5.0], "size_vox": [32, 32, 8],
                     "window": {"level": 512, "width": 3072}, "normalize": "none"},
                    {"name": "suprem_small", "center": "point_or_mask", "spacing": [2.0, 2.0, 2.0],
                     "size_vox": [64, 64, 64], "window": {"level": 37.5, "width": 425}, "normalize": "none"}],
                "encoders": {
                    "random_small_2d": {"feature_dim": 16, "rois": {"tumor": "tumor_small", "regional": "regional_small"}},
                    "random_small_3d": {"feature_dim": 16, "rois": {"tumor": "tumor_small", "regional": "regional_small"}},
                    "imagenet_r50": {"weights_path": str(base / "w" / "r50.pth"),
                                     "rois": {"tumor": "tumor_small", "regional": None}},
                    "radimagenet_r50": {"weights_path": str(base / "w" / "radr50.pth"),
                                        "rois": {"tumor": "tumor_small", "regional": None}},
                    "dinov2_vitb14": {"weights_path": str(base / "w" / "dino.pth"),
                                      "rois": {"tumor": "tumor_small", "regional": None}},
                    "ctfm": {"weights_path": str(base / "w" / "ctfm.pth"), "rois": {"tumor": "hu_small", "regional": None}},
                    "suprem": {"weights_path": str(base / "w" / "suprem.pth"), "rois": {"tumor": "suprem_small"}},
                }},
        },
    }
    write_test_config(base, extra)
    env = FakeEnv(base)
    assert env.run(fake_data.main) == 0
    assert env.run(split.main) == 0
    assert env.run(convert.main) == 0
    st = env.root / "work" / "ct" / "stomach"
    truth.mkdir()
    for f in st.glob("*.nii.gz"):
        shutil.move(str(f), truth / f.name)
    return env


# ---------------- 清单与标签 ----------------
def test_ctnm_columns_and_ct_t34(fake_env):
    from gastric_mm.labels import derive_labels
    from gastric_mm.manifest import read_manifest_csv, validate_manifest
    cfg = fake_env.cfg
    raw = read_manifest_csv(fake_env.root / "manifest.csv")
    raw.loc[0, "ct_t"] = "cT1b"
    raw.loc[1, "ct_t"] = "t4A"
    raw.loc[2, "ct_t"] = "TX"
    raw.loc[3, "ct_n"] = "cN+"
    raw.loc[1, "label_source"] = "post_neoadj"
    res = validate_manifest(raw, cfg, check_paths=False)
    assert res.n_errors == 0
    assert list(res.df.loc[:3, "ct_t"]) == ["T1b", "T4a", "TX", res.df.loc[3, "ct_t"]]
    lab = derive_labels(res.df, cfg)
    assert lab.loc[0, "ct_t_group"] == "T1" and lab.loc[0, "ct_t34"] == 0
    assert lab.loc[1, "ct_t_group"] == "T4" and lab.loc[1, "ct_t34"] == 1   # 新辅助后的标本不影响治疗前 cT
    assert pd.isna(lab.loc[2, "ct_t_group"]) and np.isnan(lab.loc[2, "ct_t34"])   # pandas 3 中缺失字符串为 NaN
    raw.loc[4, "ct_t"] = "T5"
    assert validate_manifest(raw, cfg, check_paths=False).n_errors == 1


def test_v1_fingerprint_unaffected_by_v2(v2_ct):
    """新增 v2 配置节不改变 v1 实验指纹（服务器上已有的 v1 实验不受影响）。"""
    import copy
    from gastric_mm.path_mtl.common import fingerprint
    cfg = v2_ct.cfg
    c2 = copy.deepcopy(cfg)
    del c2["v2"]
    assert fingerprint(cfg) == fingerprint(c2)


# ---------------- ① 勾画抽样 ----------------
def test_seg_sample(v2_ct):
    env = v2_ct
    assert env.run(seg_sample.main) == 0
    s = pd.read_csv(env.root / "work" / "ct" / "seg_sample.csv", encoding="utf-8-sig")
    sp = pd.read_csv(env.root / "work" / "splits" / "split_seed2026.csv")
    assert len(s) == 4
    assert set(s.study_id) <= set(sp[sp.set == "train"].study_id)       # 只从内部训练集抽
    assert int(s.already_manual.sum()) >= 1                              # 已有人工勾画的直接计入
    assert env.run(seg_sample.main) == 1                                 # 不覆盖已有名单
    assert env.run(seg_sample.main, "--overwrite") == 0


def test_seg_sample_allocation():
    from gastric_mm.ct.seg_sample import _allocate
    a = _allocate({"T1": 10, "T2": 30, "T3": 60}, 20, {"T1": 6})
    assert sum(a.values()) == 20 and a["T1"] >= 6 and a["T3"] > a["T2"]


# ---------------- ② 胃分割 ----------------
def test_stomach_seg_wrapper(v2_ct):
    env = v2_ct
    assert env.run(stomach_seg.main) == 0
    wp_st = env.root / "work" / "ct" / "stomach"
    S = sitk()
    truth = sorted((env.base / "truth").glob("*.nii.gz"))
    assert truth and all((wp_st / t.name).exists() for t in truth)
    a = S.GetArrayFromImage(S.ReadImage(str(wp_st / truth[0].name)))
    b = S.GetArrayFromImage(S.ReadImage(str(truth[0])))
    assert (a > 0).sum() == (b > 0).sum()
    run = env.latest_run("ct_stomach_seg")
    st = pd.read_csv(run / "status.csv", encoding="utf-8-sig")
    assert (st.status == "ok").all() and (st.stomach_ml > 10).all()
    assert list(run.glob("overview_stomach_p*.png"))


def test_stomach_seg_tool_failure_is_scrubbed(v2_ct, tmp_path):
    env = v2_ct
    cmd = yaml.safe_dump([PY, "-m", "gastric_mm.fake_tools.totalseg", "--truth-dir", str(tmp_path), "--fail"],
                         default_flow_style=True).strip()
    code = env.run(stomach_seg.main, "--overwrite", "--limit", "1", "--set", f"v2.external.totalseg.command={cmd}")
    assert code == 3
    st = pd.read_csv(env.latest_run("ct_stomach_seg") / "status.csv", encoding="utf-8-sig")
    assert st.status.iloc[0] == "failed" and "TotalSegmentator" in st.message.iloc[0]
    assert "work/ct" not in st.message.iloc[0]
    # 找不到命令时给出清楚的提示
    assert env.run(stomach_seg.main, "--set", "v2.external.totalseg.command=/no/such/TotalSegmentator") == 1
    # 恢复（后续测试需要胃分割）
    assert env.run(stomach_seg.main, "--overwrite") == 0


# ---------------- ② nnU-Net ----------------
def test_nnunet_train_dataset(v2_ct):
    env = v2_ct
    assert env.run(nnunet_data.main, "--mode", "train") == 0
    ds = env.root / "work" / "ct" / "nnunet" / "nnUNet_raw" / "Dataset501_GastricTumor"
    dj = json.loads((ds / "dataset.json").read_text())
    assert dj["channel_names"] == {"0": "CT", "1": "noNorm"} and dj["numTraining"] >= 1
    cmap = json.loads((ds / "case_map.json").read_text())
    sp = pd.read_csv(env.root / "work" / "splits" / "split_seed2026.csv")
    assert set(cmap.values()) <= set(sp[sp.set == "train"].study_id)    # 只用内部训练集的勾画
    S = sitk()
    for cid in cmap:
        img = S.ReadImage(str(ds / "imagesTr" / f"{cid}_0000.nii.gz"))
        lab = S.ReadImage(str(ds / "labelsTr" / f"{cid}.nii.gz"))
        assert img.GetSize() == lab.GetSize() and S.GetArrayFromImage(lab).sum() > 0
        assert (ds / "imagesTr" / f"{cid}_0001.nii.gz").exists()
    sh = (env.root / "work" / "ct" / "nnunet" / "nnunet_train.sh").read_text()
    assert 'nnUNetv2_plan_and_preprocess" -d 501' in sh and "nnUNetv2_train" in sh and "nnUNet_raw" in sh
    assert env.run(nnunet_data.main, "--mode", "train") == 1                # 已存在时不覆盖


def test_compute_box_includes_point_and_margin(v2_ct):
    from gastric_mm.ct.nnunet_data import compute_box
    S = sitk()
    cfg = v2_ct.cfg
    img = S.Image([60, 60, 20], S.sitkInt16)
    img.SetSpacing([2.0, 2.0, 5.0])
    st = np.zeros((20, 60, 60), np.uint8)
    st[8:12, 20:30, 20:30] = 1
    start, size, notes = compute_box(img, st, [50, 50, 10], cfg)
    end = np.array(start) + np.array(size) - 1
    assert start[0] <= 20 - 10 and end[0] >= 50 and start[2] <= 8 - 4
    with pytest.raises(Exception):
        compute_box(img, None, None, cfg)


def test_nnunet_predict_post_and_review(v2_ct, tmp_path):
    env = v2_ct
    assert env.run(nnunet_data.main, "--mode", "predict", "--run", "--include-manual") == 0
    out = env.root / "work" / "ct" / "nnunet" / "predict_output"
    assert list(out.glob("GC_*.nii.gz"))
    assert env.run(tumor_post.main) == 0
    auto = env.root / "work" / "ct" / "tumor_auto"
    rl = pd.read_csv(auto / "review_list.csv", encoding="utf-8-sig")
    assert len(rl) >= 5 and list(rl.priority) == sorted(rl.priority)
    assert rl["flags"].fillna("").str.contains("no_point").any()            # 有一例没有中心点
    # 自动分割与真值肿瘤（人工勾画）大体一致
    S = sitk()
    man = pd.read_csv(env.root / "manifest.csv", encoding="utf-8-sig")
    m = man[(man.mask_source == "manual") & man.ct_point_x.notna()].iloc[0]
    a = S.GetArrayFromImage(S.ReadImage(str(auto / f"{m.study_id}_tumor.nii.gz")))
    b = S.GetArrayFromImage(S.ReadImage(str(env.root / m.tumor_mask_path)))
    from gastric_mm.ct.import_reviewed import dice
    assert dice(a, b) > 0.6
    # 医生审核：一例原样保存、一例删掉一半
    rev = tmp_path / "reviewed"
    rev.mkdir()
    files = sorted(auto.glob("*_tumor.nii.gz"))
    shutil.copy(files[0], rev / files[0].name)
    img = S.ReadImage(str(files[1]))
    arr = S.GetArrayFromImage(img)
    zs = np.flatnonzero(arr.any(axis=(1, 2)))
    arr[zs[: max(1, len(zs) // 2)]] = 0
    o = S.GetImageFromArray(arr)
    o.CopyInformation(img)
    S.WriteImage(o, str(rev / files[1].name))
    assert env.run(import_reviewed.main, "--src-dir", str(rev)) == 0
    run = env.latest_run("ct_import_reviewed")
    st = pd.read_csv(run / "status.csv", encoding="utf-8-sig").set_index("study_id")
    sid0, sid1 = files[0].name.split("_tumor")[0], files[1].name.split("_tumor")[0]
    if "dice_vs_auto" in st and sid0 in st.index and sid1 in st.index:
        if st.loc[sid0, "status"] != "skipped":
            assert st.loc[sid0, "dice_vs_auto"] == pytest.approx(1.0)
        if st.loc[sid1, "status"] != "skipped":
            assert st.loc[sid1, "dice_vs_auto"] < 0.98
    mf = pd.read_csv(run / "private" / "manifest_with_reviewed.csv", encoding="utf-8-sig").set_index("study_id")
    orig = man.set_index("study_id")
    for sid in (sid0, sid1):
        if orig.loc[sid, "mask_source"] == "manual":
            assert mf.loc[sid, "mask_source"] == "manual"                   # 人工勾画不被覆盖
        else:
            assert mf.loc[sid, "mask_source"] == "auto_reviewed"


def test_select_component_rules(v2_ct):
    from gastric_mm.ct.tumor_post import select_component
    S = sitk()
    cfg = v2_ct.cfg
    native = S.Image([40, 40, 10], S.sitkUInt8)
    native.SetSpacing([1.0, 1.0, 5.0])
    a = np.zeros((10, 40, 40), np.uint8)
    a[3:7, 5:15, 5:15] = 1          # 肿瘤（2 ml）
    a[3:7, 28:38, 28:38] = 1        # 远处的另一块（2 ml）
    a[0, 0, 0] = 1                  # 噪声（< min_component_ml）
    m = S.GetImageFromArray(a)
    m.CopyInformation(native)
    box = {"start_xyz": [0, 0, 0], "size_xyz": [40, 40, 10]}
    r = select_component(m, native, [8, 8, 4], cfg, box)
    assert r["point_dist_mm"] == 0 and "multi" in r["flags"]
    assert S.GetArrayFromImage(r["mask"])[4, 8, 8] == 1 and S.GetArrayFromImage(r["mask"])[4, 34, 34] == 0
    r = select_component(m, native, [21, 21, 0], cfg, box)
    assert "not_found" in r["flags"] and r["mask"] is None
    r = select_component(m, native, [17, 8, 4], cfg, box)
    assert "nearest" in r["flags"] and r["point_dist_mm"] <= 4.0


def test_nnunet_summary(v2_ct):
    env = v2_ct
    ds = env.root / "work" / "ct" / "nnunet" / "nnUNet_raw" / "Dataset501_GastricTumor"
    cmap = json.loads((ds / "case_map.json").read_text())
    res = env.root / "work" / "ct" / "nnunet" / "nnUNet_results" / "Dataset501_GastricTumor" / \
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    for f, cid in enumerate(cmap):
        d = res / f"fold_{f}" / "validation"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"foreground_mean": {"Dice": 0.7}, "metric_per_case": [
            {"metrics": {"1": {"Dice": 0.7, "n_ref": 10, "n_pred": 9}},
             "reference_file": f"/x/gt_segmentations/{cid}.nii.gz"}]}))
    assert env.run(nnunet_data.main, "--mode", "summary") == 0
    run = env.latest_run("ct_nnunet_summary")
    rep = (run / "nnunet_cv_summary.md").read_text(encoding="utf-8")
    assert "Dice 均值 0.700" in rep and "按 cT 分组" in rep
    per = pd.read_csv(run / "dice_per_case.csv", encoding="utf-8-sig")
    assert set(per.study_id) == set(cmap.values())


# ---------------- ④ 特征提取 ----------------
def _save_weights(base: Path):
    import timm
    import torch
    w = base / "w"
    w.mkdir(exist_ok=True)
    torch.save(timm.create_model("resnet50", pretrained=False).state_dict(), w / "r50.pth")
    seq = torch.nn.Sequential(*list(timm.create_model("resnet50", pretrained=False).children())[:-1])
    torch.save(seq.state_dict(), w / "radr50.pth")
    torch.save(timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=518).state_dict(),
               w / "dino.pth")


@pytest.mark.parametrize("enc,roles,dim", [("random_small_2d", {"tumor": 5, "regional": 7}, 16),
                                           ("random_small_3d", {"tumor": 1, "regional": 1}, 16)])
def test_extract_features_random(v2_ct, enc, roles, dim):
    from gastric_mm.ct.features_io import read_ct_features
    env = v2_ct
    assert env.run(extract_features.main, "--encoder", enc) in (0, 3)
    d = env.root / "work" / "ct" / "features" / enc
    files = sorted(d.glob("*.h5"))
    assert len(files) >= 10
    rois, attrs = read_ct_features(files[0])
    assert {r: v.shape for r, v in rois.items()} == {r: (n, dim) for r, n in roles.items()}
    assert attrs["encoder"] == enc and attrs["tumor_spec"] == "tumor_small"
    st = pd.read_csv(env.latest_run("ct_features") / "status.csv", encoding="utf-8-sig")
    assert not (st.status == "failed").any()
    # 没有中心点的病例用分割质心
    man = pd.read_csv(env.root / "manifest.csv", encoding="utf-8-sig")
    nopt = man[man.ct_point_x.isna() & man.tumor_mask_path.notna() & man.ct_dir.notna()].study_id
    for sid in nopt:
        assert read_ct_features(d / f"{sid}.h5")[1]["tumor_center"] == "mask"


def test_extract_features_pretrained_loaders(v2_ct):
    from gastric_mm.ct.features_io import read_ct_features
    env = v2_ct
    _save_weights(env.base)
    for enc, dim in (("imagenet_r50", 2048), ("radimagenet_r50", 2048), ("dinov2_vitb14", 768)):
        assert env.run(extract_features.main, "--encoder", enc, "--limit", "1") == 0
        f = sorted((env.root / "work" / "ct" / "features" / enc).glob("*.h5"))[0]
        rois, attrs = read_ct_features(f)
        assert list(rois) == ["tumor"] and rois["tumor"].shape == (5, dim) and len(attrs["weights_sha256"]) == 64


def test_extract_features_monai_encoders(v2_ct):
    pytest.importorskip("monai")
    pytest.importorskip("einops")
    import torch
    from monai.networks.nets import SwinUNETR
    from monai.networks.nets.segresnet_ds import SegResEncoder
    from gastric_mm.ct.features_io import read_ct_features
    env = v2_ct
    w = env.base / "w"
    w.mkdir(exist_ok=True)
    enc = SegResEncoder(spatial_dims=3, in_channels=1, init_filters=32, blocks_down=(1, 2, 2, 4, 4))
    torch.save({"state_dict": {"module." + k: v for k, v in enc.state_dict().items()}}, w / "ctfm.pth")
    sw = SwinUNETR(in_channels=1, out_channels=32, feature_size=48)
    torch.save({"net": {"module.backbone." + k: v for k, v in sw.state_dict().items()}}, w / "suprem.pth")
    for name, dim in (("ctfm", 512), ("suprem", 768)):
        assert env.run(extract_features.main, "--encoder", name, "--limit", "1") == 0
        f = sorted((env.root / "work" / "ct" / "features" / name).glob("*.h5"))[0]
        assert read_ct_features(f)[0]["tumor"].shape == (1, dim)


def test_encoder_spec_checks(v2_ct):
    # 3D 基础模型配了 minmax 规格 → 清楚报错
    assert v2_ct.run(extract_features.main, "--encoder", "ctfm", "--limit", "1",
                     "--set", "v2.features.encoders.ctfm.rois.tumor=tumor_small") == 1
    # 权重路径不存在
    assert v2_ct.run(extract_features.main, "--encoder", "merlin") == 1
    assert v2_ct.run(extract_features.main, "--encoder", "no_such_encoder") == 1


def test_slices_2d_channels():
    from gastric_mm.ct.extract_features import slices_2d
    arr = np.arange(8, dtype=np.float32)[:, None, None] * np.ones((8, 4, 4), np.float32)
    imgs, rel = slices_2d(arr, 1, "adjacent")
    assert imgs.shape == (3, 3, 4, 4) and list(rel) == [-1, 0, 1]
    assert list(imgs[1, :, 0, 0]) == [3, 4, 5]
    imgs, rel = slices_2d(arr, None, "repeat")
    assert imgs.shape[0] == 8 and list(imgs[0, :, 0, 0]) == [0, 0, 0]


# ---------------- 隐私 ----------------
def test_v2_no_canary_leak(v2_ct):
    env = v2_ct
    canaries = [c for c in env.canaries if len(c) >= 2]
    hits = _scan([env.out, env.root / "work"], canaries)
    assert not hits, "\n".join(hits[:20])
    for log in env.out.rglob("run.log"):
        t = log.read_text(encoding="utf-8")
        assert "raw/ct/" not in t, log
