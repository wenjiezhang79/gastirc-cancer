"""病理：组织检测、切块、特征提取（随机小网络）、编码器构建、TRIDENT 导入。"""

import numpy as np
import pytest

from gastric_mm.paths import WorkPaths
from gastric_mm.utils.errors import UserError
from gastric_mm.wsi import extract_features, import_trident
from gastric_mm.wsi.extractors import build_extractor
from gastric_mm.wsi.features_io import read_features, read_tiles
from gastric_mm.wsi.tiling import plan_tiles, tile_spec_name
from gastric_mm.wsi.tissue import detect_tissue, otsu_threshold


# ---------------- 组织检测（单元） ----------------
def test_tissue_detection_excludes_pen(fake_env):
    from gastric_mm.fake_data import make_slide_image
    rng = np.random.RandomState(0)
    img = make_slide_image(768, 512, rng)
    m, info = detect_tissue(img, 2.0, fake_env.cfg.wsi.tissue)
    sat = img.max(2).astype(int) - img.min(2).astype(int)
    pink = (sat > 40) & ~((img[..., 2].astype(int) - img[..., 0].astype(int)) > 80)
    assert (m & pink).sum() / pink.sum() > 0.9
    pen_rows = slice(int(512 * 0.9), int(512 * 0.9) + 12)
    assert m[pen_rows].sum() == 0
    assert info["n_fragments"] >= 2


def test_otsu_and_plan_tiles():
    v = np.concatenate([np.full(100, 10), np.full(100, 200)]).astype(np.uint8)
    assert 10 <= otsu_threshold(v) < 200
    mask = np.zeros((100, 100), bool)
    mask[:50, :50] = True
    # 掩码 1 px = 8 µm；level0 0.5 µm/px，1600×1600 px；切块 224 px @0.5 → 112 µm = 14 掩码像素
    coords, frac = plan_tiles(mask, 8.0, 0.5, (1600, 1600), 0.5, 224, 0.5)
    assert len(coords) > 0 and (coords < 800).all() and (frac >= 0.5).all()
    # 40× 切片：同样的物理切块，level-0 尺寸加倍
    c40, _ = plan_tiles(mask, 8.0, 0.25, (3200, 3200), 0.5, 224, 0.5)
    assert len(c40) == len(coords) and np.allclose(c40, coords * 2, atol=1)


# ---------------- 切块 ----------------
def test_tiling(wsi_tiled):
    env = wsi_tiled
    cfg = env.cfg
    wp = WorkPaths(cfg)
    spec = tile_spec_name(cfg.wsi.tiling.target_mpp, cfg.wsi.tiling.tile_px)
    files = sorted((wp.wsi / "tiles" / spec).glob("*.h5"))
    assert files
    sizes = {}
    for f in files:
        coords, attrs = read_tiles(f)
        assert len(coords) > 0
        sizes[round(attrs["base_mpp"], 2)] = attrs["tile_size_level0"]
        assert wp.wsi_qc(spec, f.stem).exists()
    assert sizes[0.25] == pytest.approx(2 * sizes[0.5])      # 40× 与 20× 物理尺寸一致
    assert list(env.latest_run("wsi_tiling").glob("overview_p*.png"))


# ---------------- 特征提取 ----------------
def test_feature_extraction_and_resume(features_extracted):
    env = features_extracted
    wp = WorkPaths(env.cfg)
    files = sorted((wp.wsi / "features" / "random_small").glob("*.h5"))
    assert files
    f, c, a = read_features(files[0])
    assert f.shape[1] == 64 and len(f) == len(c) and np.isfinite(f).all()
    assert a["source"] == "gastric_mm" and a["encoder"] == "random_small"
    assert env.run(extract_features.main) == 0
    st = (env.latest_run("wsi_features") / "status.csv").read_text(encoding="utf-8-sig")
    assert "skipped" in st


def test_random_extractor_forward_backward(fake_env):
    import torch
    ext = build_extractor("random_small", fake_env.cfg)
    x = torch.randn(2, 3, 224, 224)
    y = ext.forward(ext.model, x)
    assert y.shape == (2, 64)
    # 反向：解除冻结后梯度可以回传
    for p in ext.model.parameters():
        p.requires_grad_(True)
    loss = ext.forward(ext.model, x).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in ext.model.parameters())


def test_missing_weights_message(fake_env):
    with pytest.raises(UserError) as e:
        build_extractor("uni2_h", fake_env.cfg)
    assert "weights_path" in e.value.hint
    cfg = fake_env.cfg
    cfg.features.extractors["virchow2"]["weights_path"] = "/no/such/file.bin"
    with pytest.raises(UserError) as e:
        build_extractor("virchow2", cfg)
    assert "不会联网" in e.value.hint


def test_resnet50_trunc_with_local_weights(fake_env, tmp_path):
    import timm
    import torch
    sd = timm.create_model("resnet50", pretrained=False).state_dict()   # 与 torchvision 键名一致
    p = tmp_path / "resnet50.pth"
    torch.save(sd, p)
    cfg = fake_env.cfg
    cfg.features.extractors["resnet50_trunc"]["weights_path"] = str(p)
    ext = build_extractor("resnet50_trunc", cfg)
    y = ext.forward(ext.model, torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 1024) and ext.weights_sha256


@pytest.mark.parametrize("arch,kw_name,dim", [
    ("vit_giant_patch14_224", "uni2_h", 1536),
    ("vit_huge_patch14_224", "virchow2", 1280),
    ("vit_giant_patch14_dinov2", "gigapath", 1536),
])
def test_foundation_model_architectures_meta(arch, kw_name, dim):
    """检查结构参数在当前 timm 中可用、输出维数正确。
    优先在 meta 设备上构建完整模型（不占内存）；旧版 torch 不支持时，改用 2 层的同参数模型在 CPU 上检查。"""
    import torch
    import timm
    kws = {
        "uni2_h": dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                       mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                       act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True),
        "virchow2": dict(img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375, reg_tokens=4, global_pool="",
                         dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU),
        "gigapath": dict(img_size=224, in_chans=3, patch_size=16, embed_dim=1536, depth=40, num_heads=24,
                         init_values=1e-5, mlp_ratio=5.33334, num_classes=0),
    }
    kw = kws[kw_name]
    try:
        with torch.device("meta"):
            m = timm.create_model(arch, pretrained=False, **kw)
            out = m(torch.zeros(1, 3, 224, 224))
    except (NotImplementedError, RuntimeError, AttributeError, TypeError):
        m = timm.create_model(arch, pretrained=False, **dict(kw, depth=2))
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape[-1] == dim
    if kw_name == "virchow2":
        assert out.shape[1] == 1 + 4 + 256


# ---------------- TRIDENT 导入 ----------------
def test_import_trident(trident_imported):
    import h5py
    env = trident_imported
    wp = WorkPaths(env.cfg)
    uni = sorted((wp.wsi / "features" / "uni2_h").glob("*.h5"))
    conch = sorted((wp.wsi / "features" / "conch_v15").glob("*.h5"))
    assert uni and conch and len(conch) < len(uni)          # 假数据中 ext1 的切片缺少 conch 特征
    assert not any(p.name.startswith("C2-") for p in conch)
    with h5py.File(uni[0]) as f:
        assert f["features"].shape[1] == 1536
        assert "name" not in f.attrs and "savetodir" not in f.attrs
        assert f.attrs["tile_px"] == 256 and f.attrs["source"] == "trident"
    run = env.latest_run("import_trident")
    rep = (run / "trident_import_report.md").read_text(encoding="utf-8")
    assert "缺少该编码器特征" in rep and "C2-0001" in rep
    assert (run / "private" / "unmatched_trident_files.csv").exists()
    # 重跑：全部跳过
    assert env.run(import_trident.main) == 0
    assert "skipped" in (env.latest_run("import_trident") / "status.csv").read_text(encoding="utf-8-sig")


def test_import_trident_detects_inconsistent_coords(trident_imported, tmp_path):
    import shutil
    import h5py
    env = trident_imported
    src = env.root / "trident"
    dst = tmp_path / "trident_copy"
    shutil.copytree(src, dst)
    f = [p for p in sorted((dst / "20x_256px_0px_overlap" / "features_conch_v15").glob("ZY*.h5"))
         if not p.name.startswith("ZY000000")][0]                 # 选一个清单中有的切片
    with h5py.File(f, "a") as h:
        c = h["coords"][()]
        c[0] += 7
        del h["coords"]
        h.create_dataset("coords", data=c)
    code = env.run(import_trident.main, "--set", f"trident.job_dir={dst}",
                   "--set", f"paths.work_dir={tmp_path / 'work2'}")
    assert code == 0
    rep = (env.latest_run("import_trident") / "trident_import_report.md").read_text(encoding="utf-8")
    assert "切块坐标不一致的病例\n\n无" not in rep
