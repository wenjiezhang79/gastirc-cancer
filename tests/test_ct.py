"""CT：DICOM 汇总、转换、重采样、中心点导入、分割转换、裁剪与质控。"""

import json

import numpy as np
import pytest

from gastric_mm.ct import convert, crop, import_points, summary, transfer
from gastric_mm.ct import transforms as T
from gastric_mm.ct.dicom_io import SeriesInfo, guess_phase, read_series_image, select_series
from gastric_mm.paths import WorkPaths
from gastric_mm.utils.errors import CaseError


def sitk():
    import SimpleITK
    return SimpleITK


# ---------------- 期相与序列选择（单元） ----------------
def test_guess_phase_keywords_and_delay(fake_env):
    cfg = fake_env.cfg
    kw, dl = cfg.ct.phase_keywords, cfg.ct.phase_delay_s

    def ph(desc, **kv):
        s = SeriesInfo(uid="x", description=desc, **kv)
        return guess_phase(s, kw, dl)[0]

    assert ph("Portal Venous 5mm") == "venous"
    assert ph("静脉期 1.25") == "venous"
    assert ph("Arterial 5mm") == "arterial"
    assert ph("PV 5.0") == "venous"
    assert ph("Upper part abdomen") == "unknown"     # "part" 不能误判为 art
    assert ph("Delayed") == "delayed"
    assert ph("abdomen", acq_time_s=1070.0, bolus_time_s=1000.0) == "venous"
    assert ph("abdomen", acq_time_s=1030.0, bolus_time_s=1000.0) == "arterial"


def test_select_series_rules():
    def s(n, phase, th, k="B30f", nimg=30, it=("ORIGINAL", "PRIMARY", "AXIAL")):
        x = SeriesInfo(uid=str(n), series_number=n, phase=phase, slice_thickness=th, kernel=k, image_type=it,
                       modality="CT")
        x.files = [None] * nimg
        return x
    ser = [s(1, "unknown", 5, it=("LOCALIZER",), nimg=1), s(2, "arterial", 5), s(3, "venous", 1.25),
           s(4, "venous", 5), s(5, "venous", 5, k="B70f")]
    best, why = select_series(ser, "venous", 5.0, 10)
    assert best.series_number == 4
    assert select_series(ser, "venous", 5.0, 10, forced_number=3)[0].series_number == 3
    with pytest.raises(CaseError):
        select_series([s(2, "arterial", 5)], "venous", 5.0, 10)
    best, why = select_series([s(6, "unknown", 5)], "venous", 5.0, 10)
    assert "人工核对" in why


def test_nonuniform_spacing_rejected():
    x = SeriesInfo(uid="x", series_number=1, uniform=False, spacing_min=5, spacing_max=10, n_gaps=1)
    with pytest.raises(CaseError):
        read_series_image(x)


# ---------------- 变换（单元） ----------------
def test_transforms_basic():
    S = sitk()
    img = S.Image([20, 20, 40], S.sitkFloat32)
    img.SetSpacing([1.0, 1.0, 1.25])
    img.SetOrigin([-10.0, -10.0, 0.0])
    arr = np.zeros((40, 20, 20), np.float32)
    arr[::2] = 100.0
    im2 = S.GetImageFromArray(arr)
    im2.CopyInformation(img)
    avg = T.box_average_z(im2, 5.0)
    a = S.GetArrayFromImage(avg)
    assert abs(a[10:30].mean() - 50) < 5 and a[10:30].std() < 15     # 平均后条纹被抹平
    p = T.index_to_physical(img, [3, 4, 5])
    assert np.allclose(T.physical_to_index(img, p), [3, 4, 5])
    ref = T.roi_reference([0, 0, 25], [2, 2, 5], [8, 8, 4])
    assert np.allclose(T.index_to_physical(ref, [4, 4, 2]), [0, 0, 25])
    assert T.outside_fraction(img, ref) == pytest.approx(0.0)
    ref2 = T.roi_reference([-10, -10, 0], [1, 1, 1.25], [10, 10, 10])
    assert T.outside_fraction(img, ref2) > 0.5
    w = T.window_normalize(np.array([-1000, 40, 1000]), 40, 400)
    assert np.allclose(w, [0, 0.5, 1])
    z = T.window_normalize(np.random.RandomState(0).normal(0, 50, 1000), 40, 400, "zscore")
    assert abs(z.mean()) < 1e-5


# ---------------- DICOM 汇总 ----------------
def test_dicom_summary(fake_env):
    assert fake_env.run(summary.main) == 0
    run = fake_env.latest_run("dicom_summary")
    import pandas as pd
    ser = pd.read_csv(run / "dicom_series.csv")
    sel = ser[ser.selected == "Y"]
    assert (sel.phase == "venous").all() and (sel.slice_thickness_mm == 5).all()
    assert (ser.thin_for_annotation == "Y").sum() == 2
    assert not ser[ser.localizer == "Y"].selected.eq("Y").any()
    text = (run / "dicom_summary.md").read_text(encoding="utf-8")
    assert "推荐的重采样体素" in text
    assert (run / "dicom_summary.png").exists()


def test_dicom_summary_root_mode(fake_env):
    assert fake_env.run(summary.main, "--root", str(fake_env.root / "raw" / "ct"), "--limit", "2") == 0
    run = fake_env.latest_run("dicom_summary")
    assert (run / "private" / "scan_id_map.csv").exists()
    assert "scan-0001" in (run / "dicom_series.csv").read_text(encoding="utf-8-sig")


# ---------------- 转换 ----------------
def test_convert_outputs(ct_converted):
    S = sitk()
    env = ct_converted
    wp = WorkPaths(env.cfg)
    res = __import__("gastric_mm.manifest", fromlist=["x"]).load_manifest(env.cfg)
    cases = [c for c in __import__("gastric_mm.manifest", fromlist=["x"]).to_cases(res.df) if c["ct_dir"]]
    for c in cases:
        sid = c["study_id"]
        img = S.ReadImage(str(wp.ct_native(sid)))
        assert img.GetPixelID() == S.sitkInt16
        assert np.allclose(img.GetSpacing(), [2.2, 2.2, 5.0])
        meta = json.loads(wp.ct_native_meta(sid).read_text(encoding="utf-8"))
        assert meta["series_number"] == 3 and meta["phase"] == "venous"
        rs = S.ReadImage(str(wp.ct_resampled(sid)))
        assert np.allclose(rs.GetSpacing(), env.cfg.ct.target_spacing)
        # 肿瘤中心点处的 HU 应接近静脉期肿瘤值（90）
        if c["ct_point_x"] is not None:
            a = S.GetArrayFromImage(img)
            v = a[c["ct_point_z"], c["ct_point_y"], c["ct_point_x"]]
            assert 50 < v < 130, v
    assert wp.ct_native_thin("C1-0001").exists()
    # 再跑一次全部跳过
    assert env.run(convert.main) == 0
    st = (env.latest_run("ct_convert") / "status.csv").read_text(encoding="utf-8-sig")
    assert "skipped" in st and ",ok," not in st


def test_resampled_mask_matches_native_mask(ct_converted):
    S = sitk()
    wp = WorkPaths(ct_converted.cfg)
    m = S.ReadImage(str(wp.ct_resampled_mask("C1-0001")))
    raw = sorted((ct_converted.root / "raw" / "masks").glob("*.nii.gz"))
    assert raw
    assert np.allclose(m.GetSpacing(), ct_converted.cfg.ct.target_spacing)
    assert S.GetArrayFromImage(m).sum() > 0


# ---------------- 中心点导入与分割转换 ----------------
def test_import_points(ct_converted):
    env = ct_converted
    before = (env.root / "manifest.csv").read_bytes()
    assert env.run(import_points.main) == 0
    run = env.latest_run("import_points")
    import pandas as pd
    rep = pd.read_csv(run / "points_report.csv")
    ok = rep[rep.status == "ok"]
    assert set(ok.source) == {"slicer_json", "label"}
    # 与生成器写入的真值（四舍五入后的体素坐标）相差不超过 1 个体素对应的距离
    assert (ok.dist_to_previous_mm.dropna() <= 2.2 * 1.5 + 0.1).all()
    newm = pd.read_csv(run / "private" / "manifest_with_points.csv", dtype=str, keep_default_na=False)
    row = newm[newm.study_id == "C1-0003"].iloc[0]
    assert row.ct_point_x != ""
    assert (env.root / "manifest.csv").read_bytes() == before        # 原清单不被修改


def test_fcsv_reader(tmp_path):
    p = tmp_path / "C1-0001.fcsv"
    p.write_text("# Markups fiducial file version = 4.11\n# CoordinateSystem = RAS\n"
                 "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
                 "vtkMRMLMarkupsFiducialNode_0,10,20,30,0,0,0,1,1,1,0,F-1,,\n", encoding="utf-8")
    assert np.allclose(import_points.read_fcsv(p), [-10, -20, 30])


def test_transfer_thin_masks(ct_converted, tmp_path):
    S = sitk()
    env = ct_converted
    out = tmp_path / "from_thin"
    assert env.run(transfer.main, "--src-dir", str(env.root / "raw" / "masks_thin"), "--out-dir", str(out)) == 0
    wp = WorkPaths(env.cfg)
    m = S.ReadImage(str(out / "C1-0001_tumor.nii.gz"))
    native = S.ReadImage(str(wp.ct_native("C1-0001")))
    assert m.GetSize() == native.GetSize()
    # 与直接在 5 mm 上生成的分割比较 Dice
    mp = [p for p in (env.root / "raw" / "masks").glob("*.nii.gz")]
    import pandas as pd
    man = pd.read_csv(env.root / "manifest.csv", dtype=str, keep_default_na=False)
    ref_path = env.root / man[man.study_id == "C1-0001"].iloc[0].tumor_mask_path
    a = S.GetArrayFromImage(m) > 0
    b = S.GetArrayFromImage(S.ReadImage(str(ref_path))) > 0
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    assert dice > 0.8, dice
    # 采集信息不一致时拒绝转换
    meta = wp.ct_native_thin_meta("C1-0006")
    d = json.loads(meta.read_text(encoding="utf-8"))
    d["acq_time_s"] = d["acq_time_s"] - 40
    meta.write_text(json.dumps(d), encoding="utf-8")
    code = env.run(transfer.main, "--src-dir", str(env.root / "raw" / "masks_thin"), "--out-dir", str(tmp_path / "x"))
    assert code == 3
    st = (env.latest_run("ct_transfer") / "status.csv").read_text(encoding="utf-8-sig")
    assert "采集时间" in st


# ---------------- 裁剪 ----------------
def test_crop_outputs_and_centering(ct_cropped):
    S = sitk()
    env = ct_cropped
    cfg = env.cfg
    wp = WorkPaths(cfg)
    spec = cfg.ct.crop.specs[0]
    img = S.ReadImage(str(wp.ct_crop(spec.name, "C1-0001", "img")))
    assert img.GetSize() == tuple(spec.size_vox)
    a = S.GetArrayFromImage(img)
    assert a.min() >= 0 and a.max() <= 1
    cz, cy, cx = [s // 2 for s in a.shape]
    # 中心为肿瘤（静脉期约 90 HU → 窗 [-160,240] 下约 0.62）
    assert 0.5 < a[cz, cy, cx] < 0.75, a[cz, cy, cx]
    m = S.GetArrayFromImage(S.ReadImage(str(wp.ct_crop(spec.name, "C1-0001", "mask"))))
    c = np.argwhere(m > 0).mean(axis=0)
    assert np.all(np.abs(c - np.array([cz, cy, cx])) <= np.array([1.0, 2.0, 2.0])), c
    assert wp.ct_qc(spec.name, "C1-0001").exists()
    run = env.latest_run("ct_crop")
    assert list(run.glob("overview_*_p01.png"))
    import pandas as pd
    st = pd.read_csv(run / "status.csv")
    assert (st[st.study_id == "C1-0001"].point_to_stomach_mm <= 1).all()
    # 只有分割、没有中心点的病例用分割质心
    r6 = st[st.study_id == "C1-0006"].iloc[0]
    assert r6[f"{spec.name}_center"] == "mask"


def test_crop_point_outside_fails(ct_converted, tmp_path):
    import pandas as pd
    env = ct_converted
    man = pd.read_csv(env.root / "manifest.csv", dtype=str, keep_default_na=False)
    i = man.index[man.study_id == "C1-0001"][0]
    man.loc[i, "ct_point_x"] = "999"
    p = tmp_path / "m.csv"
    man.to_csv(p, index=False)
    code = env.run(crop.main, "--set", f"paths.manifest={p}", "--study-ids", "C1-0001", "--overwrite")
    assert code == 3
    st = (env.latest_run("ct_crop") / "status.csv").read_text(encoding="utf-8-sig")
    assert "不在图像范围内" in st
