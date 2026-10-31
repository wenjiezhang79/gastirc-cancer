"""假数据生成器：切片、DICOM、分割都能被正常读取；不会覆盖非假数据目录。"""

import pytest

from gastric_mm import fake_data
from gastric_mm.config import load_config
from gastric_mm.utils.errors import UserError
from gastric_mm.wsi.reader import SlideReader, openslide_available


def test_slides_readable_by_both_backends(fake_env):
    slides = sorted((fake_env.root / "raw" / "slides").glob("*.svs"))
    assert slides
    backends = ["tifffile"] + (["openslide"] if openslide_available() else [])
    mpps = set()
    for s in slides:
        for b in backends:
            with SlideReader(s, b) as r:
                assert r.level_count >= 2
                mpps.add(round(r.base_mpp, 3))
                a = r.read_region(0, 0, 0, 64, 64)
                assert a.shape == (64, 64, 3)
    assert mpps == {0.5, 0.25}            # 20× 与 40× 各有


def test_backends_agree(fake_env):
    if not openslide_available():
        pytest.skip("未安装 openslide")
    s = sorted((fake_env.root / "raw" / "slides").glob("*.svs"))[0]
    with SlideReader(s, "openslide") as a, SlideReader(s, "tifffile") as b:
        ra = a.read_region(256, 256, 0, 128, 128)
        rb = b.read_region(256, 256, 0, 128, 128)
    assert (abs(ra.astype(int) - rb.astype(int)) <= 1).all()


def test_dicom_series(fake_env):
    from gastric_mm.ct.dicom_io import scan_dicom_dir
    cfg = fake_env.cfg
    d = sorted((fake_env.root / "raw" / "ct").iterdir())[0]
    series = scan_dicom_dir(d, cfg.ct.phase_keywords, cfg.ct.phase_delay_s)
    phases = {s.series_number: s.phase for s in series}
    assert phases[3] == "venous"
    assert any(s.is_localizer for s in series)
    s3 = [s for s in series if s.series_number == 3][0]
    assert s3.uniform and s3.spacing_median == pytest.approx(5.0)


def test_refuses_non_fake_dir(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "real_patient_data.txt").write_text("x")
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "fake_demo.yaml",
                      [f"paths.root={tmp_path / 'data'}", f"paths.output_dir={tmp_path / 'out'}"])
    with pytest.raises(UserError):
        fake_data.generate(cfg)
