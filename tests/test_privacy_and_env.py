"""隐私检查（金丝雀）与环境检查。

假数据的文件夹名、DICOM 头、TRIDENT 属性中放入了假住院号和假姓名。完整流程跑完后，
除 private/ 目录外，所有输出、日志、处理结果（含文件名和 HDF5 属性）中都不得出现它们。
"""

from pathlib import Path

from gastric_mm import env_check, manifest, split
from gastric_mm.ct import import_points, summary

TEXT_EXT = {".md", ".csv", ".log", ".json", ".yaml", ".txt"}


def _scan(paths, canaries):
    import h5py
    hits = []
    for base in paths:
        for p in Path(base).rglob("*"):
            if "private" in p.parts:
                continue
            for c in canaries:
                if c in p.name:
                    hits.append(f"文件名 {p}")
            if p.is_file() and p.suffix in TEXT_EXT:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for c in canaries:
                    if c in text:
                        hits.append(f"{p}: {c}")
            if p.is_file() and p.suffix == ".h5":
                with h5py.File(p, "r") as f:
                    vals = [str(v) for v in f.attrs.values()]
                    for k in f.keys():
                        vals += [str(v) for v in f[k].attrs.values()]
                for c in canaries:
                    if any(c in v for v in vals):
                        hits.append(f"{p} 属性: {c}")
    return hits


def test_no_canary_leak(fake_env, ct_cropped, features_extracted, trident_imported):
    env = fake_env
    # 再跑几个会写日志/报告的模块
    assert env.run(manifest.main) == 0
    assert env.run(summary.main) == 0
    env.run(import_points.main)
    env.run(split.main, "--overwrite")
    canaries = [c for c in env.canaries if len(c) >= 2]
    assert any(c.startswith("ZY") for c in canaries)
    hits = _scan([env.out, env.root / "work"], canaries)
    assert not hits, "\n".join(hits[:20])
    # 日志中也不出现原始数据目录名
    for log in env.out.rglob("run.log"):
        t = log.read_text(encoding="utf-8")
        assert "raw/ct/" not in t and "raw/slides/" not in t, log


def test_scrub_paths():
    from gastric_mm.utils.privacy import scrub
    msg = "cannot open /data/hospital/ZY123456_张三/IM0001 and C:\\pacs\\ZY123456\\x.dcm"
    out = scrub(msg, ["/data/hospital/ZY123456_张三"])
    assert "ZY123456" not in out and "张三" not in out
    tb = 'File "/home/u/repo/gastric_mm/ct/convert.py", line 3'
    assert scrub(tb) == tb


def test_env_check(fake_env):
    assert fake_env.run(env_check.main) == 0
    run = fake_env.latest_run("env_check")
    text = (run / "env_report.md").read_text(encoding="utf-8")
    assert "PyTorch" in text and "pydicom" in text
    assert (run / "packages_all.txt").exists()
