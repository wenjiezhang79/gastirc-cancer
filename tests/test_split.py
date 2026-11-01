"""数据划分：外部中心不进入训练；分层；可复现；不覆盖已有划分。"""

from pathlib import Path

import numpy as np
import pandas as pd

from gastric_mm import split as S
from gastric_mm.config import load_config

REPO = Path(__file__).resolve().parent.parent


def _labels(n_int=300, n_ext=60, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_int + n_ext):
        center = "internal" if i < n_int else "ext1"
        rows.append({"study_id": f"C1-{i:04d}", "center": center,
                     "lauren": float(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])),
                     "msi_h": float(rng.rand() < 0.1), "ebv_pos": float(rng.rand() < 0.08),
                     "her2_pos": np.nan if rng.rand() < 0.2 else float(rng.rand() < 0.12), "os_event": 0.0})
    return pd.DataFrame(rows)


def test_make_split_properties():
    cfg = load_config(REPO / "configs" / "default.yaml")
    lab = _labels()
    sp = S.make_split(lab, cfg)
    ext = sp[sp.center != "internal"]
    assert (ext.set == "external").all() and (ext.fold == -1).all()
    inter = sp[sp.center == "internal"]
    assert set(inter.set) == {"train", "test"}
    assert abs((inter.set == "test").mean() - 0.2) < 0.03
    tr = sp[sp.set == "train"]
    assert set(tr.fold) == set(range(5))
    sizes = tr.fold.value_counts()
    assert sizes.max() - sizes.min() <= 3
    # MSI 阳性在各折中分布均衡
    m = tr.merge(lab[["study_id", "msi_h"]], on="study_id")
    pos = m[m.msi_h == 1].fold.value_counts().reindex(range(5), fill_value=0)
    assert pos.max() - pos.min() <= 2
    # 可复现
    sp2 = S.make_split(lab, cfg)
    assert sp.equals(sp2)


def test_split_cli_and_no_overwrite(fake_env):
    p = fake_env.root / "work" / "splits" / "split_seed2026.csv"
    if p.exists():
        p.unlink()
    assert fake_env.run(S.main) == 0
    df = pd.read_csv(p)
    assert set(df[df.center != "internal"].set) == {"external"}
    assert fake_env.run(S.main) == 1                   # 已存在，拒绝覆盖
    assert fake_env.run(S.main, "--overwrite") == 0
    assert p.with_suffix(".csv.bak").exists()
    assert "数据划分报告" in (fake_env.latest_run("split") / "split_report.md").read_text(encoding="utf-8")
