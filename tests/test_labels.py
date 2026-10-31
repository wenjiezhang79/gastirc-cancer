"""标签派生规则（逐条）。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gastric_mm.config import load_config
from gastric_mm.labels import derive_labels, label_summary, task_definitions

REPO = Path(__file__).resolve().parent.parent


def _cfg(*over):
    return load_config(REPO / "configs" / "default.yaml", list(over))


def _df(rows):
    return pd.DataFrame(rows)


@pytest.mark.parametrize("ihc,fish,pos,three", [
    ("3+", None, 1, 2), ("2+", "pos", 1, 2), ("2+", "neg", 0, 1), ("2+", None, np.nan, np.nan),
    ("1+", None, 0, 1), ("0", None, 0, 0), (None, "pos", 1, 2), (None, "neg", np.nan, np.nan), (None, None, np.nan, np.nan),
])
def test_her2(ihc, fish, pos, three):
    lab = derive_labels(_df([{"study_id": "C1-0001", "center": "internal", "her2_ihc": ihc, "her2_fish": fish}]), _cfg())
    assert (np.isnan(lab.her2_pos[0]) and np.isnan(pos)) or lab.her2_pos[0] == pos
    assert (np.isnan(lab.her2_3cls[0]) and np.isnan(three)) or lab.her2_3cls[0] == three


def test_her2_2plus_without_fish_as_negative():
    lab = derive_labels(_df([{"study_id": "C1-0001", "center": "internal", "her2_ihc": "2+"}]),
                        _cfg("labels.her2_2plus_without_fish=negative"))
    assert lab.her2_pos[0] == 0


def test_lauren_msi_ebv_pdl1_cldn_ki67():
    rows = [
        {"study_id": "C1-0001", "center": "internal", "lauren": "intestinal", "msi": "MSI-H", "ebv": "neg",
         "pdl1_cps": 5.0, "cldn182_pct": 75.0, "ki67_pct": 40.0},
        {"study_id": "C1-0002", "center": "internal", "lauren": "mixed", "msi": "MSS", "ebv": "pos",
         "pdl1_cps": 0.5, "cldn182_pct": 74.0, "ki67_pct": None},
    ]
    lab = derive_labels(_df(rows), _cfg())
    assert list(lab.lauren) == [0, 2] and list(lab.lauren_intestinal) == [1, 0]
    assert list(lab.msi_h) == [1, 0] and list(lab.ebv_pos) == [0, 1]
    assert list(lab.pdl1_cps1) == [1, 0] and list(lab.pdl1_cps5) == [1, 0] and list(lab.pdl1_cps10) == [0, 0]
    assert list(lab.cldn182_pos) == [1, 0]
    assert lab.ki67_frac[0] == pytest.approx(0.4) and np.isnan(lab.ki67_frac[1])
    assert "ki67_high" not in lab.columns
    lab2 = derive_labels(_df(rows), _cfg("labels.ki67_threshold=30"))
    assert lab2.ki67_high[0] == 1 and np.isnan(lab2.ki67_high[1])


def test_post_neoadj_masking_and_per_task_source():
    rows = [
        {"study_id": "C1-0001", "center": "internal", "lauren": "diffuse", "msi": "MSS", "label_source": "post_neoadj",
         "os_months": 10.0, "os_event": 1.0},
        # 单项来源列优先：HER2 来自活检，不屏蔽
        {"study_id": "C1-0002", "center": "internal", "lauren": "diffuse", "her2_ihc": "3+",
         "label_source": "post_neoadj", "her2_source": "biopsy"},
    ]
    lab = derive_labels(_df(rows), _cfg())
    assert np.isnan(lab.lauren[0]) and np.isnan(lab.msi_h[0])
    assert lab.os_time[0] == 10 and lab.os_event[0] == 1     # 生存不受影响
    assert np.isnan(lab.lauren[1]) and lab.her2_pos[1] == 1
    lab2 = derive_labels(_df(rows), _cfg("labels.use_post_neoadj=true"))
    assert lab2.lauren[0] == 1


def test_survival_partial_is_missing():
    lab = derive_labels(_df([{"study_id": "C1-0001", "center": "internal", "os_months": 12.0, "os_event": None}]), _cfg())
    assert np.isnan(lab.os_time[0]) and np.isnan(lab.os_event[0])


def test_summary_and_tasks():
    cfg = _cfg()
    names = [t.name for t in task_definitions(cfg)]
    assert "pdl1_cps5" in names and "os" in names
    main = {t.name for t in task_definitions(cfg) if t.main}
    assert {"lauren", "her2_pos", "msi_h", "ebv_pos", "pdl1_cps5", "cldn182_pos", "os"} <= main
    lab = derive_labels(_df([{"study_id": "C1-0001", "center": "internal", "msi": "MSI-H"}]), cfg)
    rows = label_summary(lab, cfg)
    r = [x for x in rows if x["task"] == "msi_h" and x["center"] == "internal"][0]
    assert r["n_labeled"] == 1 and r["n_positive"] == 1
