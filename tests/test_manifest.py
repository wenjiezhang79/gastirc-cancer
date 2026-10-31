"""清单读取与校验。"""

import pandas as pd
import pytest

from gastric_mm import manifest as M
from gastric_mm.utils.errors import UserError


def _raw(fake_env):
    return M.read_manifest_csv(fake_env.root / "manifest.csv")


def _codes(res):
    return {i.code for i in res.issues}


def test_fake_manifest_passes_and_cli(fake_env):
    res = M.validate_manifest(_raw(fake_env), fake_env.cfg)
    assert res.n_errors == 0, [i.message for i in res.issues if i.level == "error"]
    assert "her2_2plus_no_fish" in _codes(res)          # 假数据中故意放了 HER2 2+ 无 FISH
    assert res.summary["by_center"]["internal"]["n"] == 6
    # 没有切片 / CT 的病例被正确统计
    assert res.summary["by_center"]["internal"]["slide"] < 6
    assert fake_env.run(M.main) == 0
    run = fake_env.latest_run("manifest")
    report = (run / "validation_report.md").read_text(encoding="utf-8")
    assert "通过" in report and "msi_h" in report
    assert (run / "labels_preview.csv").exists() and (run / "validation_issues.csv").exists()


def test_detects_errors(fake_env):
    raw = _raw(fake_env)
    bad = raw.copy()
    bad.loc[1, "study_id"] = bad.loc[0, "study_id"]            # 重复
    bad.loc[2, "study_id"] = "ZY123456"                        # 格式错误
    bad.loc[3, "center"] = "hospitalX"
    bad.loc[4, "lauren"] = "Diffuse"                           # 大小写可接受
    bad.loc[5, "her2_ihc"] = "2"                               # 应写 2+
    bad.loc[6, "cldn182_pct"] = "120"
    bad.loc[7, "ct_point_x"] = "10"
    bad.loc[7, "ct_point_y"] = ""
    bad.loc[7, "ct_point_z"] = ""
    bad.loc[8, "os_event"] = "2"
    bad["患者姓名"] = "x"
    res = M.validate_manifest(bad, fake_env.cfg)
    codes = _codes(res)
    for c in ("dup_id", "bad_id", "bad_center", "bad_value_her2_ihc", "above_max_cldn182_pct",
              "partial_point", "not_binary_os_event", "phi_column"):
        assert c in codes, c
    assert "bad_value_lauren" not in codes
    assert res.df.loc[4, "lauren"] == "diffuse"
    assert any("1+ / 2+ / 3+" in i.message for i in res.issues)


def test_missing_required_column(fake_env):
    raw = _raw(fake_env).drop(columns=["center"])
    res = M.validate_manifest(raw, fake_env.cfg)
    assert "missing_required_column" in _codes(res)


def test_path_checks_warning_vs_strict(fake_env):
    raw = _raw(fake_env)
    raw.loc[0, "slide_path"] = "raw/slides/not_here.svs"
    res = M.validate_manifest(raw, fake_env.cfg)
    iss = [i for i in res.issues if i.code == "missing_path_slide_path"]
    assert iss and iss[0].level == "warning"
    # 报错信息中不出现路径
    assert "not_here" not in iss[0].message
    cfg = fake_env.cfg
    cfg.manifest["strict_paths"] = True
    res2 = M.validate_manifest(raw, cfg)
    assert any(i.code == "missing_path_slide_path" and i.level == "error" for i in res2.issues)


def test_gbk_csv_and_load_manifest_error(fake_env, tmp_path):
    raw = _raw(fake_env)
    p = tmp_path / "gbk.csv"
    raw.to_csv(p, index=False, encoding="gb18030")
    back = M.read_manifest_csv(p)
    assert list(back.study_id) == list(raw.study_id)
    raw.loc[1, "study_id"] = raw.loc[0, "study_id"]
    p2 = tmp_path / "dup.csv"
    raw.to_csv(p2, index=False)
    cfg = fake_env.cfg
    cfg.paths["manifest"] = str(p2)
    with pytest.raises(UserError) as e:
        M.load_manifest(cfg)
    assert "01_validate_manifest" in e.value.hint


def test_cli_returns_1_on_errors(fake_env, tmp_path):
    raw = _raw(fake_env)
    raw.loc[0, "center"] = "bad"
    p = tmp_path / "bad.csv"
    raw.to_csv(p, index=False)
    code = M.main(["--config", str(fake_env.config), "--set", f"paths.manifest={p}"])
    assert code == 1


def test_to_cases_types(fake_env):
    res = M.validate_manifest(_raw(fake_env), fake_env.cfg)
    cases = M.to_cases(res.df)
    c = [x for x in cases if x["ct_point_x"] is not None][0]
    assert isinstance(c["ct_point_x"], int)
    sel = M.select_cases(cases, ["C1-0001"], None)
    assert len(sel) == 1
    with pytest.raises(UserError):
        M.select_cases(cases, ["C9-9999"], None)
