"""配置系统：继承、命令行覆盖、路径解析、报错。"""

from pathlib import Path

import pytest
import yaml

from gastric_mm.config import apply_overrides, load_config, resolve_path, root_dir, save_config
from gastric_mm.utils.errors import UserError

REPO = Path(__file__).resolve().parent.parent


def test_default_and_inheritance():
    cfg = load_config(REPO / "configs" / "fake_demo.yaml")
    assert cfg.paths.root == "data/fake_demo"           # 子配置覆盖
    assert cfg.labels.cldn182_threshold == 75           # 继承自 default
    assert cfg.ct.crop.specs[0].name == "tumor_small"   # 列表整体替换
    assert root_dir(cfg) == (REPO / "data" / "fake_demo").resolve()


def test_overrides_parse_yaml_values():
    cfg = load_config(REPO / "configs" / "default.yaml",
                      ["ct.target_spacing=[1.0,1.0,3.0]", "seed=7", "new.key=abc", "labels.ki67_threshold=20"])
    assert cfg.ct.target_spacing == [1.0, 1.0, 3.0]
    assert cfg.seed == 7 and cfg.new.key == "abc" and cfg.labels.ki67_threshold == 20


def test_bad_override_and_missing_file():
    with pytest.raises(UserError):
        apply_overrides({}, ["no_equal_sign"])
    with pytest.raises(UserError) as e:
        load_config("configs/does_not_exist.yaml")
    assert "server_example" in e.value.hint


def test_missing_key_message():
    cfg = load_config(REPO / "configs" / "default.yaml")
    with pytest.raises(AttributeError) as e:
        _ = cfg.not_a_key
    assert "default.yaml" in str(e.value)


def test_yaml_error_is_user_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(UserError):
        load_config(p)


def test_resolve_and_save(tmp_path):
    cfg = load_config(REPO / "configs" / "default.yaml", [f"paths.root={tmp_path}"])
    assert resolve_path(cfg, "a/b.csv") == tmp_path / "a" / "b.csv"
    assert resolve_path(cfg, "/abs/x") == Path("/abs/x")
    assert resolve_path(cfg, "") is None and resolve_path(cfg, None) is None
    save_config(cfg, tmp_path / "c.yaml")
    back = yaml.safe_load((tmp_path / "c.yaml").read_text(encoding="utf-8"))
    assert back["labels"]["pdl1_thresholds"] == [1, 5, 10]
