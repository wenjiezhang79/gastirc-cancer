"""测试公共设置：禁止联网；在临时目录生成一套小型假数据，供各模块的冒烟测试使用。"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------- 禁止联网 ----------------
_orig_connect = socket.socket.connect


def _guarded_connect(self, address):
    if self.family == getattr(socket, "AF_UNIX", None):
        return _orig_connect(self, address)
    host = address[0] if isinstance(address, tuple) else str(address)
    if host in ("127.0.0.1", "localhost", "::1"):
        return _orig_connect(self, address)
    raise RuntimeError(f"测试中禁止联网（尝试连接 {address}）")


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    socket.socket.connect = _guarded_connect
    os.environ["HF_HUB_OFFLINE"] = "1"
    yield
    socket.socket.connect = _orig_connect


# ---------------- 假数据 ----------------
class FakeEnv:
    def __init__(self, base: Path):
        self.base = base
        self.root = base / "data"
        self.out = base / "outputs"
        self.config = base / "test_config.yaml"

    @property
    def cfg(self):
        from gastric_mm.config import load_config
        return load_config(self.config)

    @property
    def canaries(self):
        return json.loads((self.root / "canary.json").read_text(encoding="utf-8"))["canaries"]

    def latest_run(self, step: str) -> Path:
        runs = sorted((self.out / step).iterdir())
        return runs[-1]

    def run(self, main, *args) -> int:
        return main(["--config", str(self.config), *args])


def write_test_config(base: Path, extra: dict = None) -> Path:
    cfg = {
        "_base_": str(REPO / "configs" / "fake_demo.yaml"),
        "paths": {"root": str(base / "data"), "manifest": "manifest.csv", "work_dir": "work",
                  "output_dir": str(base / "outputs")},
        "num_workers": 1,
        "fake_data": {"slide_size_px": [1536, 1024], "ct_shape": [72, 72], "ct_pixel_spacing": 2.2,
                      "ct_n_slices_5mm": 24},
        "trident": {"job_dir": "trident"},
    }
    if extra:
        from gastric_mm.config import deep_merge
        cfg = deep_merge(cfg, extra)
    p = base / "test_config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def fake_env(tmp_path_factory) -> FakeEnv:
    base = tmp_path_factory.mktemp("gmm")
    write_test_config(base)
    env = FakeEnv(base)
    from gastric_mm import fake_data
    assert env.run(fake_data.main) == 0
    return env


# 按流程顺序运行的会话级步骤（后续测试依赖前面的结果）
@pytest.fixture(scope="session")
def ct_converted(fake_env):
    from gastric_mm.ct import convert
    assert fake_env.run(convert.main) == 0
    return fake_env


@pytest.fixture(scope="session")
def ct_cropped(ct_converted):
    from gastric_mm.ct import crop
    assert ct_converted.run(crop.main) == 0
    return ct_converted


@pytest.fixture(scope="session")
def wsi_tiled(fake_env):
    from gastric_mm.wsi import tiling
    assert fake_env.run(tiling.main) == 0
    return fake_env


@pytest.fixture(scope="session")
def features_extracted(wsi_tiled):
    from gastric_mm.wsi import extract_features
    assert wsi_tiled.run(extract_features.main) == 0
    return wsi_tiled


@pytest.fixture(scope="session")
def trident_imported(fake_env):
    from gastric_mm.wsi import import_trident
    assert fake_env.run(import_trident.main) == 0
    return fake_env
