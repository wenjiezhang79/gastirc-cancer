#!/usr/bin/env bash
# 用假数据把 v0 全流程跑一遍（CPU 即可），用于在服务器上验证环境。
# 假数据写在 data/fake_demo/（不含任何真实患者信息）。用法：bash scripts/run_v0_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CONFIG=configs/fake_demo.yaml
PY="${PYTHON:-python}"
step() { echo; echo "========== $* =========="; }
step "生成假数据";        "$PY" -m gastric_mm.fake_data --config "$CONFIG" --overwrite
step "清单校验";          "$PY" -m gastric_mm.manifest --config "$CONFIG"
step "数据划分";          "$PY" -m gastric_mm.split --config "$CONFIG" --overwrite
step "DICOM 汇总";        "$PY" -m gastric_mm.ct.summary --config "$CONFIG"
step "DICOM 转 NIfTI";    "$PY" -m gastric_mm.ct.convert --config "$CONFIG"
step "导入中心点";        "$PY" -m gastric_mm.ct.import_points --config "$CONFIG"
step "薄层分割转换";      "$PY" -m gastric_mm.ct.transfer --config "$CONFIG" --src-dir data/fake_demo/raw/masks_thin --out-dir data/fake_demo/masks_from_thin
step "CT 裁剪与质控";     "$PY" -m gastric_mm.ct.crop --config "$CONFIG"
step "切片切块";          "$PY" -m gastric_mm.wsi.tiling --config "$CONFIG"
step "导入 TRIDENT 特征"; "$PY" -m gastric_mm.wsi.import_trident --config "$CONFIG"
step "特征提取（随机小网络）"; "$PY" -m gastric_mm.wsi.extract_features --config "$CONFIG"
step "环境检查";          "$PY" -m gastric_mm.env_check --config "$CONFIG"
echo; echo "全部完成。输出在 data/fake_demo/outputs/，处理结果在 data/fake_demo/work/。"
