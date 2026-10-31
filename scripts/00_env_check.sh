#!/usr/bin/env bash
# 环境检查（不联网、不读患者数据）。用法：bash scripts/00_env_check.sh [--hash] [--gpu-test]
# 还没有 configs/server.yaml 时自动使用 configs/default.yaml。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "${CONFIG:-}" ]; then
  if [ -f configs/server.yaml ]; then CONFIG=configs/server.yaml; else CONFIG=configs/default.yaml; fi
fi
export CONFIG
source scripts/_common.sh
"$PY" -m gastric_mm.env_check --config "$CONFIG" "$@"
