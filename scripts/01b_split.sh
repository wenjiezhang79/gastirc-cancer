#!/usr/bin/env bash
# 数据划分（内部中心 8:2 + 训练集 5 折）。用法：bash scripts/01b_split.sh [--overwrite]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.split --config "$CONFIG" "$@"
