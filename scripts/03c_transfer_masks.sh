#!/usr/bin/env bash
# 薄层分割转换到分析网格。用法：bash scripts/03c_transfer_masks.sh --src-dir 薄层分割目录 --out-dir 输出目录
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.transfer --config "$CONFIG" "$@"
