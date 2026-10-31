#!/usr/bin/env bash
# 切片组织检测 + 20× 切块 + 质控图。用法：bash scripts/05_wsi_tiling.sh [--limit 5]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.wsi.tiling --config "$CONFIG" "$@"
