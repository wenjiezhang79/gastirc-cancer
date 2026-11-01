#!/usr/bin/env bash
# 导入 TRIDENT 已提取的特征。用法：bash scripts/05b_import_trident.sh [--mag-dir 20x_256px_0px_overlap]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.wsi.import_trident --config "$CONFIG" "$@"
