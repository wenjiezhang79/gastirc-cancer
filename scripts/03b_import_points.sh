#!/usr/bin/env bash
# 导入医生标注的肿瘤中心点（Slicer/ITK-SNAP）。用法：bash scripts/03b_import_points.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.import_points --config "$CONFIG" "$@"
