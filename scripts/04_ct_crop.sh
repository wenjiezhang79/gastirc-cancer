#!/usr/bin/env bash
# CT 三维裁剪 + 质控图。用法：bash scripts/04_ct_crop.sh [--limit 5] [--specs tumor_96mm]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.crop --config "$CONFIG" "$@"
