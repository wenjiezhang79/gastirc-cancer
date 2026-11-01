#!/usr/bin/env bash
# DICOM 转 NIfTI + 整卷重采样。用法：bash scripts/03_ct_convert.sh [--limit 5] [--overwrite]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.convert --config "$CONFIG" "$@"
