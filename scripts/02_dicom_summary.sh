#!/usr/bin/env bash
# DICOM 信息汇总。用法：bash scripts/02_dicom_summary.sh [--limit 5] [--root CT根目录]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.summary --config "$CONFIG" "$@"
