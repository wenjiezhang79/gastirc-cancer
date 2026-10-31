#!/usr/bin/env bash
# 病理特征提取。用法：GPU=0 bash scripts/06_wsi_features.sh [--extractor uni2_h] [--limit 5]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -n "${GPU:-}" ]; then set -- --gpu "$GPU" "$@"; fi
"$PY" -m gastric_mm.wsi.extract_features --config "$CONFIG" "$@"
