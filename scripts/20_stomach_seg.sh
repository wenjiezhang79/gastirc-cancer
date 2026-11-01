#!/usr/bin/env bash
# ① 胃分割：调用 TotalSegmentator（独立环境，命令写在配置 v2.external.totalseg.command）。
# 用法：GPU=0 bash scripts/20_stomach_seg.sh [--limit 5] [--study-ids C1-0001]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.ct.stomach_seg --config "$CONFIG" "${G[@]}" "$@"
