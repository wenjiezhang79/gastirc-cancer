#!/usr/bin/env bash
# ④ CT 特征提取（冻结编码器）。用法：ENCODER=radimagenet_r50 GPU=0 bash scripts/24_ct_features.sh [--limit 5]
# 新编码器第一次使用前，请先用 --limit 3 与官方示例代码的输出核对（见 docs/v2_offline_setup.md）。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
A=(); if [ -n "${ENCODER:-}" ]; then A+=(--encoder "$ENCODER"); fi
if [ -n "${GPU:-}" ]; then A+=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.ct.extract_features --config "$CONFIG" "${A[@]}" "$@"
