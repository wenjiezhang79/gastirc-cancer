#!/usr/bin/env bash
# ② 人工勾画病例抽样：从内部训练集按 cT 分期分层抽 200–300 例（名单只含研究编号，可交给影像科）。
# 用法：bash scripts/21_seg_sample.sh [--overwrite]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.seg_sample --config "$CONFIG" "$@"
