#!/usr/bin/env bash
# v3 最终评估（模型定型后只运行一次）：EXP=实验名 COHORT=test|ext1|ext2|ext3|external bash scripts/35_v3_predict.sh
#   融合模型与后期融合实验（late_mean / late_stack）都用本脚本；可一起评估最终模型与各对照模型。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ] || [ -z "${COHORT:-}" ]; then echo "【需要处理】请用 EXP=实验名 COHORT=队列 指定"; exit 1; fi
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.fusion.predict --config "$CONFIG" --exp "$EXP" --cohort "$COHORT" --final "${G[@]}" "$@"
