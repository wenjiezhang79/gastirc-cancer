#!/usr/bin/env bash
# ⑥ 最终评估（模型定型后只运行一次）：EXP=实验名 COHORT=test|ext1|ext2|ext3|external bash scripts/28_v2_predict.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ] || [ -z "${COHORT:-}" ]; then echo "【需要处理】请用 EXP=实验名 COHORT=队列 指定"; exit 1; fi
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.ct_mtl.predict --config "$CONFIG" --exp "$EXP" --cohort "$COHORT" --final "${G[@]}" "$@"
