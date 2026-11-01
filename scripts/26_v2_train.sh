#!/usr/bin/env bash
# ⑥ CT 单模态多任务模型：交叉验证训练（只用内部训练集）。
# 用法：EXP=ct_radimagenet ENCODER=radimagenet_r50 GPU=0 bash scripts/26_v2_train.sh [--folds 0,1,2]
#   两张卡并行同一实验：... GPU=0 bash scripts/26_v2_train.sh --folds 0,1,2；... GPU=1 bash scripts/26_v2_train.sh --folds 3,4
#   影像组学基线：EXP=radiomics_bw25 RADIOMICS=1 bash scripts/26_v2_train.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ]; then echo "【需要处理】请用 EXP=实验名 指定实验，如 EXP=ct_radimagenet"; exit 1; fi
if [ -n "${RADIOMICS:-}" ]; then
  "$PY" -m gastric_mm.ct_mtl.radiomics_model --config "$CONFIG" --exp "$EXP" "$@"; exit $?
fi
A=(--exp "$EXP"); if [ -n "${ENCODER:-}" ]; then A+=(--encoder "$ENCODER"); fi
if [ -n "${GPU:-}" ]; then A+=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.ct_mtl.train --config "$CONFIG" "${A[@]}" "$@"
