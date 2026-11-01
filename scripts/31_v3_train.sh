#!/usr/bin/env bash
# v3 融合模型：交叉验证训练（只用内部训练集）。需先在配置中填 v3.init.path_exp / ct_exp（v1、v2 选定的实验）。
# 用法：EXP=fusion_main GPU=0 bash scripts/31_v3_train.sh [--folds 0,1,2]
#   两张卡并行同一实验：EXP=fusion_main GPU=0 bash scripts/31_v3_train.sh --folds 0,1,2
#                      EXP=fusion_main GPU=1 bash scripts/31_v3_train.sh --folds 3,4
#   单任务融合模型：EXP=st_msi_h SINGLE_TASK=msi_h bash scripts/31_v3_train.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ]; then echo "【需要处理】请用 EXP=实验名 指定实验，如 EXP=fusion_main"; exit 1; fi
A=(--exp "$EXP"); if [ -n "${GPU:-}" ]; then A+=(--gpu "$GPU"); fi
if [ -n "${SINGLE_TASK:-}" ]; then A+=(--single-task "$SINGLE_TASK"); fi
"$PY" -m gastric_mm.fusion.train --config "$CONFIG" "${A[@]}" "$@"
