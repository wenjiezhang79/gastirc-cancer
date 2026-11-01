#!/usr/bin/env bash
# v1 病理多任务模型：交叉验证训练（只用内部训练集）。
# 用法：EXP=uni2h_abmil GPU=0 bash scripts/10_v1_train.sh [--folds 0,1,2] [--encoders uni2_h] [--set 键=值]
#   两张卡并行同一实验：EXP=x GPU=0 bash scripts/10_v1_train.sh --folds 0,1,2
#                       EXP=x GPU=1 bash scripts/10_v1_train.sh --folds 3,4
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ]; then echo "【需要处理】请用 EXP=实验名 指定实验，如 EXP=uni2h_abmil"; exit 1; fi
args=(--exp "$EXP")
if [ -n "${GPU:-}" ]; then args+=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" "${args[@]}" "$@"
