#!/usr/bin/env bash
# ② nnU-Net 数据准备。MODE=train：用内部训练集的人工勾画生成训练集与训练脚本；
#   MODE=predict：生成其余病例的推理输入与推理脚本（加 --run 直接推理）；MODE=summary：汇总 5 折 Dice。
# 用法：MODE=train bash scripts/22_nnunet_prepare.sh
#       MODE=predict GPU=0 bash scripts/22_nnunet_prepare.sh --run
#       MODE=summary bash scripts/22_nnunet_prepare.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.ct.nnunet_data --config "$CONFIG" --mode "${MODE:-train}" "${G[@]}" "$@"
