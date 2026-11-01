#!/usr/bin/env bash
# v4 探索：免疫治疗亚组（分场景、逆概率加权、预测标志物 × 免疫交互）。
# 用法：bash scripts/45_v4_immuno.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.immuno --config "$CONFIG" "$@"
