#!/usr/bin/env bash
# v4 可解释性图：病理注意力热图与高关注切块、CT 各层注意力（选例只用内部训练集折外预测）。
# 用法：bash scripts/46_v4_explain.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.explain --config "$CONFIG" "$@"
