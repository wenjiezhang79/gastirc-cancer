#!/usr/bin/env bash
# v4 列线图：风险评分 + cTNM（+ 年龄），内部训练集拟合后冻结，各队列验证 C-index、校准与决策曲线。
# 用法：bash scripts/43_v4_nomogram.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.nomogram --config "$CONFIG" "$@"
