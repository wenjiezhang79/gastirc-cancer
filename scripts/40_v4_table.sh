#!/usr/bin/env bash
# v4 分析用数据表：合并各模型预测、协变量与冻结量（另供 R / SPSS 复核；只在服务器保存，不要放进代码仓库）。
# 用法：bash scripts/40_v4_table.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.table --config "$CONFIG" "$@"
