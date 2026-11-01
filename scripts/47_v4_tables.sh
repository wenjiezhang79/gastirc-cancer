#!/usr/bin/env bash
# v4 论文用表：表 1、病例流程图人数、TRIPOD+AI 清单对照。
# 用法：bash scripts/47_v4_tables.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.tables --config "$CONFIG" "$@"
