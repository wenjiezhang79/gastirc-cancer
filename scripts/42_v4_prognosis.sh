#!/usr/bin/env bash
# v4 预后分析四层：整体（KM、C-index、多因素 Cox、增量价值、成功标准）、分期 / 治疗分层、中心 / 校准 / 决策曲线、交互作用。
# 用法：bash scripts/42_v4_prognosis.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.prognosis --config "$CONFIG" "$@"
