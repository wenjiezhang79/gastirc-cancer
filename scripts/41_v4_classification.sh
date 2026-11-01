#!/usr/bin/env bash
# v4 分类任务评价：AUC（DeLong）、PR、敏感度 / 特异度、模型间 DeLong 比较、校准（外部另做截距重新校准）。
# 用法：bash scripts/41_v4_classification.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.classification --config "$CONFIG" "$@"
