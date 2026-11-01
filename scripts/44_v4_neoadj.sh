#!/usr/bin/env bash
# v4 探索：新辅助疗效预测（pCR / TRG 0–1），主模型输出 + 临床变量的 L1 逻辑回归，不改主模型。
# 用法：bash scripts/44_v4_neoadj.sh
#   统计分析计划见 docs/decisions.md 第十五节；参与分析的模型在配置 v4.models 中填写。
#   没有最终评估（--final）结果时为"开发模式"：只分析内部训练集折外预测。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.evaluation.neoadj --config "$CONFIG" "$@"
