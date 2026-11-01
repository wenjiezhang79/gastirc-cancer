#!/usr/bin/env bash
# v3 第一步：模态可用性检查——每例有没有病理特征、CT 肿瘤 ROI、CT 区域 ROI，缺失原因，按中心与人群汇总（不读标签）。
# 用法：bash scripts/30_v3_check.sh
#   结果：outputs/fusion_check/<时间>/availability_report.md 与 modality_availability.csv（只含 study_id）
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.fusion.check --config "$CONFIG" "$@"
