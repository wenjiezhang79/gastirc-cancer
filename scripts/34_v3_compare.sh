#!/usr/bin/env bash
# v3 比较报告：融合主模型 vs 最佳单模态 vs 后期融合（预先规定的判定规则），并列出各消融（未完成的自动跳过）。
# 用法：MAIN=fusion_main bash scripts/34_v3_compare.sh
#   结果：outputs/fusion_compare/<时间>/compare_report.md
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MAIN="${MAIN:-fusion_main}"
LATE="${LATE:-late_mean,late_stack}"
DEF=abl_concat,abl_bilinear,abl_visible,abl_noskip,abl_d256,abl_2layers,abl_moddrop0,abl_moddrop05,abl_frozen,abl_scratch,abl_fixedw,abl_aux,abl_tumoronly
DEF="$DEF,st_lauren,st_her2_pos,st_msi_h,st_ebv_pos,st_pdl1_cps5,st_cldn182_pos,st_os"
"$PY" -m gastric_mm.fusion.compare --config "$CONFIG" --main "$MAIN" --late "$LATE" --others "${OTHERS:-$DEF}" \
  --skip-missing "$@"
