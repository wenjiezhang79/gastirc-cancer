#!/usr/bin/env bash
# 早停诊断（只读训练记录）：各任务自己最好的 epoch vs 实际停止的 epoch，看总损失早停是否让某些任务停得太早。
# 用法：EXPS=path_mtl/uni2h_abmil,ct_mtl/ct_radimagenet,fusion_main bash scripts/36_v3_diagnose.sh
#   结果：outputs/fusion_diagnose/<时间>/diagnose_report.md（可贴回）
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXPS:-}" ]; then echo "【需要处理】请用 EXPS=实验1,实验2 指定（不带前缀默认在 fusion 下）"; exit 1; fi
"$PY" -m gastric_mm.fusion.diagnose --config "$CONFIG" --exps "$EXPS" "$@"
