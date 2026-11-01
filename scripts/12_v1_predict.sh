#!/usr/bin/env bash
# 最终评估（模型定型后只做一次）：5 折模型集成预测内部测试集或外部队列。
# 用法：EXP=uni2h_abmil COHORT=test bash scripts/12_v1_predict.sh --final
#       EXP=uni2h_abmil COHORT=external GPU=0 bash scripts/12_v1_predict.sh --final
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${EXP:-}" ] || [ -z "${COHORT:-}" ]; then
  echo "【需要处理】请用 EXP=实验名 COHORT=test|ext1|ext2|ext3|external 指定"; exit 1
fi
args=(--exp "$EXP" --cohort "$COHORT")
if [ -n "${GPU:-}" ]; then args+=(--gpu "$GPU"); fi
"$PY" -m gastric_mm.path_mtl.predict --config "$CONFIG" "${args[@]}" "$@"
