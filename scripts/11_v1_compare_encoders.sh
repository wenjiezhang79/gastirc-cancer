#!/usr/bin/env bash
# 编码器初筛：每个编码器用相同的 ABMIL 和固定超参数，只训练 Lauren + MSI，3 个随机种子（配置 v1.profiles.screening），
# 然后按预先规定的指标（Lauren 宏平均 AUC 与 MSI AUC 的平均）排名。
# 用法：
#   两张卡并行：GPU=0 ENCODERS="uni2_h virchow2 conch_v15 musk" bash scripts/11_v1_compare_encoders.sh
#               GPU=1 ENCODERS="gigapath chief_ctranspath resnet50_trunc" bash scripts/11_v1_compare_encoders.sh
#   全部完成后只做比较：COMPARE_ONLY=1 bash scripts/11_v1_compare_encoders.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ENCODERS="${ENCODERS:-uni2_h virchow2 conch_v15 musk gigapath chief_ctranspath resnet50_trunc}"
if [ -z "${COMPARE_ONLY:-}" ]; then
  for enc in $ENCODERS; do
    echo "========== 初筛：$enc =========="
    args=(--exp "screen_${enc}" --profile screening --encoders "$enc")
    if [ -n "${GPU:-}" ]; then args+=(--gpu "$GPU"); fi
    "$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" "${args[@]}" "$@" || echo "【提示】$enc 未完成，继续下一个"
  done
fi
echo "========== 比较 =========="
"$PY" -m gastric_mm.path_mtl.compare --config "$CONFIG" --pattern "screen_*" || echo "【提示】比较未完成（可能还有编码器没跑完）"
