#!/usr/bin/env bash
# v3 后期融合对照：组合 v1 病理与 v2 CT 的折外预测（平均 + 交叉拟合堆叠），不需要 GPU，几分钟内完成。
# 用法：bash scripts/32_v3_late_fusion.sh            # 生成 fusion/late_mean 与 fusion/late_stack
#       PREFIX=late2 bash scripts/32_v3_late_fusion.sh  # 换实验名前缀
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
P="${PREFIX:-late}"
for m in mean stack; do
  echo "========== 后期融合：${P}_${m} =========="
  "$PY" -m gastric_mm.fusion.late --config "$CONFIG" --exp "${P}_${m}" --method "$m" "$@"
done
