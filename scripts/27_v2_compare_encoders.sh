#!/usr/bin/env bash
# ⑥ CT 编码器初筛（预先规定，只比一次）：每个编码器用预设 screening（Lauren + OS + cT 辅助头，3 个种子），
# 再加影像组学基线，按"折外 OS C-index 与 Lauren 宏平均 AUC 的平均"排名。各编码器的特征须已提取（24_ct_features.sh）。
# 用法：ENCODERS="radimagenet_r50 imagenet_r50 dinov2_vitb14 ctfm merlin suprem" GPU=0 bash scripts/27_v2_compare_encoders.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ENCS="${ENCODERS:-radimagenet_r50 imagenet_r50 dinov2_vitb14 ctfm merlin suprem}"
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
EXPS=()
for e in $ENCS; do
  echo "========== screen_$e =========="
  if "$PY" -m gastric_mm.ct_mtl.train --config "$CONFIG" --exp "screen_$e" --encoder "$e" --profile screening "${G[@]}"; then
    EXPS+=("screen_$e")
  else echo "【提示】screen_$e 未完成（特征是否已提取？）"; fi
done
echo "========== 影像组学基线 =========="
if "$PY" -m gastric_mm.ct_mtl.radiomics_model --config "$CONFIG" --exp radiomics_baseline; then EXPS+=(radiomics_baseline)
else echo "【提示】影像组学基线未完成（需要分割与 25_radiomics.sh）"; fi
echo "========== 比较 =========="
IFS=,; "$PY" -m gastric_mm.ct_mtl.compare --config "$CONFIG" --exps "${EXPS[*]}"
