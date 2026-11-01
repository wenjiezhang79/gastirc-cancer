#!/usr/bin/env bash
# v3 消融全套（docs/decisions.md 第十四节第 6 小节）。每项是一个独立实验（实验名见下），已完成的折自动跳过。
# 用法：GPU=0 bash scripts/33_v3_ablation.sh
#   两张卡分担：GPU=0 PART=1 bash scripts/33_v3_ablation.sh   与   GPU=1 PART=2 bash scripts/33_v3_ablation.sh
#   只跑其中几项：ONLY="abl_concat abl_frozen" GPU=0 bash scripts/33_v3_ablation.sh
#   节省时间（非必做消融用 1 个种子）：SEEDS=0 GPU=0 bash scripts/33_v3_ablation.sh
# 必做：abl_concat（简单拼接）、st_*（单任务）；后期融合见 32_v3_late_fusion.sh；最佳单模态直接用 v1 / v2 实验。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
S=(); if [ -n "${SEEDS:-}" ]; then S=(--seeds "$SEEDS"); fi
declare -a NAMES=() ARGS=()
add() { NAMES+=("$1"); shift; ARGS+=("$*"); }
# ---- 融合结构 ----
add abl_concat      --set v3.model.fusion=concat
add abl_bilinear    --set v3.model.fusion=bilinear
add abl_visible     --set v3.model.missing=visible
add abl_d256        --set v3.model.d_model=256
add abl_2layers     --set v3.model.n_layers=2
# ---- 训练策略 ----
add abl_moddrop0    --set v3.train.modality_dropout=0.0
add abl_moddrop05   --set v3.train.modality_dropout=0.5
add abl_frozen      --set v3.train.freeze_branches=true
add abl_scratch     --set v3.init.path_exp=null --set v3.init.ct_exp=null
add abl_fixedw      --set v3.loss.weighting=fixed
add abl_aux         --set v3.model.aux_heads=true
add abl_tumoronly   --set "v3.ct_rois=[tumor]"
# ---- 单任务（必做） ----
for t in lauren her2_pos msi_h ebv_pos pdl1_cps5 cldn182_pos os; do add "st_$t" --single-task "$t"; done
N=${#NAMES[@]}
set -f   # 参数中含 [tumor] 等，关闭通配符展开
for ((i = 0; i < N; i++)); do
  name="${NAMES[$i]}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  if [ "${PART:-}" = "1" ] && (( i % 2 == 1 )); then continue; fi
  if [ "${PART:-}" = "2" ] && (( i % 2 == 0 )); then continue; fi
  echo "========== $name =========="
  # shellcheck disable=SC2086
  "$PY" -m gastric_mm.fusion.train --config "$CONFIG" --exp "$name" "${G[@]}" "${S[@]}" ${ARGS[$i]} "$@" \
    || echo "【提示】$name 未完成，请查看该实验 logs/ 下的日志"
done
echo "全部消融已运行。比较报告：bash scripts/34_v3_compare.sh"
