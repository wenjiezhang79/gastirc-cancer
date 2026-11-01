#!/usr/bin/env bash
# 消融实验（编码器选定后运行）：多任务 vs 单任务、ABMIL vs TransMIL vs 均值池化、
# 不确定性加权 vs 固定权重、离散时间生存 vs Cox。
# 用法：ENCODER=uni2_h GPU=0 bash scripts/13_v1_ablation.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ENC="${ENCODER:-uni2_h}"
G=(); if [ -n "${GPU:-}" ]; then G=(--gpu "$GPU"); fi
run() { local exp=$1; shift; echo "========== $exp =========="
  "$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" --exp "$exp" --encoders "$ENC" "${G[@]}" "$@" || echo "【提示】$exp 未完成"; }
run "abl_${ENC}_base"
run "abl_${ENC}_fixedw"   --set v1.loss.weighting=fixed
run "abl_${ENC}_transmil" --set v1.model.aggregator=transmil
run "abl_${ENC}_meanpool" --set v1.model.aggregator=meanpool
run "abl_${ENC}_cox"      --set v1.loss.survival_loss=cox
NONE=(--set "v1.tasks.multiclass=[]" --set "v1.tasks.binary=[]" --set "v1.tasks.regression=[]" --set "v1.tasks.survival=[]")
run "abl_${ENC}_single_lauren" "${NONE[@]}" --set "v1.tasks.multiclass=[lauren]"
for t in her2_pos msi_h ebv_pos pdl1_cps5 cldn182_pos; do
  run "abl_${ENC}_single_${t}" "${NONE[@]}" --set "v1.tasks.binary=[$t]"
done
run "abl_${ENC}_single_os" "${NONE[@]}" --set "v1.tasks.survival=[os]"
MAIN=lauren,her2_pos,msi_h,ebv_pos,pdl1_cps5,cldn182_pos,os
echo "========== 比较：结构与损失 =========="
"$PY" -m gastric_mm.path_mtl.compare --config "$CONFIG" --primary "$MAIN" \
  --exps "abl_${ENC}_base,abl_${ENC}_fixedw,abl_${ENC}_transmil,abl_${ENC}_meanpool,abl_${ENC}_cox" || true
echo "========== 比较：多任务 vs 单任务（逐个任务） =========="
for t in lauren her2_pos msi_h ebv_pos pdl1_cps5 cldn182_pos os; do
  "$PY" -m gastric_mm.path_mtl.compare --config "$CONFIG" --primary "$t" \
    --exps "abl_${ENC}_base,abl_${ENC}_single_${t}" || true
done
