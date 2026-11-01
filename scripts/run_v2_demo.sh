#!/usr/bin/env bash
# 用假数据把 v2 全流程跑一遍（CPU 即可，约 3–5 分钟），用于在服务器上验证环境。全部为假数据。
# 第一部分（体模 CT，data/fake_v2_ct/）：胃分割 → 勾画抽样 → nnU-Net 数据/推理/后处理 → 特征提取
#   （TotalSegmentator 与 nnU-Net 用假命令代替，只检验本项目的封装与前后处理）；
# 第二部分（多模态假特征队列，data/fake_v2/）：CT 单模态交叉验证 → 编码器初筛 → 影像组学基线 → 比较 → 最终评估。
# 用法：bash scripts/run_v2_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
step() { echo; echo "========== $* =========="; }
C1=configs/fake_v2_ct.yaml
if [ -f data/fake_v2_ct/.fake_data_marker ]; then rm -rf data/fake_v2_ct; fi
TRUTH="$ROOT/data/fake_v2_ct/truth"
TS=(--set "v2.external.totalseg.command=[$PY, -m, gastric_mm.fake_tools.totalseg, --truth-dir, $TRUTH]")
NN=(--set "v2.external.nnunet.predict_command=[$PY, -m, gastric_mm.fake_tools.nnunet_predict]")
step "体模假数据 → 划分 → DICOM 转换"
"$PY" -m gastric_mm.fake_data --config "$C1"
"$PY" -m gastric_mm.split --config "$C1"
"$PY" -m gastric_mm.ct.convert --config "$C1"
mkdir -p "$TRUTH" && mv data/fake_v2_ct/work/ct/stomach/*.nii.gz "$TRUTH"/
step "① 胃分割（假 TotalSegmentator）";   "$PY" -m gastric_mm.ct.stomach_seg --config "$C1" "${TS[@]}"
step "② 勾画病例抽样";                    "$PY" -m gastric_mm.ct.seg_sample --config "$C1"
step "② nnU-Net 训练集与脚本";            "$PY" -m gastric_mm.ct.nnunet_data --config "$C1" --mode train
step "② nnU-Net 推理（假命令）与后处理"
"$PY" -m gastric_mm.ct.nnunet_data --config "$C1" --mode predict --run --include-manual "${NN[@]}"
"$PY" -m gastric_mm.ct.tumor_post --config "$C1"
step "④ CT 特征提取（随机小网络）"
"$PY" -m gastric_mm.ct.extract_features --config "$C1" --encoder random_small_2d
"$PY" -m gastric_mm.ct.extract_features --config "$C1" --encoder random_small_3d

C2=configs/fake_v2.yaml
if [ -f data/fake_v2/.fake_data_marker ]; then rm -rf data/fake_v2; fi
step "多模态假特征队列";                    "$PY" -m gastric_mm.fake_data --config "$C2" --mm-cohort
step "⑥ CT 单模态交叉验证（两次运行模拟两张卡）"
"$PY" -m gastric_mm.ct_mtl.train --config "$C2" --exp demo_ct --folds 0,1,2 --no-aggregate
"$PY" -m gastric_mm.ct_mtl.train --config "$C2" --exp demo_ct --folds 3,4
step "⑥ 编码器初筛（2 个假编码器）+ 影像组学基线 + 比较"
for e in fake_ct_a fake_ct_b; do
  "$PY" -m gastric_mm.ct_mtl.train --config "$C2" --exp "screen_$e" --encoder "$e" --profile screening --set v2.train.epochs=10
done
"$PY" -m gastric_mm.ct_mtl.radiomics_model --config "$C2" --exp radiomics_baseline
"$PY" -m gastric_mm.ct_mtl.compare --config "$C2" --exps screen_fake_ct_a,screen_fake_ct_b,radiomics_baseline
step "⑥ 最终评估演示（内部测试集，--final）"
"$PY" -m gastric_mm.ct_mtl.predict --config "$C2" --exp demo_ct --cohort test --final
echo; echo "全部完成。CT 流程结果在 data/fake_v2_ct/，模型结果在 data/fake_v2/outputs/ct_mtl/。"
