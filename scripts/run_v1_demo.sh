#!/usr/bin/env bash
# 用假特征把 v1 全流程跑一遍（CPU 即可，约 2–3 分钟），用于在服务器上验证环境。
# 假数据写在 data/fake_v1/（埋入已知信号，不含任何真实患者信息）。用法：bash scripts/run_v1_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CONFIG=configs/fake_v1.yaml
PY="${PYTHON:-python}"
step() { echo; echo "========== $* =========="; }
if [ -f data/fake_v1/.fake_data_marker ]; then rm -rf data/fake_v1; fi
step "生成假特征队列";       "$PY" -m gastric_mm.fake_data --config "$CONFIG" --features-only
step "多任务交叉验证（两次运行模拟两张卡）"
"$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" --exp demo_multitask --folds 0,1,2 --no-aggregate
"$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" --exp demo_multitask --folds 3,4
step "编码器初筛（2 个假编码器）"
for e in uni2_h conch_v15; do
  "$PY" -m gastric_mm.path_mtl.train --config "$CONFIG" --exp "screen_$e" --profile screening --encoders "$e" \
    --set v1.train.epochs=10
done
"$PY" -m gastric_mm.path_mtl.compare --config "$CONFIG" --pattern "screen_*"
step "最终评估演示（内部测试集，--final）"
"$PY" -m gastric_mm.path_mtl.predict --config "$CONFIG" --exp demo_multitask --cohort test --final
echo; echo "全部完成。结果在 data/fake_v1/outputs/path_mtl/。"
