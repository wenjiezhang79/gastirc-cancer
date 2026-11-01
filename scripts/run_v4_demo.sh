#!/usr/bin/env bash
# 用假数据把 v4 全流程跑一遍（CPU 约 10 分钟），用于在服务器上验证环境。全部为假数据（data/fake_v4/），含已知的治疗效应。
# 流程：多模态假队列（含治疗、TRG/pCR、免疫场景）→ v1 / v2 / v3 主模型与后期融合 → 各模型对内部测试集与外部队列做一次 --final
#       → v4 全部分析（分析表、分类、预后四层、列线图、新辅助、免疫亚组、可解释性图、论文用表）
# 用法：bash scripts/run_v4_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
step() { echo; echo "========== $* =========="; }
C=configs/fake_v4.yaml
if [ -f data/fake_v4/.fake_data_marker ]; then rm -rf data/fake_v4; fi
step "多模态假队列（含治疗、TRG/pCR、免疫场景）"; "$PY" -m gastric_mm.fake_data --config "$C" --mm-cohort
step "v1 病理 / v2 CT / v3 融合 / 后期融合"
"$PY" -m gastric_mm.path_mtl.train --config "$C" --exp demo_path
"$PY" -m gastric_mm.ct_mtl.train --config "$C" --exp demo_ct
"$PY" -m gastric_mm.fusion.train --config "$C" --exp demo_fusion
"$PY" -m gastric_mm.fusion.late --config "$C" --exp late_stack --method stack
step "最终评估（演示：各模型对内部测试集与外部队列各运行一次 --final）"
for c in test external; do
  "$PY" -m gastric_mm.path_mtl.predict --config "$C" --exp demo_path --cohort "$c" --final
  "$PY" -m gastric_mm.ct_mtl.predict --config "$C" --exp demo_ct --cohort "$c" --final
  "$PY" -m gastric_mm.fusion.predict --config "$C" --exp demo_fusion --cohort "$c" --final
  "$PY" -m gastric_mm.fusion.predict --config "$C" --exp late_stack --cohort "$c" --final
done
step "v4 全部分析"
for m in table classification prognosis nomogram neoadj immuno explain tables; do
  "$PY" -m gastric_mm.evaluation.$m --config "$C"
done
echo; echo "全部完成。结果在 data/fake_v4/outputs/v4/。"
