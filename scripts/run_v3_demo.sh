#!/usr/bin/env bash
# 用假数据把 v3 全流程跑一遍（CPU 即可，约 5–10 分钟），用于在服务器上验证环境。全部为假数据（data/fake_v3/）。
# 流程：多模态假特征队列 → 模态可用性检查 → v1 病理 / v2 CT 单模态（共用内层验证名单）→ 融合（两次运行模拟两张卡）
#       → 后期融合 → 两项消融 → 比较报告 → 最终评估（内部测试集，--final）
# 用法：bash scripts/run_v3_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
step() { echo; echo "========== $* =========="; }
C=configs/fake_v3.yaml
if [ -f data/fake_v3/.fake_data_marker ]; then rm -rf data/fake_v3; fi
step "多模态假特征队列";          "$PY" -m gastric_mm.fake_data --config "$C" --mm-cohort
step "模态可用性检查";            "$PY" -m gastric_mm.fusion.check --config "$C"
step "v1 病理单模态（demo_path）"; "$PY" -m gastric_mm.path_mtl.train --config "$C" --exp demo_path
step "v2 CT 单模态（demo_ct）";    "$PY" -m gastric_mm.ct_mtl.train --config "$C" --exp demo_ct
step "v3 融合（两次运行模拟两张卡）"
"$PY" -m gastric_mm.fusion.train --config "$C" --exp demo_fusion --folds 0,1,2 --no-aggregate
"$PY" -m gastric_mm.fusion.train --config "$C" --exp demo_fusion --folds 3,4
step "后期融合（平均 / 交叉拟合堆叠）"
for m in mean stack; do "$PY" -m gastric_mm.fusion.late --config "$C" --exp "late_$m" --method "$m"; done
step "消融演示：简单拼接、从头训练"
"$PY" -m gastric_mm.fusion.train --config "$C" --exp abl_concat --set v3.model.fusion=concat
"$PY" -m gastric_mm.fusion.train --config "$C" --exp abl_scratch --set v3.init.path_exp=null --set v3.init.ct_exp=null
step "比较报告（预先规定的判定规则）"
"$PY" -m gastric_mm.fusion.compare --config "$C" --main demo_fusion --late late_mean,late_stack --others abl_concat,abl_scratch
step "最终评估演示（内部测试集，--final）"
"$PY" -m gastric_mm.fusion.predict --config "$C" --exp demo_fusion --cohort test --final
echo; echo "全部完成。结果在 data/fake_v3/outputs/（fusion/、fusion_check/、fusion_compare/）。"
