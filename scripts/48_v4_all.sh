#!/usr/bin/env bash
# v4 一键运行全部分析（40–47），任一步失败会提示并继续下一步。
# 用法：bash scripts/48_v4_all.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for m in table classification prognosis nomogram neoadj immuno explain tables; do
  echo "========== v4 $m =========="
  "$PY" -m gastric_mm.evaluation.$m --config "$CONFIG" "$@" || echo "【提示】v4 $m 未完成，请查看其运行目录下的 run.log"
done
echo "全部完成。结果在 outputs/v4/ 下各子目录（每个子目录有 report.md 与 figs/）。"
