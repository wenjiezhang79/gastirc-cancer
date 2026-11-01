#!/usr/bin/env bash
# 公共设置：切换到仓库根目录；配置文件默认 configs/server.yaml，可用环境变量 CONFIG 指定；
# Python 解释器默认 python，可用环境变量 PYTHON 指定（如某个 conda 环境的 python）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/server.yaml}"
PY="${PYTHON:-python}"
if [ ! -f "$CONFIG" ]; then
  echo "【需要处理】找不到配置文件 $CONFIG"
  echo "  → 怎么办：cp configs/server_example.yaml configs/server.yaml 后修改其中的路径；"
  echo "    或用 CONFIG=其他配置文件 bash scripts/xxx.sh 指定。"
  exit 1
fi
