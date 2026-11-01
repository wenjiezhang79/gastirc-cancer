#!/usr/bin/env bash
# ⑤ 影像组学特征提取：在 pyradiomics 独立环境中运行。
# 用法：RADIOMICS_PY=/opt/envs/radiomics/bin/python bash scripts/25_radiomics.sh [--num-workers 8]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RPY="${RADIOMICS_PY:-$PY}"
"$RPY" -m gastric_mm.ct.radiomics --config "$CONFIG" "$@"
