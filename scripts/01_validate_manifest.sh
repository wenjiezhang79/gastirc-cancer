#!/usr/bin/env bash
# 病例清单校验，生成报告。用法：bash scripts/01_validate_manifest.sh [--no-path-check]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.manifest --config "$CONFIG" "$@"
