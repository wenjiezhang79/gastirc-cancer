#!/usr/bin/env bash
# ② 导入医生审核后的分割（mask_source=auto_reviewed）。
# 用法：SRC=<审核后分割目录> bash scripts/23b_import_reviewed.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ -z "${SRC:-}" ]; then echo "【需要处理】请用 SRC=审核后分割所在目录 指定"; exit 1; fi
"$PY" -m gastric_mm.ct.import_reviewed --config "$CONFIG" --src-dir "$SRC" "$@"
