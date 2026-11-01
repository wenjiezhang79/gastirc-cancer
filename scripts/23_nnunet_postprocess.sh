#!/usr/bin/env bash
# ② nnU-Net 推理结果后处理：贴回原图、按中心点选肿瘤连通域、生成候选分割与审核清单。
# 用法：bash scripts/23_nnunet_postprocess.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" -m gastric_mm.ct.tumor_post --config "$CONFIG" "$@"
