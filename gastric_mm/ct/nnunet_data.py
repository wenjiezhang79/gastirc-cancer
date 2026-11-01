"""② 第二阶段肿瘤分割（nnU-Net v2）：数据集准备、训练/推理脚本、交叉验证 Dice 汇总。

第一阶段胃分割（stomach_seg）给出胃的位置；本模块把每例图像裁到"胃分割外扩 v2.tumor_seg.margin_mm
（并保证中心点周围 ±point_margin_mm 在内）"的框内，作为 nnU-Net 的输入（不重采样，nnU-Net 自己处理间距）。
可选第 2 个输入通道 = 胃分割（v2.tumor_seg.stomach_channel）。

三种模式（--mode）：
- ``train``：用**内部训练集**中的人工勾画（mask_source=manual）生成 nnU-Net 训练集
  ``nnUNet_raw/Dataset<id>_<名称>/``，并生成训练脚本 ``<work_dir>/ct/nnunet/nnunet_train.sh``
  （在 nnU-Net 独立环境中运行，5 折，两张卡并行）。内部测试集和外部中心的勾画不会用于训练。
- ``predict``：为其余病例（默认跳过已有人工勾画的）生成推理输入 ``<work_dir>/ct/nnunet/predict_input/``
  与推理脚本 ``nnunet_predict.sh``；加 ``--run`` 直接调用 nnUNetv2_predict。之后运行 tumor_post 做后处理。
- ``summary``：读取 nnU-Net 5 折验证结果，汇总 Dice（总体与按 cT 分组），报告可贴回。

nnU-Net 的病例名 = ``GC_`` + study_id（非字母数字替换为下划线），对照表在 case_map.json；裁剪框在
``<work_dir>/ct/nnunet/boxes/<study_id>.json``，后处理时用于把结果贴回原图网格。

命令行：
    python -m gastric_mm.ct.nnunet_data --config configs/server.yaml --mode train
    python -m gastric_mm.ct.nnunet_data --config configs/server.yaml --mode predict [--run --gpu 0]
    python -m gastric_mm.ct.nnunet_data --config configs/server.yaml --mode summary
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config, load_config, resolve_path
from ..labels import derive_labels
from ..manifest import load_manifest, select_cases, to_cases
from ..paths import WorkPaths
from ..split import load_split
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import CaseError, UserError
from ..utils.runio import Run
from ..utils.runner import exit_code, process_cases
from . import transforms as T
from .external import describe, resolve_command, run_tool, tool_env


# ---------------- 路径与命名 ----------------
def case_id(sid: str) -> str:
    return "GC_" + re.sub(r"[^A-Za-z0-9]", "_", sid)


def nnunet_dirs(cfg: Config) -> Dict[str, Path]:
    wp = WorkPaths(cfg)
    nc = cfg.v2.external.nnunet
    base = wp.ct_nnunet

    def d(key, default):
        v = resolve_path(cfg, nc.get(key))
        return v if v is not None else base / default

    return {"raw": d("raw_dir", "nnUNet_raw"), "preprocessed": d("preprocessed_dir", "nnUNet_preprocessed"),
            "results": d("results_dir", "nnUNet_results"), "predict_input": base / "predict_input",
            "predict_output": base / "predict_output", "boxes": base / "boxes", "base": base}


def dataset_name(cfg: Config) -> str:
    ts = cfg.v2.tumor_seg
    return f"Dataset{int(ts.dataset_id):03d}_{ts.dataset_name}"


def nnunet_env(cfg: Config) -> Dict[str, str]:
    d = nnunet_dirs(cfg)
    return {"nnUNet_raw": str(d["raw"]), "nnUNet_preprocessed": str(d["preprocessed"]),
            "nnUNet_results": str(d["results"])}


def box_path(cfg: Config, sid: str) -> Path:
    return nnunet_dirs(cfg)["boxes"] / f"{sid}.json"


# ---------------- 裁剪框 ----------------
def _bbox_index(arr_zyx: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    zyx = np.argwhere(arr_zyx > 0)
    if len(zyx) == 0:
        return None
    return zyx.min(0)[::-1].astype(float), zyx.max(0)[::-1].astype(float)   # (x, y, z)


def compute_box(native, stomach_arr: Optional[np.ndarray], point_idx: Optional[List[float]], cfg: Config,
                extra_arr: Optional[np.ndarray] = None) -> Tuple[List[int], List[int], List[str]]:
    """返回 (起点索引 xyz, 大小 xyz, 说明)。框 = 胃分割外扩 margin ∪ 中心点 ± point_margin ∪ extra（训练时的肿瘤）。"""
    ts = cfg.v2.tumor_seg
    sp = np.array(native.GetSpacing())
    size = np.array(native.GetSize())
    lo, hi, notes = None, None, []

    def add(a, b):
        nonlocal lo, hi
        lo = a if lo is None else np.minimum(lo, a)
        hi = b if hi is None else np.maximum(hi, b)

    if stomach_arr is not None:
        bb = _bbox_index(stomach_arr)
        if bb is not None:
            m = float(ts.get("margin_mm", 20)) / sp
            add(bb[0] - m, bb[1] + m)
        else:
            notes.append("胃分割为空")
    else:
        notes.append("没有胃分割")
    if point_idx is not None:
        m = float(ts.get("point_margin_mm", 48)) / sp
        p = np.array(point_idx, dtype=float)
        add(p - m, p + m)
    if extra_arr is not None:
        bb = _bbox_index(extra_arr)
        if bb is not None:
            if lo is not None and (np.any(bb[0] < lo) or np.any(bb[1] > hi)):
                notes.append("肿瘤勾画超出胃外扩范围，已扩大裁剪框")
            add(bb[0] - 2, bb[1] + 2)
    if lo is None:
        raise CaseError("既没有胃分割也没有中心点，无法确定肿瘤分割的范围")
    lo = np.clip(np.floor(lo), 0, size - 1).astype(int)
    hi = np.clip(np.ceil(hi), 0, size - 1).astype(int)
    return lo.tolist(), (hi - lo + 1).tolist(), notes


def crop_to_box(img, start, size):
    import SimpleITK as sitk
    return sitk.RegionOfInterest(img, [int(v) for v in size], [int(v) for v in start])


def _same_grid(a, b) -> bool:
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-3)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-3))


def load_on_native(path: Path, native, binary_resample: bool = True):
    """读取分割并对齐到 native 网格（网格不同时按体积占比重采样）。"""
    import SimpleITK as sitk
    m = sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8)
    if not _same_grid(m, native):
        m = T.resample_mask(m, native) if binary_resample else T.resample_label(m, native)
    m.CopyInformation(native)
    return m


# ---------------- 逐例：写 nnU-Net 输入 ----------------
def prepare_case(case: dict, cfg: Config) -> dict:
    import SimpleITK as sitk

    wp = WorkPaths(cfg)
    sid = case["study_id"]
    mode = case["_mode"]
    native_p = wp.ct_native(sid)
    if not native_p.exists():
        raise CaseError("没有 native NIfTI，请先运行 CT 转换")
    native = sitk.ReadImage(str(native_p))
    st_p = wp.ct_stomach(sid)
    stomach = load_on_native(st_p, native, binary_resample=False) if st_p.exists() else None
    st_arr = sitk.GetArrayFromImage(stomach) if stomach is not None else None
    pt = [case.get(f"ct_point_{a}") for a in "xyz"]
    point_idx = pt if all(v is not None for v in pt) else None
    tumor = None
    if mode == "train":
        mp = resolve_path(cfg, case.get("tumor_mask_path"))
        if mp is None or not mp.exists():
            raise CaseError("清单中的 tumor_mask_path 指向的文件不存在")
        tumor = load_on_native(mp, native)
        if T.mask_volume_ml(tumor) == 0:
            raise CaseError("人工勾画为空")
    start, size, notes = compute_box(native, st_arr, point_idx, cfg,
                                     sitk.GetArrayFromImage(tumor) if tumor is not None else None)
    cid = case["_case_id"]
    if mode == "train":
        d = Path(case["_dataset_dir"])
        img_dir, lab_dir = d / "imagesTr", d / "labelsTr"
    else:
        img_dir, lab_dir = Path(case["_input_dir"]), None
    img_dir.mkdir(parents=True, exist_ok=True)
    roi = crop_to_box(T.to_int16_hu(native, tuple(cfg.ct.hu_clip)), start, size)
    sitk.WriteImage(roi, str(img_dir / f"{cid}_0000.nii.gz"), True)
    if cfg.v2.tumor_seg.get("stomach_channel", True):
        if stomach is None:
            st = sitk.Image(native.GetSize(), sitk.sitkUInt8)
            st.CopyInformation(native)
            notes.append("没有胃分割，第 2 通道全为 0")
        else:
            st = stomach
        sitk.WriteImage(crop_to_box(st, start, size), str(img_dir / f"{cid}_0001.nii.gz"), True)
    res = {"case_id": cid, "box_size": "x".join(str(v) for v in size)}
    if tumor is not None:
        lab_dir.mkdir(parents=True, exist_ok=True)
        lab = crop_to_box(tumor, start, size)
        sitk.WriteImage(lab, str(lab_dir / f"{cid}.nii.gz"), True)
        inside = T.mask_volume_ml(lab) / max(T.mask_volume_ml(tumor), 1e-9)
        res["tumor_ml"] = round(T.mask_volume_ml(tumor), 2)
        if inside < 0.999:
            notes.append(f"裁剪框只包含 {inside * 100:.0f}% 的肿瘤")
    bp = box_path(cfg, sid)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps({"study_id": sid, "case_id": cid, "start_xyz": start, "size_xyz": size,
                              "native_size_xyz": list(native.GetSize())}), encoding="utf-8")
    if notes:
        res["status"] = "warning"
        res["message"] = "；".join(notes)
    return res


# ---------------- 脚本 ----------------
def _bin(cfg: Config) -> str:
    b = cfg.v2.external.nnunet.get("bin_dir")
    return (str(resolve_path(cfg, b)).rstrip("/") + "/") if b else ""


def train_script(cfg: Config) -> str:
    ts = cfg.v2.tumor_seg
    env = nnunet_env(cfg)
    b = _bin(cfg)
    folds = [int(f) for f in ts.folds]
    ga, gb = folds[0::2], folds[1::2]
    did, conf, tr = int(ts.dataset_id), ts.configuration, ts.trainer
    L = ["#!/usr/bin/env bash",
         "# 由 gastric_mm.ct.nnunet_data --mode train 生成。在 nnU-Net 独立环境中运行（离线）：",
         "#   bash nnunet_train.sh            # 默认两张卡：GPU_A=0 跑偶数折，GPU_B=1 跑奇数折",
         "#   GPU_A=0 GPU_B=0 bash nnunet_train.sh   # 只有一张卡时",
         "# 训练需要数天；建议用 nohup bash nnunet_train.sh > nnunet_train.log 2>&1 & 在后台运行。",
         "set -euo pipefail"]
    L += [f'export {k}="{v}"' for k, v in env.items()]
    L += ['export HF_HUB_OFFLINE=1', 'GPU_A="${GPU_A:-0}"', 'GPU_B="${GPU_B:-1}"',
          f'"{b}nnUNetv2_plan_and_preprocess" -d {did} -c {conf} --verify_dataset_integrity',
          f'(for f in {" ".join(map(str, ga))}; do CUDA_VISIBLE_DEVICES="$GPU_A" "{b}nnUNetv2_train" {did} {conf} "$f" -tr {tr} --npz; done) &']
    if gb:
        L.append(f'(for f in {" ".join(map(str, gb))}; do CUDA_VISIBLE_DEVICES="$GPU_B" "{b}nnUNetv2_train" {did} {conf} "$f" -tr {tr} --npz; done) &')
    L += ["wait", 'echo "nnU-Net 5 折训练完成。下一步：python -m gastric_mm.ct.nnunet_data --mode summary 汇总 Dice。"', ""]
    return "\n".join(L)


def predict_args(cfg: Config, gpu: Optional[int]) -> List[str]:
    ts = cfg.v2.tumor_seg
    nc = cfg.v2.external.nnunet
    d = nnunet_dirs(cfg)
    cmd = nc.get("predict_command") or f"{_bin(cfg)}nnUNetv2_predict"
    base = resolve_command(cmd, "nnUNetv2_predict", "v2.external.nnunet.bin_dir / predict_command")
    return base + ["-i", str(d["predict_input"]), "-o", str(d["predict_output"]), "-d", str(int(ts.dataset_id)),
                   "-c", str(ts.configuration), "-f", *[str(int(f)) for f in ts.folds], "-tr", str(ts.trainer),
                   "-device", "cuda" if gpu is not None else "cpu"]


def predict_script(cfg: Config) -> str:
    ts = cfg.v2.tumor_seg
    d = nnunet_dirs(cfg)
    b = _bin(cfg)
    L = ["#!/usr/bin/env bash", "# 由 gastric_mm.ct.nnunet_data --mode predict 生成。在 nnU-Net 独立环境中运行：",
         "#   GPU=0 bash nnunet_predict.sh", "set -euo pipefail"]
    L += [f'export {k}="{v}"' for k, v in nnunet_env(cfg).items()]
    L += ['export HF_HUB_OFFLINE=1', 'export CUDA_VISIBLE_DEVICES="${GPU:-0}"',
          f'"{b}nnUNetv2_predict" -i "{d["predict_input"]}" -o "{d["predict_output"]}" -d {int(ts.dataset_id)} '
          f'-c {ts.configuration} -f {" ".join(str(int(f)) for f in ts.folds)} -tr {ts.trainer}',
          'echo "推理完成。下一步：python -m gastric_mm.ct.tumor_post --config <配置> 做后处理与审核清单。"', ""]
    return "\n".join(L)


# ---------------- 交叉验证汇总 ----------------
def summarize_cv(cfg: Config, labels: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    ts = cfg.v2.tumor_seg
    d = nnunet_dirs(cfg)
    model_dir = d["results"] / dataset_name(cfg) / f"{ts.trainer}__nnUNetPlans__{ts.configuration}"
    cmap_p = d["raw"] / dataset_name(cfg) / "case_map.json"
    cmap = json.loads(cmap_p.read_text(encoding="utf-8")) if cmap_p.exists() else {}
    rows = []
    found = []
    for f in ts.folds:
        sp = model_dir / f"fold_{int(f)}" / "validation" / "summary.json"
        if not sp.exists():
            continue
        found.append(int(f))
        s = json.loads(sp.read_text(encoding="utf-8"))
        for item in s.get("metric_per_case", []):
            ref = Path(str(item.get("reference_file", ""))).name
            cid = ref[: -len(".nii.gz")] if ref.endswith(".nii.gz") else Path(ref).stem
            met = item.get("metrics", {})
            m1 = met.get("1", met.get(1, {})) if isinstance(met, dict) else {}
            rows.append({"study_id": cmap.get(cid, ""), "fold": int(f), "dice": m1.get("Dice"),
                         "n_ref": m1.get("n_ref"), "n_pred": m1.get("n_pred")})
    if not found:
        raise UserError("没有找到 nnU-Net 的交叉验证结果（fold_*/validation/summary.json）",
                        "请确认 nnunet_train.sh 已全部跑完，且 v2.external.nnunet.results_dir 与训练时一致。")
    df = pd.DataFrame(rows)
    df = df.merge(labels[["study_id", "ct_t_group"]], on="study_id", how="left")
    dice = pd.to_numeric(df["dice"], errors="coerce")
    L = ["# nnU-Net 肿瘤分割：交叉验证 Dice 汇总", "",
         f"- 数据集：{dataset_name(cfg)}；配置 {ts.configuration}；已完成的折：{found}",
         f"- 病例：{len(df)}；Dice 均值 {dice.mean():.3f}，中位数 {dice.median():.3f}，"
         f"四分位 {dice.quantile(0.25):.3f}–{dice.quantile(0.75):.3f}",
         f"- Dice < 0.3（基本失败）：{int((dice < 0.3).sum())} 例；< 0.5：{int((dice < 0.5).sum())} 例", "",
         "## 按 cT 分组", "", "| cT | 例数 | Dice 均值 | 中位数 | <0.5 例数 |", "|---|---|---|---|---|"]
    for g, sub in df.groupby(df["ct_t_group"].fillna("缺失")):
        v = pd.to_numeric(sub["dice"], errors="coerce")
        L.append(f"| {g} | {len(sub)} | {v.mean():.3f} | {v.median():.3f} | {int((v < 0.5).sum())} |")
    L += ["", "## 各折", "", "| 折 | 例数 | Dice 均值 |", "|---|---|---|"]
    for f, sub in df.groupby("fold"):
        L.append(f"| {f} | {len(sub)} | {pd.to_numeric(sub['dice'], errors='coerce').mean():.3f} |")
    L += ["", "逐例结果见 dice_per_case.csv（只含研究编号）。早期（T1）肿瘤 Dice 偏低是常见现象；"
          "自动分割的结果都要经医生审核后才作为 auto_reviewed 使用。", ""]
    return df, "\n".join(L)


# ---------------- 命令行 ----------------
def _cases_for(mode: str, df: pd.DataFrame, cfg: Config, log, include_manual: bool) -> List[dict]:
    split = load_split(cfg)[["study_id", "set"]]
    m = df.merge(split, on="study_id", how="left")
    cases = to_cases(m)
    wp = WorkPaths(cfg)
    cases = [c for c in cases if c.get("ct_dir") and wp.ct_native(c["study_id"]).exists()]
    has_manual = [c for c in cases if c.get("mask_source") == "manual" and c.get("tumor_mask_path")]
    if mode == "train":
        out = [c for c in has_manual if c.get("set") == "train"]
        other = [c for c in has_manual if c.get("set") != "train"]
        if other:
            log.warning("%d 例人工勾画属于内部测试集或外部中心，按规定不用于训练 nnU-Net", len(other))
        return out
    if include_manual:
        return cases
    return [c for c in cases if not (c.get("mask_source") == "manual" and c.get("tumor_mask_path"))]


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, f"ct_nnunet_{args.mode}", args.run_name)
    log = run.logger
    d = nnunet_dirs(cfg)
    res = load_manifest(cfg, log, check_paths=False)
    if args.mode == "summary":
        labels = derive_labels(res.df, cfg)
        df, rep = summarize_cv(cfg, labels)
        df.drop(columns=["ct_t_group"]).to_csv(run.path("dice_per_case.csv"), index=False, encoding="utf-8-sig")
        run.write_text(rep, "nnunet_cv_summary.md")
        log.info("汇总完成：%s", run.path("nnunet_cv_summary.md"))
        return 0
    cases = _cases_for(args.mode, res.df, cfg, log, args.include_manual)
    cases = select_cases(cases, parse_id_list(args.study_ids), args.limit)
    if not cases:
        raise UserError("没有可处理的病例" + ("（内部训练集中没有人工勾画）" if args.mode == "train" else ""),
                        "训练模式需要先完成人工勾画并在清单中填写 tumor_mask_path、mask_source=manual；"
                        "两种模式都需要先完成 CT 转换与胃分割。")
    ids = {}
    for c in cases:
        cid = case_id(c["study_id"])
        if cid in ids:
            raise UserError(f"study_id {c['study_id']} 与 {ids[cid]} 转换成 nnU-Net 病例名后重复")
        ids[cid] = c["study_id"]
        c.update(_mode=args.mode, _case_id=cid)
    if args.mode == "train":
        ds = d["raw"] / dataset_name(cfg)
        if ds.exists() and any(ds.iterdir()) and not args.overwrite:
            raise UserError(f"nnU-Net 数据集 {ds.name} 已存在",
                            "如人工勾画有更新、需要重新生成，加 --overwrite（会删除旧数据集；已训练的模型需要重训）。")
        if ds.exists():
            import shutil
            shutil.rmtree(ds)
        for c in cases:
            c["_dataset_dir"] = str(ds)
        results = process_cases(cases, prepare_case, cfg, log, 1, "生成 nnU-Net 训练集")
        ok = [r for r in results if r.get("status") != "failed"]
        ch = {"0": "CT"}
        if cfg.v2.tumor_seg.get("stomach_channel", True):
            ch["1"] = "noNorm"
        dj = {"channel_names": ch, "labels": {"background": 0, "tumor": 1}, "numTraining": len(ok),
              "file_ending": ".nii.gz", "overwrite_image_reader_writer": "SimpleITKIO",
              "description": "gastric tumor within stomach ROI (gastric_mm v2)"}
        (ds / "dataset.json").write_text(json.dumps(dj, indent=2), encoding="utf-8")
        (ds / "case_map.json").write_text(json.dumps({r["case_id"]: r["study_id"] for r in ok}, indent=2),
                                          encoding="utf-8")
        script = d["base"] / "nnunet_train.sh"
        script.write_text(train_script(cfg), encoding="utf-8")
        run.write_text(train_script(cfg), "nnunet_train.sh")
        run.write_table(results, "status.csv", ["study_id", "status", "case_id", "box_size", "tumor_ml", "message"])
        log.info("训练集 %d 例：%s。下一步在 nnU-Net 环境中运行：bash %s", len(ok), ds, script)
        if len(ok) < 50:
            log.warning("训练病例只有 %d 例（计划 200–300 例），分割效果可能不稳定", len(ok))
        return exit_code(results)

    # predict
    inp = d["predict_input"]
    if inp.exists() and not args.keep_existing:
        import shutil
        shutil.rmtree(inp)
    for c in cases:
        c["_input_dir"] = str(inp)
    results = process_cases(cases, prepare_case, cfg, log, 1, "生成 nnU-Net 推理输入")
    ok = [r for r in results if r.get("status") != "failed"]
    inp.mkdir(parents=True, exist_ok=True)
    (d["base"] / "predict_case_map.json").write_text(
        json.dumps({r["case_id"]: r["study_id"] for r in ok}, indent=2), encoding="utf-8")
    script = d["base"] / "nnunet_predict.sh"
    script.write_text(predict_script(cfg), encoding="utf-8")
    run.write_text(predict_script(cfg), "nnunet_predict.sh")
    run.write_table(results, "status.csv", ["study_id", "status", "case_id", "box_size", "message"])
    log.info("推理输入 %d 例：%s", len(ok), inp)
    if args.run:
        argv = predict_args(cfg, args.gpu)
        log.info("运行：%s", describe(argv))
        d["predict_output"].mkdir(parents=True, exist_ok=True)
        try:
            run_tool(argv, tool_env(nnunet_env(cfg), args.gpu), float(cfg.v2.external.nnunet.get("timeout_s", 172800)),
                     "nnUNetv2_predict")
        except CaseError as e:
            raise UserError(str(e), "请确认 nnU-Net 环境与模型（--mode train 后的 5 折训练）已就绪；"
                            "也可以在 nnU-Net 环境中手动运行 nnunet_predict.sh。") from None
        log.info("推理完成。下一步：python -m gastric_mm.ct.tumor_post --config <配置>")
    else:
        log.info("下一步在 nnU-Net 环境中运行：GPU=0 bash %s；或加 --run 由本程序调用", script)
    return exit_code(results)


def main(argv=None) -> int:
    p = base_parser("nnU-Net 肿瘤分割：生成训练集/推理输入与脚本，汇总交叉验证 Dice。", workers=False, gpu=True)
    p.add_argument("--mode", required=True, choices=["train", "predict", "summary"],
                   help="train=生成训练集；predict=生成推理输入（--run 直接推理）；summary=汇总 5 折 Dice")
    p.add_argument("--run", action="store_true", help="predict 模式：生成输入后直接调用 nnUNetv2_predict")
    p.add_argument("--include-manual", action="store_true", help="predict 模式：已有人工勾画的病例也做自动分割")
    p.add_argument("--keep-existing", action="store_true", help="predict 模式：保留推理输入目录中已有的文件")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
