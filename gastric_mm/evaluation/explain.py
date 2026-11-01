"""v4 可解释性图（docs/decisions.md 第十五节第 8 小节）：病理注意力热图 + 高关注切块拼图，CT 各层注意力。

选例（只用内部训练集折外预测，每例的注意力来自没见过它的那个折模型）：每个任务（v4.explain.tasks）选
预测最有把握的真阳性与真阴性各 n_cases 例（生存任务为风险最高 / 最低者）。
- 病理：注意力叠加在切片缩略图上（有切片文件且装有 openslide 时），否则画在切块坐标上；有切片时另拼出注意力最高的切块。
- CT：肿瘤 ROI / 区域 ROI 各层的注意力权重条形图（2.5D 编码器；3D 编码器只有 1 行时不画）。
文件名只用 study_id 与任务名；图供病理医生判读高关注区域是否对应有意义的形态。

命令行：
    python -m gastric_mm.evaluation.explain --config configs/server.yaml

输出：<output_dir>/v4/explain/<时间>/report.md、cases.csv、figs/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..paths import WorkPaths
from ..stats.plots import _plt, _save, tr as ftr
from ..utils.cli import base_parser, run_main
from ..utils.runio import Run
from ..wsi.features_io import read_features
from .common import ModelPreds, ft, load_covariates, merged_table, setup_figures, v4


def load_attention(exp: Path) -> Dict[str, dict]:
    """读取实验第一个种子各折的 attention.h5 → {sid: {角色: 数组}}；v1 的数据集形状为 [K, N]，转为 {path: [N, K]}。"""
    import h5py

    seeds = sorted(p for p in exp.glob("seed*") if p.is_dir())
    out: Dict[str, dict] = {}
    if not seeds:
        return out
    for f in sorted(seeds[0].glob("fold*/attention.h5")):
        with h5py.File(f, "r") as h:
            for sid in h:
                obj = h[sid]
                if isinstance(obj, h5py.Dataset):
                    a = obj[()]
                    out[sid] = {"path": a.T if a.ndim == 2 else a[:, None]}
                else:
                    out[sid] = {r: obj[r][()] for r in obj}
    return out


def path_encoders(exp: Path) -> list:
    meta = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
    return list(meta.get("path_encoders") or meta.get("encoders") or [])


def select_cases(df: pd.DataFrame, task: str, n: int):
    g = df[df.pred_source == "internal_train"]
    if task in ("os", "dfs", "pfs"):
        r = g[[f"{task}_risk", "study_id"]].dropna().sort_values(f"{task}_risk")
        return [("高风险", s) for s in r.study_id.iloc[::-1][:n]] + [("低风险", s) for s in r.study_id[:n]]
    if task == "lauren":
        out = []
        for k, nm in enumerate(("肠型", "弥漫型", "混合型")):
            c = g[(g.lauren == k)].dropna(subset=[f"lauren_p{k}"]).sort_values(f"lauren_p{k}", ascending=False)
            out += [(nm, s) for s in c.study_id[:n]]
        return out
    col = f"{task}_prob"
    if col not in g:
        return []
    pos = g[g[task] == 1].dropna(subset=[col]).sort_values(col, ascending=False)
    neg = g[g[task] == 0].dropna(subset=[col]).sort_values(col)
    return [("阳性", s) for s in pos.study_id[:n]] + [("阴性", s) for s in neg.study_id[:n]]


def _thumbnail(slide_path: Optional[str], coords: np.ndarray, patch: int):
    """有切片时返回 (缩略图数组, 缩放比例)；否则 None。"""
    if not slide_path:
        return None
    p = Path(slide_path)
    if not p.exists():
        return None
    try:
        import openslide
        sl = openslide.OpenSlide(str(p))
        w, h = sl.dimensions
        scale = 2048 / max(w, h)
        th = sl.get_thumbnail((int(w * scale), int(h * scale)))
        return np.asarray(th), scale, sl
    except Exception:
        return None


def path_figure(sid: str, task: str, label: str, att: np.ndarray, coords: np.ndarray, patch: int, slide, out: Path):
    plt = _plt()
    a = att.mean(1) if att.ndim == 2 else att
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    fig, axes = plt.subplots(1, 2 if slide else 1, figsize=(8 if slide else 4.2, 4))
    ax = axes[0] if slide else axes
    if slide:
        img, scale, _ = slide
        ax.imshow(img)
        ax.scatter((coords[:, 0] + patch / 2) * scale, (coords[:, 1] + patch / 2) * scale, c=a, cmap="jet", s=max(4, patch * scale),
                   marker="s", alpha=0.45, linewidths=0)
    else:
        sc = ax.scatter(coords[:, 0], -coords[:, 1], c=a, cmap="jet", s=12, marker="s", linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.04, label=ftr("注意力（归一化）", "Attention (normalized)"))
        ax.set_aspect("equal")
    ax.set_title(f"{sid} · {task} · {ft(label)}")
    ax.axis("off")
    if slide:
        _, _, sl = slide
        top = np.argsort(-a)[:9]
        tiles = []
        for i in top:
            reg = sl.read_region((int(coords[i, 0]), int(coords[i, 1])), 0, (patch, patch)).convert("RGB")
            tiles.append(np.asarray(reg.resize((128, 128))))
        while len(tiles) < 9:
            tiles.append(np.ones((128, 128, 3), np.uint8) * 255)
        grid = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
        axes[1].imshow(grid)
        axes[1].set_title(ftr("注意力最高的 9 个切块", "Top-9 attention tiles"))
        axes[1].axis("off")
    return _save(fig, out)


def ct_figure(sid: str, task: str, label: str, att: Dict[str, np.ndarray], out: Path):
    roles = [r for r in ("tumor", "regional") if r in att and np.asarray(att[r]).size > 1]
    if not roles:
        return []
    plt = _plt()
    fig, axes = plt.subplots(1, len(roles), figsize=(3.2 * len(roles), 2.4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, roles):
        a = np.asarray(att[r], float).ravel()
        c = len(a) // 2
        ax.bar(np.arange(len(a)) - c, a, color="#1f4e79")
        ax.set_xlabel(ftr("相对中心层的层号", "Slice offset from center"))
        ax.set_ylabel(ftr("注意力", "Attention"))
        ax.set_title(ftr("肿瘤" if r == "tumor" else "区域", "Tumor" if r == "tumor" else "Regional") + " ROI")
    fig.suptitle(f"{sid} · {task} · {ft(label)}", fontsize=9)
    return _save(fig, out)


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    sec = v4(cfg)
    ec = sec.get("explain", {}) or {}
    run = Run(cfg, "v4/explain", args.run_name)
    log = run.logger
    figs = run.dir / "figs"
    figs.mkdir(exist_ok=True)
    setup_figures(cfg, log)
    cov = load_covariates(cfg, log)
    mp = ModelPreds(cfg, sec.models.main)
    df = merged_table(cov, mp)
    att = load_attention(mp.dir)
    encs = path_encoders(mp.dir)
    wp = WorkPaths(cfg)
    from ..manifest import load_manifest
    man = load_manifest(cfg, log, check_paths=False).df.set_index("study_id")
    rows, L = [], ["# v4 可解释性图", "", f"- 模型：{sec.models.main}；选例只用内部训练集折外预测（每例的注意力来自没见过它的折模型）。",
                    "- 病理热图：有切片文件时叠加在缩略图上并拼出注意力最高的 9 个切块；否则画在切块坐标上。", ""]
    n_case = int(ec.get("n_cases", 4))
    for task in ec.get("tasks", ["msi_h", "lauren", "os"]):
        cases = select_cases(df, task, n_case)
        L.append(f"## {task}（{len(cases)} 例）")
        for label, sid in cases:
            a = att.get(sid, {})
            made = []
            if "path" in a and encs:
                fp = wp.wsi_features(encs[0], sid)
                if fp.exists():
                    _, coords, attrs = read_features(fp)
                    if len(coords) == len(a["path"]):
                        patch = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)) or 256)
                        sp = man.loc[sid, "slide_path"] if sid in man.index and "slide_path" in man else None
                        slide = _thumbnail(str(resolve_path(cfg, sp)) if isinstance(sp, str) and sp else None, coords, patch)
                        path_figure(sid, task, label, np.asarray(a["path"], float), coords, patch, slide,
                                    figs / f"path_{task}_{sid}")
                        made.append("病理")
            if ct_figure(sid, task, label, a, figs / f"ct_{task}_{sid}"):
                made.append("CT")
            rows.append({"task": task, "group": label, "study_id": sid, "figures": "+".join(made) or "无注意力"})
        L.append("")
    pd.DataFrame(rows).to_csv(run.path("cases.csv"), index=False, encoding="utf-8-sig")
    L += ["| 任务 | 分组 | study_id | 图 |", "|---|---|---|---|"] + [f"| {r['task']} | {r['group']} | {r['study_id']} | {r['figures']} |" for r in rows]
    run.write_text("\n".join(L), "report.md")
    log.info("可解释性图：%s（%d 例）", figs, len(rows))
    return 0


def main(argv=None) -> int:
    p = base_parser("v4 可解释性图：病理注意力热图与高关注切块、CT 各层注意力（选例只用内部训练集折外预测）。",
                    cases=False, workers=False)
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
