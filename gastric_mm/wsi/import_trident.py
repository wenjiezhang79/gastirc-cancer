"""导入 TRIDENT 已提取的病理特征，转成按 study_id 命名的统一格式。

TRIDENT 输出结构（默认设置）：
    <job_dir>/<倍率>x_<切块>px_<重叠>px_overlap/features_<编码器>/<切片文件名>.h5
    每个 h5：features [N, D]、coords [N, 2]（level-0），coords 的属性里有 patch_size、level0_magnification 等。

本工具：
1. 自动识别倍率/切块大小目录和其中的全部编码器（目录名 → 本项目编码器名见配置 trident.name_map）。
2. 用清单 slide_path 的文件名（不含扩展名）对应 TRIDENT 的 h5 文件名 → study_id。
3. 检查：特征维数是否符合预期、特征与坐标数是否一致、是否有 NaN、切块数是否过少、
   同一切片在不同编码器之间的切块坐标是否一致。
4. 写出 ``<work_dir>/wsi/features/<编码器>/<study_id>.h5``；**不复制** TRIDENT 中的原文件名和保存路径属性。

输出（运行目录）：
- trident_import_report.md   各编码器的导入统计与检查结果（只含 study_id，可贴回）
- status.csv                 每个（编码器, 病例）一行
- private/unmatched_trident_files.csv   清单中对不上的 TRIDENT 文件名（可能含住院号，不要外传）

命令行：
    python -m gastric_mm.wsi.import_trident --config configs/server.yaml [--mag-dir 20x_256px_0px_overlap]
        [--encoders uni_v2,conch_v15] [--overwrite]
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config, load_config, resolve_path
from ..manifest import load_manifest, select_cases, to_cases
from ..paths import WorkPaths
from ..utils.cli import base_parser, parse_id_list, run_main
from ..utils.errors import UserError
from ..utils.runio import Run, write_csv
from .features_io import _clean_attr, write_features

MAG_DIR_RE = re.compile(r"^(\d+(?:\.\d+)?)x_(\d+)px_(\d+)px_overlap$")
COPY_ATTRS = ("patch_size", "patch_size_level0", "level0_magnification", "target_magnification", "overlap")


def find_mag_dirs(job_dir: Path) -> List[Path]:
    return sorted(p for p in job_dir.iterdir() if p.is_dir() and MAG_DIR_RE.match(p.name))


def slide_key(path_str: str) -> str:
    name = Path(str(path_str)).name
    for ext in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(ext):
            return name[: -len(ext)].lower()
    return Path(name).stem.lower()


def read_trident_h5(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    import h5py

    with h5py.File(path, "r") as f:
        if "features" not in f:
            raise ValueError("文件中没有 features 数据集")
        feats = f["features"][()]
        coords = f["coords"][()] if "coords" in f else None
        attrs = {}
        for src in (f.attrs, f["coords"].attrs if "coords" in f else {}):
            for k in COPY_ATTRS:
                if k in src:
                    attrs[k] = _clean_attr(src[k])
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords is not None and coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return feats, coords, attrs


def _run(args) -> int:
    cfg = load_config(args.config, args.set)
    run = Run(cfg, "import_trident", args.run_name)
    log = run.logger
    tc = cfg.trident
    if not tc.get("job_dir"):
        raise UserError("配置中没有填写 trident.job_dir", "请填写 TRIDENT 输出的根目录（含 20x_256px_0px_overlap 之类的子目录）。")
    job_dir = resolve_path(cfg, tc.job_dir)
    if not job_dir.is_dir():
        raise UserError("trident.job_dir 指向的目录不存在", "请检查配置中的 trident.job_dir。")
    mags = find_mag_dirs(job_dir)
    if args.mag_dir:
        mags = [m for m in mags if m.name == args.mag_dir]
    if not mags:
        raise UserError("在 trident.job_dir 下没有找到形如 20x_256px_0px_overlap 的目录",
                        "请确认 job_dir 填的是 TRIDENT 的输出根目录。")
    if len(mags) > 1:
        raise UserError(f"找到多个倍率/切块目录：{', '.join(m.name for m in mags)}",
                        "请用 --mag-dir 指定要导入哪一个。")
    mag = mags[0]
    mm = MAG_DIR_RE.match(mag.name)
    mag_info = {"magnification": float(mm.group(1)), "patch_px": int(mm.group(2)), "overlap_px": int(mm.group(3))}
    log.info("TRIDENT 目录：%s（%gx，切块 %d px，重叠 %d px）", mag.name, mag_info["magnification"],
             mag_info["patch_px"], mag_info["overlap_px"])

    enc_dirs = sorted(p for p in mag.iterdir() if p.is_dir() and p.name.startswith("features_"))
    want = parse_id_list(args.encoders)
    if want:
        enc_dirs = [p for p in enc_dirs if p.name[len("features_"):] in want]
    if not enc_dirs:
        raise UserError("没有找到 features_<编码器> 目录", "请确认 TRIDENT 已完成特征提取（--task feat）。")
    name_map = dict(tc.get("name_map", {}) or {})
    expected = dict(tc.get("expected_dims", {}) or {})
    min_tiles = int(tc.get("min_tiles_warn", 10))

    res = load_manifest(cfg, log, check_paths=False)
    cases = select_cases(to_cases(res.df), parse_id_list(args.study_ids), args.limit)
    key2sid: Dict[str, str] = {}
    dup: Dict[str, List[str]] = defaultdict(list)
    for c in cases:
        if c.get("slide_path"):
            k = slide_key(c["slide_path"])
            dup[k].append(c["study_id"])
            key2sid[k] = c["study_id"]
    dups = {k: v for k, v in dup.items() if len(v) > 1}
    if dups:
        ids = sorted({s for v in dups.values() for s in v})
        raise UserError(f"清单中有 {len(dups)} 组切片文件名相同（不含扩展名），无法与 TRIDENT 文件一一对应："
                        f"{', '.join(ids[:20])}", "请检查这些病例的 slide_path。")
    with_slide = {c["study_id"] for c in cases if c.get("slide_path")}

    wp = WorkPaths(cfg)
    status: List[dict] = []
    unmatched: List[dict] = []
    per_enc: "OrderedDict[str, dict]" = OrderedDict()
    coords_by_sid: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    for ed in enc_dirs:
        tname = ed.name[len("features_"):]
        ename = name_map.get(tname, tname)
        files = sorted(ed.glob("*.h5"))
        st = {"trident_name": tname, "n_files": len(files), "matched": 0, "imported": 0, "skipped": 0,
              "failed": 0, "dims": set(), "tiles": [], "few_tiles": [], "unmatched": 0}
        log.info("编码器 %s → %s：%d 个文件", tname, ename, len(files))
        for f in files:
            sid = key2sid.get(f.stem.lower())
            if sid is None:
                st["unmatched"] += 1
                unmatched.append({"encoder": tname, "file": f.name})
                continue
            st["matched"] += 1
            row = {"encoder": ename, "study_id": sid}
            out = wp.wsi_features(ename, sid)
            try:
                feats, coords, attrs = read_trident_h5(f)
                if feats.ndim != 2:
                    raise ValueError(f"特征形状异常 {feats.shape}")
                if coords is None:
                    coords = np.full((len(feats), 2), -1, np.int64)
                    row["message"] = "缺少坐标"
                if len(coords) != len(feats):
                    raise ValueError(f"特征数 {len(feats)} 与坐标数 {len(coords)} 不一致")
                if not np.isfinite(feats).all():
                    raise ValueError("特征中有 NaN/Inf")
                coords_by_sid[sid][ename] = (len(coords), hash(np.asarray(coords, np.int64).tobytes()))
                st["dims"].add(int(feats.shape[1]))
                st["tiles"].append(int(len(feats)))
                if len(feats) < min_tiles:
                    st["few_tiles"].append(sid)
                exp = expected.get(ename)
                if exp and int(exp) != int(feats.shape[1]):
                    row["message"] = f"维数 {feats.shape[1]} 与预期 {exp} 不符"
                row.update(n_tiles=int(len(feats)), feature_dim=int(feats.shape[1]))
                if out.exists() and not args.overwrite:
                    st["skipped"] += 1
                    row["status"] = "skipped"
                else:
                    a = {"encoder": ename, "source": "trident", "trident_encoder": tname, "trident_dir": mag.name,
                         "magnification": mag_info["magnification"], "tile_px": mag_info["patch_px"],
                         "overlap_px": mag_info["overlap_px"]}
                    a.update(attrs)
                    write_features(out, feats, coords, a, str(cfg.features.get("dtype", "float32")))
                    st["imported"] += 1
                    row["status"] = "ok"
            except Exception as e:
                st["failed"] += 1
                row.update(status="failed", message=f"{type(e).__name__}: {str(e)[:200]}")
            status.append(row)
        per_enc[ename] = st

    # 跨编码器坐标一致性
    inconsistent = [sid for sid, d in coords_by_sid.items() if len(set(d.values())) > 1]
    run.write_table(status, "status.csv", ["encoder", "study_id", "status", "n_tiles", "feature_dim", "message"])
    if unmatched:
        write_csv(unmatched, run.private_dir / "unmatched_trident_files.csv")
    run.write_text(build_report(per_enc, mag, mag_info, with_slide, status, inconsistent, expected, min_tiles),
                   "trident_import_report.md")
    n_fail = sum(st["failed"] for st in per_enc.values())
    log.info("导入完成：%s", "；".join(f"{k} 导入 {v['imported']}、跳过 {v['skipped']}、失败 {v['failed']}"
                                    for k, v in per_enc.items()))
    n_unm = sum(st["unmatched"] for st in per_enc.values())
    if n_unm:
        log.warning("有 %d 个 TRIDENT 文件在清单中找不到对应切片，文件名列表在运行目录 private/ 下（不要外传）", n_unm)
    if inconsistent:
        log.warning("%d 例在不同编码器之间的切块坐标不一致：%s", len(inconsistent), ", ".join(inconsistent[:20]))
    log.info("报告：%s", run.path("trident_import_report.md"))
    return 3 if n_fail else 0


def build_report(per_enc, mag: Path, mag_info: dict, with_slide: set, status: List[dict], inconsistent: List[str],
                 expected: dict, min_tiles: int) -> str:
    L = ["# TRIDENT 特征导入报告", "",
         f"- 目录：{mag.name}（{mag_info['magnification']:g}x，切块 {mag_info['patch_px']} px，重叠 {mag_info['overlap_px']} px）",
         f"- 清单中有切片的病例：{len(with_slide)}", "",
         "| 编码器 | TRIDENT 名 | 文件数 | 对上清单 | 对不上 | 导入 | 跳过 | 失败 | 维数（预期） | 切块数 中位/最少/最多 | 切块<{} 的病例 |".format(min_tiles),
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ename, st in per_enc.items():
        t = st["tiles"]
        tiles = f"{int(np.median(t))}/{min(t)}/{max(t)}" if t else ""
        dims = ",".join(str(d) for d in sorted(st["dims"]))
        exp = expected.get(ename)
        L.append(f"| {ename} | {st['trident_name']} | {st['n_files']} | {st['matched']} | {st['unmatched']} | "
                 f"{st['imported']} | {st['skipped']} | {st['failed']} | {dims}（{exp or '?'}） | {tiles} | "
                 f"{len(st['few_tiles'])} |")
    L.append("")
    have = defaultdict(set)
    for r in status:
        if r.get("status") in ("ok", "skipped"):
            have[r["encoder"]].add(r["study_id"])
    L += ["## 清单中有切片、但缺少该编码器特征的病例", ""]
    any_missing = False
    for ename in per_enc:
        miss = sorted(with_slide - have[ename])
        if miss:
            any_missing = True
            L.append(f"- {ename}（{len(miss)} 例）：{', '.join(miss[:50])}{' ...' if len(miss) > 50 else ''}")
    if not any_missing:
        L.append("无。")
    L += ["", "## 不同编码器之间切块坐标不一致的病例", ""]
    L.append(", ".join(inconsistent[:100]) if inconsistent else "无（同一切片在各编码器中使用相同的切块）。")
    fails = [r for r in status if r.get("status") == "failed" or r.get("message")]
    if fails:
        L += ["", "## 失败或有提示的条目", "", "| 编码器 | 病例 | 状态 | 说明 |", "|---|---|---|---|"]
        for r in fails[:200]:
            L.append(f"| {r['encoder']} | {r['study_id']} | {r.get('status', '')} | {r.get('message', '')} |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = base_parser("导入 TRIDENT 已提取的病理特征（按 study_id 重命名，并做一致性检查）。", workers=False)
    p.add_argument("--mag-dir", default=None, help="TRIDENT 的倍率/切块目录名，如 20x_256px_0px_overlap（只有一个时可不填）")
    p.add_argument("--encoders", default=None, help="只导入这些 TRIDENT 编码器（逗号分隔，如 uni_v2,conch_v15）")
    return run_main(_run, p, argv)


if __name__ == "__main__":
    raise SystemExit(main())
