"""假 nnUNetv2_predict：参数与真实命令一致（-i -o -d -c -f ...）。

对每个输入 ``<case>_0000.nii.gz``，把 CT 值在 [lo, hi] 内的体素作为"肿瘤"写出 ``<case>.nii.gz``。
假数据体模中肿瘤约 90 HU、胃壁约 55 HU、加噪声，因此结果含真实肿瘤和少量散在噪声，可检验连通域选择。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    import numpy as np
    import SimpleITK as sitk

    p = argparse.ArgumentParser()
    p.add_argument("-i", required=True)
    p.add_argument("-o", required=True)
    p.add_argument("-d", default="501")
    p.add_argument("-c", default="3d_fullres")
    p.add_argument("-f", nargs="*", default=["0"])
    p.add_argument("-tr", default="nnUNetTrainer")
    p.add_argument("-device", default="cuda")
    p.add_argument("--lo", type=float, default=75.0)
    p.add_argument("--hi", type=float, default=150.0)
    a, _ = p.parse_known_args(argv)
    out = Path(a.o)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.i).glob("*_0000.nii.gz")):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        m = ((arr >= a.lo) & (arr <= a.hi)).astype(np.uint8)
        o = sitk.GetImageFromArray(m)
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out / (f.name[: -len("_0000.nii.gz")] + ".nii.gz")), True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
