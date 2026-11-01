"""假 TotalSegmentator：命令行参数与真实工具一致（-i 输入 -o 输出目录 --roi_subset stomach ...）。

胃分割结果取自 ``--truth-dir/<输入文件名>``（假数据生成器写出的真值），用于测试封装流程。
用法（配置中）：v2.external.totalseg.command: [python, -m, gastric_mm.fake_tools.totalseg, --truth-dir, <目录>]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--truth-dir", required=True)
    p.add_argument("-i", required=True)
    p.add_argument("-o", required=True)
    p.add_argument("--roi_subset", nargs="*", default=None)
    p.add_argument("-d", "--device", default="gpu")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--fail", action="store_true", help="模拟运行失败")
    a = p.parse_args(argv)
    if a.fail:
        print(f"Traceback: failed reading {a.i}", file=sys.stderr)
        return 1
    src = Path(a.truth_dir) / Path(a.i).name
    out = Path(a.o)
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"no truth for {src.name}", file=sys.stderr)
        return 1
    shutil.copy(src, out / "stomach.nii.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
