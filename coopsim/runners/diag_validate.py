"""Validate the paper's per-architecture error table against episode data on disk, and compute the
missing paired A-BL vs A0 statistics and wall-clock per slot (the tbd cells).
Seeds 0-5 were overwritten by later config tests, so recomputation uses plain-seed
episodes with seed >= 6 only; small differences from the published (pre-overwrite,
44-49 seed) cells are expected and reported.
Run: python -u coopsim/runners/diag_validate.py
"""
import glob
import os
import re

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from coopsim.metrics.gauge import gauge_decompose

from coopsim.paths import RUNS
RUNS = str(RUNS)


def load(arch, regime="R1"):
    out = {}
    for p in sorted(glob.glob(f"{RUNS}/urban_{regime}_{arch}_s*.npz")):
        tag = os.path.basename(p)[:-4].rsplit("_s", 1)[1]
        if not tag.isdigit() or int(tag) < 6:
            continue
        d = np.load(p)
        T = d["x_est"].shape[0]
        f3 = slice(2 * T // 3, T)
        g = gauge_decompose(d["X_true"][:, :, :2], d["x_est"][:, :, :2])
        w = float(d["wall_per_slot"]) if "wall_per_slot" in d else float("nan")
        out[tag] = (float(g["common"][f3].mean()),
                    float(g["relative"][f3].mean()), w)
    return out


def main():
    cells = {a: load(a)
             for a in ("A0", "A_BL", "A_BLS", "A_BLF", "A_HY", "A_HYG", "A_J")}
    print("== recomputed cells (clean seeds >= 6, urban R1, final-third) ==")
    print(f"{'arch':6s} {'n':>3s} {'cm':>6s} {'rel':>6s} {'wall ms/slot':>13s}")
    for a, d in cells.items():
        if not d:
            print(f"{a:6s}  no episodes")
            continue
        cm = np.mean([v[0] for v in d.values()])
        rel = np.mean([v[1] for v in d.values()])
        w = np.nanmean([v[2] for v in d.values()])
        print(f"{a:6s} {len(d):3d} {cm:6.2f} {rel:6.2f} {1e3 * w:13.1f}")

    print("\n== paired differences vs A0 ==")
    for arch in ("A_BL", "A_BLS", "A_BLF", "A_HY", "A_HYG", "A_J"):
        common = sorted(set(cells["A0"]) & set(cells[arch]), key=int)
        if not common:
            print(f"{arch:6s}  no paired episodes")
            continue
        dcm = np.array([cells[arch][s][0] - cells["A0"][s][0] for s in common])
        drel = np.array([cells[arch][s][1] - cells["A0"][s][1] for s in common])
        print(f"{arch:6s} n={len(common):3d}  d_cm={dcm.mean():+.4f} "
              f"sd={dcm.std(ddof=1):.4f}  d_rel={drel.mean():+.4f} "
              f"sd={drel.std(ddof=1):.4f}")
    print("DONE diag_validate")


if __name__ == "__main__":
    main()
