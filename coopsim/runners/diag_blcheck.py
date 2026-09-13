"""Non-vacuity check for the A-BL variants: does each variant actually change
what the task estimator consumes relative to A0? A gate that never fires or a
schedule identical to the baseline would reproduce A0 trivially rather than by
Proposition 1, which would make the reported coincidence vacuous.

Compares, per seed, the scheduled-link pattern and the delivered-packet count
against A0 on the same seed under common random numbers.
Run: python -u coopsim/runners/diag_blcheck.py
"""
import glob
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from coopsim.paths import RUNS
RUNS = str(RUNS)


def seeds_of(arch):
    out = {}
    for p in sorted(glob.glob(f"{RUNS}/urban_R1_{arch}_s*.npz")):
        tag = os.path.basename(p)[:-4].rsplit("_s", 1)[1]
        if tag.isdigit() and int(tag) >= 6:
            out[tag] = p
    return out


def main():
    a0 = seeds_of("A0")
    for arch in ("A_BL", "A_BLS", "A_BLF"):
        var = seeds_of(arch)
        common = sorted(set(a0) & set(var), key=int)[:8]
        if not common:
            print(f"{arch}: no paired episodes yet")
            continue
        sched_diff, deliv_a0, deliv_v = [], [], []
        for s in common:
            d0, dv = np.load(a0[s]), np.load(var[s])
            n = min(d0["sched"].shape[0], dv["sched"].shape[0])
            s0, sv = d0["sched"][:n] > 0, dv["sched"][:n] > 0
            sched_diff.append((s0 != sv).mean())
            deliv_a0.append(d0["succ"][:n].sum(axis=(1, 2)).mean())
            deliv_v.append(dv["succ"][:n].sum(axis=(1, 2)).mean())
        print(f"{arch}: n={len(common)}  scheduled-link disagreement vs A0 = "
              f"{100 * np.mean(sched_diff):.1f}%  "
              f"delivered/slot {np.mean(deliv_a0):.2f} -> {np.mean(deliv_v):.2f}")
    print("DONE diag_blcheck")


if __name__ == "__main__":
    main()
