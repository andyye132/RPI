"""Does the scalar interface actually deliver more USABLE relative measurements?

Delivery alone is not what reaches the task level: a relative factor forms only
when a link both delivers and is inside the relative-measurement range. If the
extra deliveries a delivery-optimizing scheduler wins are out of range, the task
level sees no additional measurements and the claim that the interface is
"not inert" would be wrong.
Run: python -u coopsim/runners/diag_relcount.py
"""
import glob
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from coopsim.config import load_config

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
    rng_max = load_config().relative.range
    print(f"relative-measurement range = {rng_max} m")
    a0 = seeds_of("A0")
    print(f"{'arch':7s} {'n':>3s} {'deliv/slot':>11s} {'in-range rel/slot':>18s}")
    for arch in ("A0", "A_BL", "A_BLS", "A_BLF"):
        var = seeds_of(arch)
        common = sorted(set(a0) & set(var), key=int)[:8]
        if not common:
            print(f"{arch:7s} none")
            continue
        dl, rl = [], []
        for s in common:
            d = np.load(var[s])
            succ, X = d["succ"], d["X_true"]
            T = min(succ.shape[0], X.shape[0])
            succ, X = succ[:T], X[:T, :, :2]
            dist = np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=-1)
            usable = succ & (dist <= rng_max)
            dl.append(succ.sum(axis=(1, 2)).mean())
            rl.append(usable.sum(axis=(1, 2)).mean())
        print(f"{arch:7s} {len(common):3d} {np.mean(dl):11.3f} {np.mean(rl):18.3f}")
    print("DONE diag_relcount")


if __name__ == "__main__":
    main()
