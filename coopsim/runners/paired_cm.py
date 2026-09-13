"""Fast paired A_J-vs-A0 common-mode / relative test (no GOSPA), matched by seed
under common random numbers. This is the decisive statistic for whether the
estimated-field common-mode advantage is real or seed noise.
Run: python -u -m coopsim.runners.paired_cm
"""
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim.metrics.gauge import gauge_decompose

from coopsim.paths import RUNS
RUNS = str(RUNS)

# mirrors interfaces.ARCHS (not imported: that would pull jax into this numpy-only tool)
ARCHS = ("A0", "A_BL", "A_BLS", "A_BLF", "A_HY", "A_HYG", "A_HYM", "A_J")
ARCH_A = os.environ.get("PAIR_A", "A0")     # reference arm
ARCH_B = os.environ.get("PAIR_B", "A_J")    # arm under test


def load(p):
    d = np.load(p, allow_pickle=True)
    T = d["x_est"].shape[0]
    f3 = slice(2 * T // 3, T)
    g = gauge_decompose(d["X_true"][:, :, :2], d["x_est"][:, :, :2])
    return float(g["common"][f3].mean()), float(g["relative"][f3].mean())


def main():
    cells = defaultdict(dict)                                  # (world,regime,arch)->{seed:(cm,rel)}
    for p in sorted(glob.glob(f"{RUNS}/*.npz")):
        name = Path(p).stem
        if "_s" not in name or not name.startswith(("urban", "terrain", "forest", "open")):
            continue
        stem, seed = name.rsplit("_s", 1)
        # longest name first: A_HYG must not be swallowed by A_HY, A_BLS/F by A_BL
        arch = next((a for a in sorted(ARCHS, key=len, reverse=True)
                     if stem.endswith(a)), None)
        if arch is None or not seed.isdigit():
            continue
        world, regime = stem[: -len(arch) - 1].split("_", 1)
        try:
            cells[(world, regime, arch)][seed] = load(p)
        except Exception as e:
            print(f"SKIP {Path(p).name}: {e}", flush=True)

    print(f"{'cell':16s} {'n':>3s} {ARCH_A+'_cm':>6s} {ARCH_B+'_cm':>6s} {'d_cm':>7s} "
          f"{'cm_win':>6s}  {ARCH_A+'_rel':>6s} {ARCH_B+'_rel':>6s} {'d_rel':>7s} "
          f"{'rel_win':>7s}")
    for world, regime in sorted({(k[0], k[1]) for k in cells}):
        a0 = cells.get((world, regime, ARCH_A), {})
        aj = cells.get((world, regime, ARCH_B), {})
        common = sorted(set(a0) & set(aj), key=int)
        if not common:
            continue
        c0 = np.array([a0[s][0] for s in common]); cj = np.array([aj[s][0] for s in common])
        r0 = np.array([a0[s][1] for s in common]); rj = np.array([aj[s][1] for s in common])
        dcm, drel = cj - c0, rj - r0
        print(f"{world}_{regime:10s} {len(common):3d} {c0.mean():6.2f} {cj.mean():6.2f} "
              f"{dcm.mean():7.3f} {100*(dcm<0).mean():5.0f}%  "
              f"{r0.mean():6.2f} {rj.mean():6.2f} {drel.mean():7.3f} "
              f"{100*(drel<0).mean():6.0f}%")
    print("DONE paired_cm")


if __name__ == "__main__":
    main()
