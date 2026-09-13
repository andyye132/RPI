"""Offline analysis of saved axis episodes: the metric-decomposition data
(fig:decomp) and per-architecture aggregates. Pure numpy, no GPU.

Reads $AETHER_SCRATCH/runs/<world>_<regime>_<arch>_s<seed>.npz and
reports, per (world, regime, arch): final-third common-mode / relative pose error,
GOSPA total and its localization / missed / false components (mean over robots),
IDF1, delivered fraction. Theory alignment checks: architecture should move the
common mode and the GOSPA localization+missed split, not the false/clutter side.
Run: python -m coopsim.runners.analyze_axis
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim.metrics.gauge import gauge_decompose
from coopsim.metrics.gospa import gospa_series
from coopsim.metrics.identity import identity_metrics

from coopsim.paths import RUNS
RUNS = str(RUNS)

# mirrors interfaces.ARCHS (not imported: that would pull jax into this numpy-only tool)
ARCHS = ("A0", "A_BL", "A_BLS", "A_BLF", "A_HY", "A_HYG", "A_HYM", "A_J")


def analyze(path):
    d = np.load(path, allow_pickle=True)
    X_true, x_est = d["X_true"][:, :, :2], d["x_est"][:, :, :2]
    T = x_est.shape[0]
    f3 = slice(2 * T // 3, T)
    g = gauge_decompose(X_true, x_est)
    n = d["tracks"].shape[1]
    # GOSPA on the final third, subsampled every 10 frames (the metric is a mean,
    # so subsampling is an unbiased estimate at ~30x less Hungarian cost)
    ss = slice(2 * T // 3, T, 10)
    loc, missed, false, tot, idf = [], [], [], [], []
    for r in range(n):
        s = gospa_series(d["tau"][ss], d["alive"][ss], d["tracks"][ss, r, :, :2],
                         d["tracks_mask"][ss, r])
        loc.append(s["loc"].mean()); missed.append(s["missed"].mean())
        false.append(s["false"].mean()); tot.append(s["gospa"].mean())
        idm = identity_metrics(d["tau"][ss], d["alive"][ss],
                               d["tracks"][ss, r, :, :2],
                               d["tracks"][ss, r, :, 4].astype(int),
                               d["tracks_mask"][ss, r])
        idf.append(idm["idf1"])
    return dict(cm=g["common"][f3].mean(), rel=g["relative"][f3].mean(),
                gospa=np.mean(tot), loc=np.mean(loc), missed=np.mean(missed),
                false=np.mean(false), idf1=np.mean(idf),
                deliv=float(d["succ"].mean()))


def main():
    cells = defaultdict(list)
    for p in sorted(glob.glob(f"{RUNS}/*.npz")):
        name = Path(p).stem
        if "_s" not in name or not name.startswith(("urban", "terrain", "forest", "open")):
            continue                                          # skip non-episode npz
        stem = name.rsplit("_s", 1)[0]
        # longest name first: A_HYG must not be swallowed by A_HY, A_BLS/F by A_BL
        arch = next((a for a in sorted(ARCHS, key=len, reverse=True)
                     if stem.endswith(a)), None)
        if arch is None:
            continue
        world, regime = stem[: -len(arch) - 1].split("_", 1)
        seed = name.rsplit("_s", 1)[1]
        try:
            r = analyze(p)
            r["seed"] = seed
            cells[(world, regime, arch)].append(r)
        except Exception as e:
            print(f"SKIP {Path(p).name}: {e}", flush=True)

    hdr = f"{'cell':28s} {'cm':>6s} {'rel':>6s} {'gospa':>6s} {'loc':>6s} " \
          f"{'miss':>6s} {'false':>6s} {'idf1':>6s} {'deliv':>6s} n"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells):
        rows = cells[key]
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != "seed"}
        print(f"{'_'.join(key):28s} {m['cm']:6.2f} {m['rel']:6.2f} {m['gospa']:6.2f} "
              f"{m['loc']:6.2f} {m['missed']:6.2f} {m['false']:6.2f} "
              f"{m['idf1']:6.3f} {m['deliv']:6.3f} {len(rows)}")

    # PAIRED A_J vs A0 by seed under CRN (the correct statistic): report mean
    # paired difference, win fraction, and a sign test on common-mode + relative.
    print("\n== paired A_J vs A0 (matched by seed; negative = A_J better) ==")
    print(f"{'cell':12s} {'n':>3s}  {'d_cm':>7s} {'win%':>5s}  {'d_rel':>7s} "
          f"{'win%':>5s}  {'d_false':>8s} {'d_idf1':>7s}")
    for world, regime in sorted({(k[0], k[1]) for k in cells}):
        a0 = {r["seed"]: r for r in cells.get((world, regime, "A0"), [])}
        aj = {r["seed"]: r for r in cells.get((world, regime, "A_J"), [])}
        common = sorted(set(a0) & set(aj))
        if not common:
            continue
        dcm = np.array([aj[s]["cm"] - a0[s]["cm"] for s in common])
        drel = np.array([aj[s]["rel"] - a0[s]["rel"] for s in common])
        dfa = np.array([aj[s]["false"] - a0[s]["false"] for s in common])
        did = np.array([aj[s]["idf1"] - a0[s]["idf1"] for s in common])
        print(f"{world}_{regime:9s} {len(common):3d}  {dcm.mean():7.3f} "
              f"{100*(dcm<0).mean():4.0f}%  {drel.mean():7.3f} "
              f"{100*(drel<0).mean():4.0f}%  {dfa.mean():8.3f} {did.mean():7.3f}")
    print("DONE analyze_axis")


if __name__ == "__main__":
    main()
