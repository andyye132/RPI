"""A2: the common-mode drift rate behind the paper's deployable claim.

Fits each seed's common mode against sqrt(t) and reports the rate. Two fits, and
the distinction matters elsewhere in this project:

  WITH INTERCEPT    e = a + b sqrt(t).  `b` is a growth rate.
  ZERO INTERCEPT    e = k sqrt(t).      `k` is a LEVEL readout -- a constant
                    signal at level L returns k = L sum(sqrt t)/sum(t), so a flat
                    signal reproduces a plausible-looking "rate".

STORE: coopsim.paths defaults to $SCRATCH/netcomm_v3, which lost most of its
episodes on 2026-08-17. Reading it silently yields a smaller n and different
numbers, so set AETHER_SCRATCH or DRIFT_RUNS to the regenerated store. Check the
per-arm n in the output against the store you intended.

Same window convention as diag_dichotomy (last 75% of the episode) and the two
share `growth`/`rates`, so the conventions cannot drift apart.

Run: srun -p batch -t 30 --mem=32G python -u coopsim/runners/diag_drift.py
Env: DRIFT_RUNS (episode store), DRIFT_ARMS ("A_J,A0"), DRIFT_SEEDS (6),
     DRIFT_EXPECTED_N (optional required paired sample size).
Seeds 0--5 are tuning seeds and are excluded by default. Set DRIFT_SEEDS=0
only to reconstruct historical summaries that included them.
"""
import glob
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from coopsim.metrics.gauge import gauge_decompose
from coopsim.paths import RUNS
from coopsim.runners.diag_dichotomy import growth, rates

STORE = os.environ.get("DRIFT_RUNS", str(RUNS))
ARMS = os.environ.get("DRIFT_ARMS", "A_J,A0").split(",")
SEED_LO = int(os.environ.get("DRIFT_SEEDS", "6"))
EXPECTED_N = int(os.environ.get("DRIFT_EXPECTED_N", "0"))
DT = 0.1


def episodes(arch):
    out = {}
    for p in sorted(glob.glob(f"{STORE}/urban_R1_{arch}_s*.npz")):
        tag = os.path.basename(p)[:-4].rsplit("_s", 1)[1]
        if tag.isdigit() and int(tag) >= SEED_LO:
            out[int(tag)] = p
    return out


def common_mode(path):
    d = np.load(path, allow_pickle=True)
    Xt, Xe = d["X_true"], d["x_est"]
    T = min(len(Xt), len(Xe))
    cm = gauge_decompose(Xt[:T, :, :2], Xe[:T, :, :2])["common"]
    return cm, np.arange(T) * DT


def ci95(v):
    """Mean and half-width of a 95% t-interval across seeds."""
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        raise ValueError("At least two seeds are required for a confidence interval")
    se = v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(student_t.ppf(0.975, n - 1) * se)


def main():
    print(f"A2 common-mode drift rate\n  store {STORE}\n  arms {ARMS}"
          f"  seeds >= {SEED_LO}\n")
    selected = {arch: episodes(arch) for arch in ARMS}
    for arch, eps in selected.items():
        if len(eps) < 2:
            raise ValueError(f"{arch}: fewer than two episodes in {STORE}")
        if EXPECTED_N and len(eps) != EXPECTED_N:
            raise ValueError(f"{arch}: expected {EXPECTED_N} seeds, found {len(eps)}")
    if len(ARMS) == 2 and set(selected[ARMS[0]]) != set(selected[ARMS[1]]):
        raise ValueError("Paired arms have different seed sets; select a matched store")
    per = {}
    for arch in ARMS:
        eps = selected[arch]
        rows = {}
        for s, p in eps.items():
            cm, t_s = common_mode(p)
            b, level, frac = growth(cm, t_s)
            k, final3 = rates(cm, t_s)
            rows[s] = dict(b=b, k=k, level=level, frac=frac, final3=final3)
        per[arch] = rows
        b = [r["b"] for r in rows.values()]
        k = [r["k"] for r in rows.values()]
        f3 = [r["final3"] for r in rows.values()]
        fr = [r["frac"] for r in rows.values()]
        mb, hb = ci95(b)
        mk, hk = ci95(k)
        mf, hf = ci95(f3)
        print(f"  {arch:5s} n={len(rows):3d}")
        print(f"    seeds                 {sorted(rows)}")
        print(f"    rate WITH intercept   {mb:+.4f}  [{mb-hb:+.4f}, {mb+hb:+.4f}]"
              f"   m s^-1/2      <- the growth rate")
        print(f"    rate ZERO intercept   {mk:+.4f}  [{mk-hk:+.4f}, {mk+hk:+.4f}]"
              f"   m s^-1/2      <- LEVEL readout, not a rate")
        print(f"    final-third mean      {mf:.3f} m; seed SD {np.std(f3, ddof=1):.3f};"
              f" 95% CI [{mf-hf:.3f}, {mf+hf:.3f}]")
        print(f"    realized/implied rise {np.mean(fr):.2f}"
              f"   (near 0 => flat, the zero-intercept fit is reading the level)")

    if len(per) == 2 and all(per.values()):
        a, b_ = ARMS[0], ARMS[1]
        common = sorted(set(per[a]) & set(per[b_]))
        if common:
            d = np.array([per[a][s]["b"] - per[b_][s]["b"] for s in common])
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"\n  PAIRED {a} - {b_} on the intercept-fitted rate, "
                  f"n={len(common)}")
            tval = d.mean() / se if se else float("nan")
            md, hd = ci95(d)
            print(f"    difference {md:+.4f}; seed SD {d.std(ddof=1):.4f};"
                  f" SE {se:.4f}; 95% CI [{md-hd:+.4f}, {md+hd:+.4f}];"
                  f" t({len(d)-1}) = {tval:+.2f}")

    print("\nDONE diag_drift")

if __name__ == "__main__":
    main()
