"""Proposition 3's degenerate geometries, measured.

Runs the joint estimator with the field SUPPLIED on three fields whose stabilizer
subalgebra is nontrivial by construction, and reports the drift in each gauge mode
separately. The prediction is that the mode in the stabilizer drifts while the
others stay bounded: constant fixes all of SE(2), stratified along y leaves
translation in x free, radial about the centre leaves rotation free. Urban, whose
stabilizer is trivial, is the control.

  DEG_SEEDS=4 DEG_STEPS=3000 python -u -m coopsim.runners.run_degenerate
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim import harness
from coopsim.config import load_config
from coopsim.metrics.gauge import gauge_modes

KINDS = ("urban", "constant", "stratified", "radial")
PREDICT = {"urban": "none", "constant": "tx, ty, rot", "stratified": "tx",
           "radial": "rot"}   # modes whose stabilizer membership kills the benefit


def main():
    """Proposition 3 predicts that channel factors contribute NO information along a
    stabilizer mode. That is a statement about the channel's contribution, not about
    total error, which is dominated by the shared odometry random walk. So we measure
    the contribution directly, within each field, as the gauge error with channel
    factors (A_J) subtracted from the error without them (A0), per mode, CRN-paired.

    Pre-registered: the benefit should be positive in every mode for urban (trivial
    stabilizer), absent in tx for stratified, absent in rot for radial, and absent
    everywhere for constant."""
    seeds = int(os.environ.get("DEG_SEEDS", 12))
    steps = int(os.environ.get("DEG_STEPS", 3000))
    print(f"degenerate geometries: {seeds} seeds x {steps} steps, supplied field, "
          f"A0 vs A_J within each field", flush=True)
    per = {}
    for kind in KINDS:
        cfg = load_config(overrides=[f"world.kind={kind}",
                                     "estimator.oracle_field=true"])
        for s in range(seeds):
            for arch in ("A0", "A_J"):
                _, out = harness.run_episode(cfg, arch, s, steps=steps, out_dir=None)
                g = gauge_modes(out["X_true"], out["x_est"])
                f3 = slice(2 * len(g["tx"]) // 3, None)
                for m in ("tx", "ty", "rot"):
                    per[(kind, s, arch, m)] = float(np.mean(g[m][f3]))
            print(f"  {kind} seed {s}: "
                  + " ".join(f"{m}={per[(kind,s,'A0',m)]:.2f}->"
                             f"{per[(kind,s,'A_J',m)]:.2f}"
                             for m in ("tx", "ty", "rot")), flush=True)

    print("\n== channel-factor benefit, A0 minus A_J, paired over seeds ==",
          flush=True)
    print(f"{'field':<12}{'tx':>18}{'ty':>18}{'rot':>18}   no benefit predicted")
    for kind in KINDS:
        cells = []
        for m in ("tx", "ty", "rot"):
            d = np.array([per[(kind, s, "A0", m)] - per[(kind, s, "A_J", m)]
                          for s in range(seeds)])
            tv = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() else 0.0
            cells.append(f"{d.mean():+7.2f} (t={tv:+5.2f})")
        print(f"{kind:<12}" + "".join(f"{c:>18}" for c in cells)
              + f"   {PREDICT[kind]}", flush=True)
    print("DONE run_degenerate", flush=True)


if __name__ == "__main__":
    main()
