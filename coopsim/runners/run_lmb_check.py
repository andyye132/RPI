"""LMB-lite back end against the jitted GM-PHD on the B5 canned scenario.

Both filters consume the identical detection stream (run_targets_check's
make_scenario). PASS = LMB GOSPA series mean within 2x of the GM-PHD mean and
LMB expected cardinality (sum of existence) tracks the true target count.
Run: python -u -m coopsim.runners.run_lmb_check
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim import targets_lmb
from coopsim.config import load_config
from coopsim.metrics.gospa import gospa
from coopsim.runners.run_targets_check import DT, make_scenario, run_jax


def run_lmb(det_stream):
    lmb = targets_lmb.LMB(robot_id=0)
    est_out, card = [], []
    for z, R, p_d, kappa in det_stream:
        lmb.predict(DT, 0.3, 0.995)
        lmb.update(z, R, np.ones(len(z), bool), p_d, kappa)
        est, emask = lmb.extract(16)
        est_out.append(est[emask][:, :2].reshape(-1, 2))
        card.append(sum(t["r"] for t in lmb.tracks))
    return est_out, card


def main():
    cfg = load_config()
    rng = np.random.default_rng(0)
    tau, alive, dets = make_scenario(rng)

    est_g, _ = run_jax(dets, cfg)
    est_l, card = run_lmb(dets)

    g_g = np.array([gospa(tau[t][alive[t]], est_g[t])["gospa"]
                    for t in range(len(dets))])
    g_l = np.array([gospa(tau[t][alive[t]], est_l[t])["gospa"]
                    for t in range(len(dets))])
    settle = slice(30, None)
    mg, ml = g_g[settle].mean(), g_l[settle].mean()
    print(f"GOSPA mean: gmphd={mg:.3f} lmb={ml:.3f} ratio={ml / mg:.2f}")
    ok1 = ml <= 2.0 * mg
    print(f"{'PASS' if ok1 else 'FAIL'} lmb.gospa_within_2x")

    card_err = np.abs(np.array(card)[settle] - alive.sum(1)[settle]).mean()
    print(f"cardinality mean abs err = {card_err:.2f}")
    ok2 = card_err < 1.0
    print(f"{'PASS' if ok2 else 'FAIL'} lmb.cardinality")

    print("ALL PASS" if (ok1 and ok2) else "FAILURES PRESENT")
    print("DONE lmb_check")


if __name__ == "__main__":
    main()
