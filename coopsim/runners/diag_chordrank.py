"""Finite-N identifiability of the chord-to-voxel operator (Assumption 3, finite form).

Assumption 3 asks for Fresnel weights dense in the test functions, which finitely many
chords never satisfy. This measures the finite-N shortfall directly: it accumulates the
line-integral operator A over an episode's geometry and reports how many voxel
directions the chords actually inform, against the 1600 the field carries. The N=8 and
N=16 regimes of Sec. VI should separate here if identifiability is what distinguishes
them.

  CR_N=8,16 CR_STEPS=3000 CR_EVERY=10 python -u -m coopsim.runners.diag_chordrank
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import world
from coopsim.config import load_config, crn_keys
from coopsim.field import line_integral


def operator_spectrum(cfg, steps, every):
    """Eigenvalues of A^T A for the episode's realized chords. A's rows are the
    per-chord voxel weights, obtained as the gradient of the (linear) line integral."""
    G = cfg.shapes.grid
    ext = cfg.world.extent
    keys = crn_keys(cfg.world.kind, "R1", 0)
    world.make_world(keys.world_gen, cfg)
    X, _ = world.make_trajectories(keys.target_gen, cfg)
    F0 = jnp.zeros((G, G))
    row = jax.jit(jax.grad(lambda F, xi, xj: line_integral(xi, xj, F, ext)))
    n = cfg.shapes.n_robots
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    M = np.zeros((G * G, G * G), np.float64)
    for t in range(0, min(steps, len(X)), every):
        P = jnp.asarray(X[t][:, :2])
        R = np.stack([np.asarray(row(F0, P[i], P[j])).ravel() for i, j in pairs])
        M += R.T @ R
    return np.linalg.eigvalsh(M)[::-1], len(pairs)


def main():
    Ns = [int(x) for x in os.environ.get("CR_N", "8,16").split(",")]
    steps = int(os.environ.get("CR_STEPS", 3000))
    every = int(os.environ.get("CR_EVERY", 10))
    print(f"chord-to-voxel identifiability: N={Ns}, {steps} steps every {every}",
          flush=True)
    print(f"{'N':>4}{'chords':>9}{'voxels':>9}{'rank':>8}{'eff rank':>10}"
          f"{'informed':>10}{'cond':>12}", flush=True)
    for N in Ns:
        cfg = load_config(overrides=[f"shapes.n_robots={N}"])
        ev, npair = operator_spectrum(cfg, steps, every)
        nvox = cfg.shapes.grid ** 2
        tol = ev[0] * 1e-10
        rank = int((ev > tol).sum())
        eff = float(ev.sum() ** 2 / (ev ** 2).sum())
        # directions the chords inform better than the voxel prior does
        informed = int((ev > 1.0 / cfg.field.prior_sigma_db ** 2).sum())
        pos = ev[ev > tol]
        cond = float(pos[0] / pos[-1]) if len(pos) else float("inf")
        print(f"{N:>4}{npair:>9}{nvox:>9}{rank:>8}{eff:>10.1f}{informed:>10}"
              f"{cond:>12.3g}", flush=True)
    print("DONE diag_chordrank", flush=True)


if __name__ == "__main__":
    main()
