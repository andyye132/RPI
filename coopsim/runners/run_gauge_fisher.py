"""Gauge-Fisher measurement (fig:interface): accumulated task-level Fisher
information along the scene gauge modes, per architecture, on one replayed episode.

Per slot, project each architecture's task-level measurement information onto the
three normalized gauge generators (tx, ty, rotation about the centroid):
  A0 / A-BL: relative factors only (scalar reweighting cannot change a zero
             projection, so one curve serves both). Prediction: identically zero.
  A-J oracle field: full channel-factor pose information.
  A-J / A-HY estimated field: the Schur pose message with the inner's accumulated
             field information (the two coincide up to linearization).
  joint marginal: the same channel factors with the field eliminated ONCE against the
             total field information, sum_t G Axx G^T - (sum_t G Axf)(sum_t Aff +
             prior)^-1 (sum_t G Axf)^T. This is what the per-slot sum is supposed to
             approximate; the gap between them is the Remark-3 double counting.
Run: python -u -m coopsim.runners.run_gauge_fisher
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import channel, estimator, factors, world
from coopsim.config import crn_keys, load_config
from coopsim.runners.run_m1_plateau import make_sched

STEPS = int(os.environ.get("GF_STEPS", "2000"))


def normalized_generators(X_slot):
    G = factors.gauge_generators(X_slot[None], jnp.ones(1, bool))   # (3, 3N)
    return G / jnp.linalg.norm(G, axis=1, keepdims=True)


def rel_slot_axx(X_slot, rel_mask, w_rel):
    e, Ji, Jj = factors.relpose_factors(X_slot[None], jnp.zeros((1,) + rel_mask.shape + (3,)))
    N = X_slot.shape[0]
    ii = jnp.arange(N)[:, None].repeat(N, 1).reshape(-1)
    jj = jnp.arange(N)[None, :].repeat(N, 0).reshape(-1)
    Wm = w_rel[None, :] * rel_mask.reshape(-1)[:, None]
    Axx = jnp.zeros((N, N, 3, 3))
    JiW, JjW = Ji * Wm[:, :, None], Jj * Wm[:, :, None]
    Axx = Axx.at[ii, ii].add(jnp.einsum("mki,mkj->mij", JiW, Ji))
    Axx = Axx.at[jj, jj].add(jnp.einsum("mki,mkj->mij", JjW, Jj))
    Hij = jnp.einsum("mki,mkj->mij", JiW, Jj)
    Axx = Axx.at[ii, jj].add(Hij)
    Axx = Axx.at[jj, ii].add(Hij.transpose(0, 2, 1))
    return Axx.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def main():
    cfg = load_config()
    cfg.world.kind = "urban"
    cfg.world.trajectory = "transect"
    cfg.time.steps = STEPS
    cfg.channel.ge_p_bad_los = 0.0
    cfg.channel.ge_p_bad_nlos = 0.0
    n = cfg.shapes.n_robots
    ext, dt = cfg.world.extent, cfg.time.dt
    keys = crn_keys("urban", "GF", 0)
    wld = world.make_world(keys.world_gen, cfg)
    X_true, _ = world.make_trajectories(keys.target_gen, cfg)
    c = estimator.make_consts(cfg)
    F_true = jnp.asarray(wld.F_true)
    lut = channel.nakagami_lut(cfg.channel.nakagami_m.urban)
    ch_state = channel.init_state(keys.fading, n)
    ch_step = jax.jit(lambda t, pos, vel, s, tx: channel.step(
        t, keys, pos, vel, F_true, s, tx, lut, cfg.channel, ext, dt))
    sched = make_sched(n, int(cfg.channel.budget_frac * n), cfg.channel.conflict_radius)
    age = jnp.zeros(n * n)

    G2 = F_true.size
    Lam_F = jnp.zeros((G2, G2))
    M_rel = np.zeros((3, 3)); M_or = np.zeros((3, 3)); M_est = np.zeros((3, 3))
    Bxf = np.zeros((3, G2))     # sum_t G_t Axf_t, for the eliminate-once marginal

    blocks = jax.jit(lambda Xs, d: estimator._slot_channel_blocks(Xs, F_true, d, c))

    for t in range(1, STEPS):
        pos = X_true[t, :, :2]
        vel = (X_true[t, :, :2] - X_true[t - 1, :, :2]) / dt
        tx, age = sched(pos, age)
        obs, ch_state = ch_step(t, jnp.asarray(pos), jnp.asarray(vel), ch_state, tx)
        X_slot = jnp.asarray(X_true[t])
        Gg = np.asarray(normalized_generators(X_slot))

        d_ij = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        rel_mask = jnp.asarray(np.asarray(obs.success) & (d_ij <= cfg.relative.range))
        Axx_r = np.asarray(rel_slot_axx(X_slot, rel_mask, c.w_rel))
        M_rel += Gg @ Axx_r @ Gg.T

        zero = jnp.zeros((n, n))
        d = estimator.plain_slot(jnp.zeros((n, 3)), jnp.zeros((n, n, 3)),
                                 jnp.zeros((n, n), bool), obs.rssi_q, obs.success,
                                 obs.attempted, obs.success, zero, zero)
        Axx, Axf, Aff, _, _ = blocks(X_slot, d)
        M_or += Gg @ np.asarray(Axx) @ Gg.T

        Ff = F_true.reshape(-1)
        Aff_tot = Aff + jnp.eye(G2) / c.f_prior_var + Lam_F + 1e-6 * jnp.eye(G2)
        Y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Aff_tot), Axf.T)
        msg = np.asarray(Axx - Axf @ Y)
        M_est += Gg @ (0.5 * (msg + msg.T)) @ Gg.T
        Lam_F = Lam_F + Aff
        Bxf += Gg @ np.asarray(Axf)

        if t % 200 == 0 or t == STEPS - 1:
            Bff = np.asarray(Lam_F) + (1.0 / c.f_prior_var + 1e-6) * np.eye(G2)
            M_jt = M_or - Bxf @ np.linalg.solve(Bff, Bxf.T)

            def s(M):
                ev = np.linalg.eigvalsh(0.5 * (M + M.T))
                return f"tr={np.trace(M):9.2f} eigmin={ev[0]:9.4f}"
            print(f"t={t:5d}  rel[{s(M_rel)}]  oracle[{s(M_or)}]  est[{s(M_est)}]"
                  f"  joint[{s(M_jt)}]", flush=True)

    print("DONE gauge_fisher")


if __name__ == "__main__":
    main()
