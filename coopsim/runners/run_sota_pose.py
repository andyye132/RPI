"""SOTA pose-estimator rows (Table tab:l1 pose block): invariant-EKF and GBP,
standalone (drift predicted by Theorem 1) and hybridized with AETHER channel
factors at the oracle field (plateau predicted; estimator-independence of the
bound). Same CRN truth, schedule, and channel realizations as run_m1_plateau.
Env: SOTA_STEPS, SOTA_SEEDS, SOTA_ARMS (inekf_rel,inekf_chan,gbp_rel,gbp_chan).
Run: python -u -m coopsim.runners.run_sota_pose
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import channel, estimator, factors, se2, sota
from coopsim.config import load_config
from coopsim.field import emission
from coopsim.runners.run_m1_plateau import gen_truth, make_sched

STEPS = int(os.environ.get("SOTA_STEPS", "3000"))
SEEDS = int(os.environ.get("SOTA_SEEDS", "4"))
ARMS = os.environ.get("SOTA_ARMS", "inekf_rel,inekf_chan,gbp_rel,gbp_chan").split(",")


def channel_scalars(c, X, i, j, F, rssi, success):
    """(z, var) pairs for the hybridized channel updates at the current estimate."""
    pos = X[:, :2]
    pr = float(emission(pos[i], pos[j], F, c.extent, P0=c.p0, gamma=c.gamma)) \
        + c.fade_mean
    out = []
    if success:
        shift, jfac = factors.truncation_correction(jnp.asarray(pr), c.theta, c.sigma_f)
        if float(jfac) > 0.05:
            out.append(("rssi", float(rssi) - float(shift), c.rssi_var0))
    e_ob, w_ob = factors.probit_pseudo(jnp.asarray(pr), jnp.asarray(success),
                                       c.theta, c.sigma_f)
    out.append(("bit", pr - float(e_ob), 1.0 / max(float(w_ob), 1e-8)))
    return out


def run_arm(arm, cfg, keys, wld, X_true, U_meas, rng):
    n = cfg.shapes.n_robots
    ext, dt = cfg.world.extent, cfg.time.dt
    c = estimator.make_consts(cfg)
    F_true = jnp.asarray(wld.F_true)
    lut = channel.nakagami_lut(cfg.channel.nakagami_m.urban)
    ch_state = channel.init_state(keys.fading, n)
    ch_step = jax.jit(lambda t, pos, vel, s, tx: channel.step(
        t, keys, pos, vel, F_true, s, tx, lut, cfg.channel, ext, dt))
    sched = make_sched(n, int(cfg.channel.budget_frac * n),
                       cfg.channel.conflict_radius)
    age = jnp.zeros(n * n)

    sr_o = np.deg2rad(cfg.robots.sigma_rot_deg)
    q_diag = jnp.array([cfg.robots.sigma_trans ** 2, cfg.robots.sigma_trans ** 2,
                        sr_o ** 2])
    sr_r = np.deg2rad(cfg.relative.sigma_rot_deg)
    rel_sig = np.array([cfg.relative.sigma_trans, cfg.relative.sigma_trans, sr_r])
    r_diag = jnp.asarray(rel_sig ** 2)
    use_chan = arm.endswith("_chan")
    is_ekf = arm.startswith("inekf")

    s = sota.EKF(X=jnp.asarray(X_true[0]),
                 P=jnp.eye(3 * n) * 0.25)
    predict = jax.jit(lambda st, U: sota.ekf_predict(st, U, q_diag))
    upd_rel = jax.jit(lambda st, i, j, z: sota.ekf_update_rel(st, i, j, z, r_diag),
                      static_argnums=(1, 2))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def upd_chan(i, j):
        h = lambda Xp: emission(Xp[i, :2], Xp[j, :2], F_true, c.extent,
                                P0=c.p0, gamma=c.gamma) + c.fade_mean
        return jax.jit(lambda st, z, var: sota.ekf_update_scalar(st, h, z, var))

    @lru_cache(maxsize=None)
    def chan_jac(i, j):
        def rfun(Xp, xi):
            Xi = se2.retract(Xp[i], xi[:3])
            Xj = se2.retract(Xp[j], xi[3:])
            return emission(Xi[:2], Xj[:2], F_true, c.extent,
                            P0=c.p0, gamma=c.gamma) + c.fade_mean
        return jax.jit(lambda Xp: (rfun(Xp, jnp.zeros(6)),
                                   jax.jacfwd(lambda x: rfun(Xp, x))(jnp.zeros(6))))
    marks = {}
    t0 = time.time()
    for t in range(1, STEPS):
        pos = X_true[t, :, :2]
        vel = (X_true[t, :, :2] - X_true[t - 1, :, :2]) / dt
        tx, age = sched(pos, age)
        obs, ch_state = ch_step(t, jnp.asarray(pos), jnp.asarray(vel), ch_state, tx)
        succ = np.asarray(obs.success)
        rssi = np.asarray(obs.rssi_q)
        att = np.asarray(obs.attempted)
        d_ij = np.linalg.norm(pos[:, None] - pos[None], axis=-1)

        rels = []
        for i, j in zip(*np.nonzero(succ)):
            if d_ij[i, j] <= cfg.relative.range:
                z = np.asarray(se2.between(jnp.asarray(X_true[t, i]),
                                           jnp.asarray(X_true[t, j])))
                noise = rng.normal(0, 1, 3) * rel_sig
                if rng.random() < cfg.relative.outlier_rate:
                    noise *= cfg.relative.outlier_mag
                rels.append((int(i), int(j), jnp.asarray(z + noise)))

        if is_ekf:
            s = predict(s, jnp.asarray(U_meas[t - 1]))
            for i, j, z in rels:
                s = upd_rel(s, i, j, z)
            if use_chan:
                for i, j in zip(*np.nonzero(att)):
                    for kind, z, var in channel_scalars(
                            c, np.asarray(s.X), i, j, F_true,
                            rssi[i, j], bool(succ[i, j])):
                        s = upd_chan(int(i), int(j))(s, z, var)
        else:
            # GBP: propagate marginals, message-pass over this slot's factors
            Fs, Gs = jax.vmap(sota._prop_jacs)(s.X, jnp.asarray(U_meas[t - 1]))
            X_prop = jax.vmap(se2.compose)(s.X, jnp.asarray(U_meas[t - 1]))
            P_marg = s.P.reshape(n, 3, n, 3)
            P_prop = jnp.stack([Fs[i] @ P_marg[i, :, i, :] @ Fs[i].T
                                + Gs[i] @ jnp.diag(q_diag) @ Gs[i].T
                                for i in range(n)])
            prior_lam = jnp.stack([jnp.linalg.inv(P_prop[i] + 1e-9 * jnp.eye(3))
                                   for i in range(n)])
            prior_eta = jnp.zeros((n, 3))
            pfs = []
            for i, j, z in rels:
                r0, H6 = sota._rel_jac(X_prop, i, j, z)
                mah = float(r0 @ jnp.linalg.solve(jnp.diag(r_diag), r0))
                if mah > 30.0:
                    continue
                W = jnp.diag(1.0 / r_diag)
                pfs.append(sota.GBPFactor(i=i, j=j, Lam=H6.T @ W @ H6,
                                          eta=-(H6.T @ W @ r0)))
            if use_chan:
                for i, j in zip(*np.nonzero(att)):
                    for kind, z, var in channel_scalars(
                            c, np.asarray(X_prop), i, j, F_true,
                            rssi[i, j], bool(succ[i, j])):
                        pr0, H6v = chan_jac(int(i), int(j))(X_prop)
                        r0 = float(pr0) - z
                        H6 = np.asarray(H6v)[None, :]
                        # bounded influence, matching the joint estimator's guards
                        w = 1.0 / (var * max(1.0, abs(r0) / (2.0 * np.sqrt(var))) ** 2)
                        pfs.append(sota.GBPFactor(
                            i=int(i), j=int(j),
                            Lam=jnp.asarray(w * H6.T @ H6),
                            eta=jnp.asarray(-(w * H6.T * r0)[:, 0])))
            deltas, lams = sota.gbp_solve(prior_lam, prior_eta, pfs, n, iters=15)
            X_new = jax.vmap(se2.retract)(X_prop, deltas)
            P_new = jnp.zeros((3 * n, 3 * n))
            for i in range(n):
                P_new = P_new.at[3 * i:3 * i + 3, 3 * i:3 * i + 3].set(
                    jnp.linalg.inv(lams[i] + 1e-9 * jnp.eye(3)))
            s = sota.EKF(X=X_new, P=P_new)

        if t % 250 == 0 or t == STEPS - 1:
            err = np.asarray(s.X[:, :2]) - X_true[t, :, :2]
            cm = float(np.linalg.norm(err.mean(0)))
            marks[t] = cm
            print(f"  t={t:5d} cm={cm:6.3f} ({(time.time()-t0)/t*1000:.0f} ms/slot)",
                  flush=True)
    return marks


def main():
    cfg = load_config()
    cfg.world.kind = "urban"
    cfg.world.trajectory = "transect"
    cfg.time.steps = STEPS
    cfg.channel.ge_p_bad_los = 0.0
    cfg.channel.ge_p_bad_nlos = 0.0
    print(f"SOTA pose rows: steps={STEPS} seeds={SEEDS} arms={ARMS}", flush=True)
    results = {a: [] for a in ARMS}
    for seed in range(SEEDS):
        keys, wld, X, U_meas, _ = gen_truth(cfg, seed)
        for arm in ARMS:
            rng = np.random.default_rng(seed + 999)
            print(f"[seed {seed}] {arm}", flush=True)
            results[arm].append(run_arm(arm, cfg, keys, wld, X, U_meas, rng))
    print("\n== summary (common-mode, mean over seeds) ==")
    for arm in ARMS:
        ts = sorted(results[arm][0])
        line = "  ".join(f"t={t}: "
                         f"{np.mean([m[t] for m in results[arm]]):.3f}"
                         for t in ts[-4:])
        print(f"{arm:12s} {line}")
    print("DONE sota_pose")


if __name__ == "__main__":
    main()
