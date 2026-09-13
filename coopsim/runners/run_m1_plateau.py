"""B6/B8 gate: common-mode plateau on R1 urban with the v3 joint estimator.

Arms (shared CRN truth, odometry, channel realizations per seed):
  A0      relative + odometry only, the drift baseline
  oracleF A-J channel factors with the true field held fixed
  estF    A-J estimating the field jointly (Huber-TV prior + LM), THE blocker fix

Gates: A0 common-mode grows; oracleF plateaus well below it; estF bounded with field
IoU rising and LM accepting cost-decreasing steps (telemetry logged).
GE blockage is disabled here (B6 isolates the estimator); B8 turns it back on with
the regime bank. Env knobs: M1_STEPS, M1_SEEDS, M1_ARMS, M1_GE.
Run: python -u -m coopsim.runners.run_m1_plateau
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import channel, estimator, regimes, scheduler, world
from coopsim.config import crn_keys, load_config
from coopsim.metrics.field_metrics import field_iou, field_mae
from coopsim import se2

STEPS = int(os.environ.get("M1_STEPS", "1500"))
SEEDS = int(os.environ.get("M1_SEEDS", "4"))
SEED_OFFSET = int(os.environ.get("M1_SEED_OFFSET", "0"))
WARMUP = int(os.environ.get("M1_WARMUP", "0"))   # estF: field-only channel slots
BOOT = int(os.environ.get("M1_BOOT", "0"))       # estF: batch-field bootstrap slots


def batch_field(obs_pos, obs_att, extent, res, c):
    """Linear tomographic batch solve of the field from accumulated
    (pos_i, pos_j, attenuation) at accurate early poses. Anchors the field to the
    world so the estimated-field common mode is bounded (breaks the joint gauge)."""
    from coopsim.field import line_integral
    from coopsim.runners.run_rti_real import solve_field
    grad = jax.jit(jax.grad(lambda F, xi, xj: line_integral(xi, xj, F, extent)))
    W = np.array([np.asarray(grad(jnp.zeros((res, res)), jnp.asarray(pi),
                                  jnp.asarray(pj))).reshape(-1)
                  for pi, pj in obs_pos])
    return solve_field(W, np.clip(np.array(obs_att), 0, None), res,
                       lam_tv=c.tv_lambda, lam0=1.0 / c.f_prior_var)
ARMS = os.environ.get("M1_ARMS", "A0,oracleF,estF").split(",")
USE_GE = os.environ.get("M1_GE", "0") == "1"


def make_sched(n, budget, r_conflict):
    """Conflict-aware round-robin: greedy MWIS on link ages. Uses true positions
    (benign gate simplification; the full harness schedules on estimates)."""
    mwis = jax.jit(lambda w, adj, cand: scheduler.greedy_mwis(w, adj, budget, cand))

    def step(pos, age):
        adj = scheduler.conflict_adj(jnp.asarray(pos), n, r_conflict)
        _, _, valid = scheduler.link_index(n)
        chosen = mwis(age, adj, valid)
        age_new = jnp.where(chosen, 0.0, age + 1.0)
        return chosen.reshape(n, n), age_new

    return step


def gen_truth(cfg, seed):
    keys = crn_keys("urban", "M1", seed)
    wld = world.make_world(keys.world_gen, cfg)
    X, U = world.make_trajectories(keys.target_gen, cfg)
    rng = np.random.default_rng(seed + 777)
    sr = np.deg2rad(cfg.robots.sigma_rot_deg)
    U_meas = U + rng.normal(0, 1, U.shape) * np.array(
        [cfg.robots.sigma_trans, cfg.robots.sigma_trans, sr])
    return keys, wld, X, U_meas, rng


def run_arm(arm, cfg, keys, wld, X_true, U_meas, rng, seed=0):
    n, W = cfg.shapes.n_robots, cfg.shapes.window
    ext, dt = cfg.world.extent, cfg.time.dt
    use_channel = arm in ("oracleF", "estF")
    est_field = arm == "estF"
    # partial-gauge ladder: exogenous absolute measurements fused per slot with
    # steady-state Kalman gains (compass 2 deg, anchor ranges 0.1 m)
    ladder = arm in ("A_M", "A_A1", "A_A2")
    rng_abs = np.random.default_rng(seed + 4242)
    SIG_H, K_TH = np.deg2rad(2.0), 0.1
    SIG_R, K_P = 0.1, 0.2
    ANCHORS = np.array([[10.0, 10.0], [40.0, 40.0]])
    n_anch = {"A_A1": 1, "A_A2": 2}.get(arm, 0)

    c = estimator.make_consts(cfg)
    F_true = jnp.asarray(wld.F_true)
    F0 = F_true if arm == "oracleF" else jnp.zeros_like(F_true)
    est = estimator.init(jnp.asarray(X_true[0]), F0, W)
    lut = channel.nakagami_lut(cfg.channel.nakagami_m.urban)
    ch_state = channel.init_state(keys.fading, n)
    belief = regimes.init_belief(n)

    ch_step = jax.jit(lambda t, pos, vel, s, tx: channel.step(
        t, keys, pos, vel, F_true, s, tx, lut, cfg.channel, ext, dt))
    gn = jax.jit(partial(estimator.gn_update, c=c, use_channel=use_channel,
                         estimate_field=est_field, iters=cfg.estimator.gn_iters))
    warm = jax.jit(estimator.push_warmup)
    slide = jax.jit(partial(estimator.push_slide, c=c, use_channel=use_channel,
                            estimate_field=est_field))
    # field-warmup variant (estF): channel updates the FIELD only via the inner
    # machinery while poses run channel-free, breaking the wrong-field feedback loop
    warmup = WARMUP if arm == "estF" else 0
    if warmup:
        gn_w = jax.jit(partial(estimator.gn_update, c=c, use_channel=False,
                               estimate_field=False, iters=cfg.estimator.gn_iters))
        slide_w = jax.jit(partial(estimator.push_slide, c=c, use_channel=False,
                                  estimate_field=False))
        inner = estimator.init_inner(jnp.zeros_like(F_true))
        inner_step = jax.jit(lambda i, x, d: estimator.hybrid_inner_step(i, x, d, c))
    reg_update = jax.jit(lambda b, obs, prh, plos: regimes.update(
        b, obs, prh, plos, cfg.channel, c.rssi_var0, c.sigma_f))

    sr = np.deg2rad(cfg.relative.sigma_rot_deg)
    rel_sig = np.array([cfg.relative.sigma_trans, cfg.relative.sigma_trans, sr])
    budget = int(cfg.channel.budget_frac * n)
    sched_step = make_sched(n, budget, cfg.channel.conflict_radius)
    age = jnp.zeros(n * n)
    marks = {}
    telemetry = []
    field_snaps = []
    snap_ts, err_means = [], []
    save_field = os.environ.get("M1_SAVE_FIELD", "")
    freeze = int(os.environ.get("M1_FREEZE", "0")) if arm == "estF" else 0
    boot = BOOT if (est_field and arm == "estF") else 0
    boot_pos, boot_att = [], []
    n_succ = 0
    t0 = time.time()
    for t in range(1, STEPS):
        pos, vel = X_true[t, :, :2], (X_true[t, :, :2] - X_true[t - 1, :, :2]) / dt
        tx, age = sched_step(pos, age)
        obs, ch_state = ch_step(t, jnp.asarray(pos), jnp.asarray(vel),
                                ch_state, jnp.asarray(tx))
        succ = np.asarray(obs.success)

        # delivered payload: the sender's relative measurement of the receiver
        zrel = np.zeros((n, n, 3), np.float32)
        rel_mask = np.zeros((n, n), bool)
        d_ij = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        for i in range(n):
            for j in range(n):
                if succ[i, j] and d_ij[i, j] <= cfg.relative.range:
                    z = np.asarray(se2.between(jnp.asarray(X_true[t, i]),
                                               jnp.asarray(X_true[t, j])))
                    noise = rng.normal(0, 1, 3) * rel_sig
                    if rng.random() < cfg.relative.outlier_rate:
                        noise *= cfg.relative.outlier_mag
                    zrel[i, j] = z + noise
                    rel_mask[i, j] = True

        if use_channel and USE_GE:
            cur = min(t, W - 1)
            pos_hat = est.X[cur, :, :2]
            pr_hat = channel.pr_mean(pos_hat, est.F, ext, cfg.channel.p0_dbm,
                                     cfg.channel.gamma) + c.fade_mean
            plos_hat = channel.p_los(pos_hat, est.F, ext, 6.0, 1.5)
            belief = reg_update(belief, obs, pr_hat, plos_hat)
            hbar, hvar = regimes.h_moments(belief, cfg.channel)
        else:
            hbar = jnp.zeros((n, n)); hvar = jnp.zeros((n, n))

        d = estimator.plain_slot(
            jnp.asarray(U_meas[t - 1]), jnp.asarray(zrel), jnp.asarray(rel_mask),
            obs.rssi_q,
            obs.success if use_channel else jnp.zeros((n, n), bool),
            obs.attempted if use_channel else jnp.zeros((n, n), bool),
            obs.success, hbar, hvar)

        in_warmup = warmup and t < warmup
        if in_warmup:
            cur = min(t - 1, W - 1)
            inner, _, _ = inner_step(inner, est.X[cur], d)
        if t < W:
            est = warm(est, d, t)
        else:
            est = (slide_w if in_warmup else slide)(est, d)
        est, telem = (gn_w if in_warmup else gn)(est)
        if warmup and t == warmup:
            n3 = 3 * n
            est = est._replace(F=inner.F,
                               Lam_p=est.Lam_p.at[n3:, n3:].add(inner.Lam_F))

        if ladder:
            cur = min(t, W - 1)
            if arm == "A_M":
                z = X_true[t, :, 2] + rng_abs.normal(0, SIG_H, n)
                th = np.asarray(est.X[cur, :, 2])
                innov = np.arctan2(np.sin(z - th), np.cos(z - th))
                Xc = est.X[cur].at[:, 2].add(K_TH * jnp.asarray(innov))
                est = est._replace(X=est.X.at[cur].set(Xc))
            else:
                p = np.asarray(est.X[cur, :, :2])
                for a in ANCHORS[:n_anch]:
                    z = (np.linalg.norm(X_true[t, :, :2] - a, axis=1)
                         + rng_abs.normal(0, SIG_R, n))
                    dv = p - a
                    dh = np.linalg.norm(dv, axis=1) + 1e-9
                    p = p + K_P * (z - dh)[:, None] * (dv / dh[:, None])
                Xc = est.X[cur].at[:, :2].set(jnp.asarray(p))
                est = est._replace(X=est.X.at[cur].set(Xc))

        if boot:
            cur = min(t, W - 1)
            ph = np.asarray(est.X[cur, :, :2])           # accurate early poses
            rq = np.asarray(obs.rssi_q)
            for i, j in zip(*np.nonzero(succ)):
                if i < j:
                    dij = np.linalg.norm(ph[i] - ph[j]) + 1e-6
                    emit0 = (cfg.channel.p0_dbm - 10 * cfg.channel.gamma
                             * np.log10(dij) + c.fade_mean)
                    boot_pos.append((ph[i], ph[j])); boot_att.append(emit0 - rq[i, j])
            if t == boot:                                # batch-solve + inject field
                Fb = batch_field(boot_pos, boot_att, ext, F_true.shape[0], c)
                n3 = 3 * n
                est = est._replace(F=jnp.asarray(Fb, jnp.float32),
                                   Lam_p=est.Lam_p.at[n3:, n3:].add(
                                       jnp.eye(F_true.size) / c.f_prior_var))
                print(f"  [boot t={t}] batch field IoU="
                      f"{field_iou(Fb, np.asarray(wld.F_true)):.2f} "
                      f"({len(boot_att)} obs)", flush=True)

        if freeze and t == freeze:
            # map-then-localize: hold the current estimated field fixed from here on
            # (channel factors stay on; only field updates stop)
            gn = jax.jit(partial(estimator.gn_update, c=c, use_channel=use_channel,
                                 estimate_field=False, iters=cfg.estimator.gn_iters))
            slide = jax.jit(partial(estimator.push_slide, c=c,
                                    use_channel=use_channel, estimate_field=False))
            print(f"  [freeze t={t}] field frozen "
                  f"(IoU={field_iou(np.asarray(est.F), np.asarray(wld.F_true)):.2f})",
                  flush=True)

        if t % 100 == 0 or t == STEPS - 1:
            cur = min(t, W - 1)
            err = np.asarray(est.X[cur, :, :2]) - X_true[t, :, :2]
            if save_field and est_field:
                field_snaps.append(np.asarray(est.F))
                snap_ts.append(t)
                err_means.append(err.mean(0))
            cm = float(np.linalg.norm(err.mean(0)))
            rel = float(np.sqrt((np.linalg.norm(err - err.mean(0), axis=1) ** 2).mean()))
            iou = field_iou(np.asarray(est.F), np.asarray(wld.F_true)) if est_field else -1
            cost0, cost1, lam = [np.asarray(x) for x in telem]
            marks[t] = (cm, rel, iou)
            telemetry.append((t, float(cost1[-1]), float(lam[-1])))
            print(f"  t={t:5d} cm={cm:6.3f} rel={rel:6.3f} iou={iou:5.2f} "
                  f"cost={cost1[-1]:.1f} lam={lam[-1]:.1e} "
                  f"({(time.time()-t0)/t*1000:.0f} ms/slot)", flush=True)
    mae = field_mae(np.asarray(est.F), np.asarray(wld.F_true)) if est_field else -1
    if save_field and est_field and field_snaps:
        stem = save_field[:-4] if save_field.endswith(".npz") else save_field
        fname = f"{stem}_s{seed}.npz"
        np.savez_compressed(fname, field_snaps=np.stack(field_snaps),
                            F_true=np.asarray(wld.F_true),
                            snap_t=np.array(snap_ts),
                            err_mean=np.stack(err_means))
        print(f"  field snapshots -> {fname}", flush=True)
    return marks, mae


def main():
    overrides = [s for s in os.environ.get("CFG_SET", "").split(",") if s]
    if overrides:
        print(f"CFG_SET overrides: {overrides}", flush=True)
    cfg = load_config(overrides=overrides)
    cfg.world.kind = "urban"
    cfg.world.trajectory = "transect"
    cfg.time.steps = STEPS
    cfg.channel.burst_k = int(os.environ.get("M1_BURST", "1"))
    mu, var = channel.fade_stats(cfg.channel.nakagami_m.urban)
    cfg.channel._fade_mu, cfg.channel._fade_var = mu, var
    if not USE_GE:
        cfg.channel.ge_p_bad_los = 0.0
        cfg.channel.ge_p_bad_nlos = 0.0
    print(f"M1 v3 gate: steps={STEPS} seeds={SEEDS} arms={ARMS} GE={USE_GE}", flush=True)

    results = {a: [] for a in ARMS}
    for seed in range(SEED_OFFSET, SEED_OFFSET + SEEDS):
        keys, wld, X, U_meas, _ = gen_truth(cfg, seed)
        for arm in ARMS:
            rng = np.random.default_rng(seed + 999)   # same rel-noise stream per arm
            print(f"[seed {seed}] {arm}", flush=True)
            marks, mae = run_arm(arm, cfg, keys, wld, X, U_meas, rng, seed=seed)
            results[arm].append(marks)
            if mae >= 0:
                print(f"  field MAE={mae:.2f} dB/m", flush=True)

    print("\n== summary (common-mode, mean +/- std over seeds) ==")
    final = {}
    for arm in ARMS:
        ts = sorted(results[arm][0])
        line = []
        for t in ts:
            cms = np.array([m[t][0] for m in results[arm]])
            line.append(f"t={t}: {cms.mean():.3f}+/-{cms.std():.3f}")
        final[arm] = np.array([m[max(ts)][0] for m in results[arm]]).mean()
        print(f"{arm:8s} " + "  ".join(line[-4:]))

    if "A0" in final and "oracleF" in final:
        ok_o = final["oracleF"] < 0.6 * final["A0"]
        print(f"{'PASS' if ok_o else 'FAIL'} gate.oracleF_plateau "
              f"(oracleF={final['oracleF']:.3f} vs A0={final['A0']:.3f})")
    if "A0" in final and "estF" in final:
        ok_e = final["estF"] < 0.6 * final["A0"]
        print(f"{'PASS' if ok_e else 'FAIL'} gate.estF_bounded "
              f"(estF={final['estF']:.3f} vs A0={final['A0']:.3f})")
    print("DONE m1_v3")


if __name__ == "__main__":
    main()
