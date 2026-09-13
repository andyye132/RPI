"""Episode driver: one full AETHER v3 run (world -> slots -> npz + parquet row).

Python loop over slots calling jitted subroutines (channel, detections, regimes,
estimator, GM-PHD, scheduler); the slot body is pure in fixed-shape carries, so a
lax.scan flip is a later optimization. CRN discipline: all stochastic streams evolve
from named subkeys independent of the schedule and the architecture. SOTA swap rows
run in numpy off the jit path: fusion.rule ci_tracks | kla exchange extracted tracks
or PHD posteriors instead of raw detections, gmphd.backend lmb swaps the GM-PHD for
the LMB-lite tracker on the same fused detection sets.
"""
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import (channel, detmodel, dualctrl, estimator, fusion, guidance,
                     interfaces, ladder, power, regimes)
from coopsim import scheduler, se2, targets, targets_lmb, world
from coopsim.config import config_hash, crn_keys

MAX_FUSED = 64


def run_episode(cfg, arch, seed, steps=None, out_dir=None):
    T = steps or cfg.time.steps
    n, W = cfg.shapes.n_robots, cfg.shapes.window
    Mt, D = cfg.shapes.max_targets, cfg.shapes.max_det
    ML, J = cfg.shapes.max_labels, cfg.shapes.max_comp
    ext, dt = cfg.world.extent, cfg.time.dt
    kind, regime = cfg.world.kind, cfg.targets.regime
    rule, backend = cfg.fusion.rule, cfg.gmphd.backend

    keys = crn_keys(kind, regime, seed)
    wld = world.make_world(keys.world_gen, cfg)
    X_true, U_gt = world.make_trajectories(keys.target_gen, cfg)
    X_true = np.array(X_true[:T], np.float64)
    U_gt = np.array(U_gt[:T - 1], np.float64)
    tau, alive = world.gen_targets(keys.target_gen, cfg)
    tau, alive = tau[:T], alive[:T]

    rng = np.random.default_rng(seed + 31337)
    sr_o = np.deg2rad(cfg.robots.sigma_rot_deg)
    odo_noise = rng.normal(0, 1, U_gt.shape)
    U_meas = U_gt + odo_noise * np.array(
        [cfg.robots.sigma_trans, cfg.robots.sigma_trans, sr_o])
    sr_r = np.deg2rad(cfg.relative.sigma_rot_deg)
    rel_sig = np.array([cfg.relative.sigma_trans, cfg.relative.sigma_trans, sr_r])
    # Pre-drawn and indexed by (t,i,j): drawing inside the delivery loop makes the
    # stream position depend on how many links delivered earlier, silently breaking
    # paired CRN for any arm that changes the schedule.
    rel_noise = rng.normal(0, 1, (T, n, n, 3))
    rel_outl = rng.random((T, n, n))

    arch_fns = interfaces.make_arch(cfg, arch)
    c = arch_fns["consts"]
    F_true = jnp.asarray(wld.F_true)
    F_sense = jnp.asarray(wld.F_sense)
    F0 = F_true if (arch == "A_J" and cfg.estimator.oracle_field) \
        else jnp.zeros_like(F_true)
    est = estimator.init(jnp.asarray(X_true[0]), F0, W)
    lut = channel.nakagami_lut(getattr(cfg.channel.nakagami_m, kind))
    ch_state = channel.init_state(keys.fading, n)
    belief = regimes.init_belief(n)
    phd = jax.vmap(lambda r: targets.init(J, r))(jnp.arange(n))
    lmb = [targets_lmb.LMB(robot_id=i, cap=J) for i in range(n)] \
        if backend == "lmb" else None
    kappa = cfg.detection.clutter_rate / (np.pi * cfg.detection.r_max ** 2)

    ch_step = jax.jit(lambda t, pos, vel, s, tx, ptx: channel.step(
        t, keys, pos, vel, F_true, s, tx, lut, cfg.channel, ext, dt, ptx))
    det_step = jax.jit(lambda t, X, tt, al: detmodel.gen_detections(
        t, keys, X, tt, al, F_sense, ext, cfg.detection, D))
    reg_update = jax.jit(lambda b, obs, prh, plos: regimes.update(
        b, obs, prh, plos, cfg.channel, c.rssi_var0, c.sigma_f))
    to_world = jax.jit(jax.vmap(fusion.dets_to_world))
    phd_step = jax.jit(jax.vmap(
        lambda p, z, R, m, pd: targets.step(p, z, R, m, pd, kappa, cfg.gmphd,
                                            dt, cfg.targets.sigma_a,
                                            cfg.targets.survival, J)))
    phd_extract = jax.jit(jax.vmap(
        lambda p: targets.extract(p, cfg.gmphd.extract, ML)))
    phd_extract_cov = jax.jit(jax.vmap(
        lambda p: targets.extract_with_cov(p, cfg.gmphd.extract, ML)))
    pd_comp = jax.jit(jax.vmap(jax.vmap(
        lambda xi, m, F: detmodel.p_detect(xi, m[:2], F, ext, cfg.detection),
        in_axes=(None, 0, None)), in_axes=(0, 0, None)))
    policy = arch_fns["sched_policy"]

    bl = cfg.get("bilevel")
    bl_kind = bl.inner if bl is not None else "ge"
    bl_est = None
    if arch.startswith("A_BL") and bl_kind == "bernoulli":
        bl_est = ladder.BernoulliInner(n)
    elif arch.startswith("A_BL") and bl_kind == "gudmundson":
        bl_est = ladder.GudmundsonInner()
    prev_exp_del = None

    def _sched(pos, age, belief_s, pr_hat_s, cov_diag, exp_del_s):
        extras = {"pose_cov_diag": cov_diag, "belief": belief_s,
                  "pr_hat": pr_hat_s, "cfg_ch": cfg.channel,
                  "exp_del": exp_del_s if bl_est is not None else None}
        return scheduler.schedule(policy, pos, n, int(cfg.channel.budget_frac * n),
                                  cfg.channel.conflict_radius, age, extras)
    sched_fn = jax.jit(_sched)

    budget = int(cfg.channel.budget_frac * n)
    tx = jnp.asarray(scheduler_bootstrap(n, budget))
    age = jnp.zeros(n * n)
    # DELIVERY age (slots since a delivery from i reached j), distinct from the
    # per-link ATTEMPT age above that drives round-robin fairness.
    aoi = np.zeros((n, n))

    # Off by default: published comparisons keep the fixed-power radio.
    pw_cfg = getattr(cfg, "power", None)
    pw_on = bool(getattr(pw_cfg, "enable", False))
    pw_mode = str(getattr(pw_cfg, "mode", "info"))
    pw_target = float(getattr(pw_cfg, "p_target", 0.5))
    pw_floor = float(getattr(pw_cfg, "floor_dbm", -90.0))
    p_tx = jnp.full((n,), float(cfg.channel.p0_dbm))

    # Off by default: published comparisons keep the age-ordered schedule.
    gd_cfg = getattr(cfg, "guidance", None)
    gd_on = bool(getattr(gd_cfg, "enable", False))
    if gd_on:
        gd_ii, gd_jj, gd_valid = scheduler.link_index(n)
        gd_sens = guidance.make_link_sens(cfg.world.extent, cfg.channel.p0_dbm,
                                          cfg.channel.gamma, n, gd_ii, gd_jj)
        gd_pick = jax.jit(lambda sn, wg, adj, cand, Sp, bn: guidance.greedy_gauge_links(
            sn, wg, adj, cand, budget, Sp, float(getattr(gd_cfg, "eps", 1e-9)), bn))
        gd_S = jnp.zeros((3, 3))
        gd_base = jnp.array([cfg.world.extent / 2.0, cfg.world.extent / 2.0])
        gd_decay = float(getattr(gd_cfg, "decay", 0.98))
        gd_mode = str(getattr(gd_cfg, "mode", "gauge"))
        gd_lam_route = float(getattr(gd_cfg, "lam_route", 0.0))
        gd_lam_rel = float(getattr(gd_cfg, "lam_rel", 2.0))
        gd_lam_age = float(getattr(gd_cfg, "lam_age", 0.02))
        # From the initial ESTIMATE, never the truth; frozen so the rotation
        # column keeps one scale for the episode.
        _q0 = np.asarray(est.X[0, :, :2]) - np.asarray(gd_base)
        gd_rho0 = float(np.sqrt((_q0 ** 2).sum() + n))
        gd_dtheta = float(getattr(gd_cfg, "dtheta_deg", 12.0)) * np.pi / 180.0
        # X_true is MUTATED in place by the traj modes, so the untouched survey
        # must be kept separately.
        gd_Xnom = X_true.copy()
        gd_corr = float(getattr(gd_cfg, "corridor_m", 0.0))
        gd_cover = guidance.make_chord_coverage(cfg.world.extent, cfg.shapes.grid,
                                                n, gd_ii, gd_jj)
        gd_live = jnp.asarray(np.asarray(gd_valid), jnp.float64)
        gd_cov = jnp.zeros(cfg.shapes.grid * cfg.shapes.grid)
        gd_dev = []
        gd_tie = []
        # Trailing window of the realized per-slot gauge Gram. The endpoint is a
        # min over windows of length T_0, which the scheduler's decayed
        # accumulator does not track.
        from collections import deque as _deque
        gd_T0 = int(getattr(gd_cfg, "excite_window", 25))
        # Effort-matched control: random headings from the same candidate set
        # under the same corridor, so only the objective differs.
        gd_randhead = bool(getattr(gd_cfg, "rand_head", False))
        gd_ring = _deque(maxlen=gd_T0)
        gd_rng = np.random.default_rng(seed + 8191)
    rel_trust = jnp.ones((n, n))
    exp_del_sched = jnp.zeros((n, n))
    hy_oracle = arch.startswith("A_HY") and cfg.estimator.oracle_field
    F_in = F_true if hy_oracle else jnp.zeros_like(F_true)
    inner = (estimator.init_inner_once(F_in, n) if arch in ("A_HYM", "A_HYK")
             else estimator.init_inner(F_in))
    inner_step = jax.jit(lambda i, x, d: estimator.hybrid_inner_step(i, x, d, c))
    once_step = jax.jit(lambda i, x, d: estimator.hybrid_once(i, x, d, c))
    oncek_step = jax.jit(lambda i, x, d: estimator.hybrid_once_gauge(i, x, d, c))
    msg_relin = jax.jit(lambda i, x, d: estimator.hybrid_message(i, x, d, c))
    msg_known = jax.jit(lambda x, d: estimator.hybrid_message_known(F_true, x, d, c))
    if arch_fns["project_gauge"]:
        gproj = jax.jit(estimator.gauge_project)
    elif arch_fns.get("project_rand"):
        # Rate-matched control: random 3-D subspace instead of the gauge one. Key
        # folded on the slot, so it is redrawn each slot but stays a deterministic
        # function of (seed, t) and keeps CRN intact.
        _rk = jax.random.PRNGKey(seed + 4242)
        _rct = [0]

        def gproj(l, e, x):
            _rct[0] += 1
            return estimator.rand_project(l, e, x,
                                          jax.random.fold_in(_rk, _rct[0]))
    else:
        gproj = (lambda l, e, x: (l, e))
    logs = dict(x_est=np.zeros((T, n, 3), np.float32),
                tracks=np.zeros((T, n, ML, 5), np.float32),
                tracks_mask=np.zeros((T, n, ML), bool),
                sched=np.zeros((T, n, n), bool), succ=np.zeros((T, n, n), bool),
                field_snaps=[], telem=np.zeros((T, 2), np.float32),
                fused_drop=np.zeros(T, np.float32),
                p_tx=np.full((T, n), np.float32(cfg.channel.p0_dbm)),
                gauge_lmin=np.zeros(T, np.float32),
                deliv_pred=[], deliv_real=[])
    t0 = time.time()

    for t in range(1, T):
        if gd_on and gd_mode in ("traj", "trajrand", "excite") \
                and not (gd_mode == "excite" and gd_corr <= 0.0):
            # Zero budget must skip, not clamp: recomputing poses is not
            # bit-identical to the emitted odometry, so a clamped controller
            # would still perturb the estimator and break gd_corr = 0.
            # Heading is perturbed around the nominal mission, capped by gd_dtheta,
            # so the team still flies the survey.
            prev = X_true[t - 1]
            nom = X_true[t]                                  # nominal next pose
            nom_th = np.arctan2(nom[:, 1] - prev[:, 1], nom[:, 0] - prev[:, 0])
            step = np.linalg.norm(nom[:, :2] - prev[:, :2], axis=1)
            cand = np.array([0.0, -1.0, 1.0, -2.0, 2.0]) * gd_dtheta
            newP = prev[:, :2].copy()
            if gd_mode == "excite":
                # Maximize the increment to lambda_min of the accumulated gauge
                # Gram (the quantity Assumption 3 is about), one pass of
                # coordinate descent per slot, inside a corridor of gd_corr
                # metres. gd_corr = 0 reproduces the nominal arm.
                F_ex = est.F if arch == "A_J" else \
                    (inner.F if arch.startswith("A_HY") else F_true)
                nom_dir = np.arctan2(gd_Xnom[t, :, 1] - gd_Xnom[t - 1, :, 1],
                                     gd_Xnom[t, :, 0] - gd_Xnom[t - 1, :, 0])
                th = nom_dir.copy()
                if gd_randhead:
                    th = nom_dir + cand[gd_rng.integers(0, len(cand), size=n)]
                for r in ([] if gd_randhead else range(n)):
                    best, bth = -np.inf, th[r]
                    gs = []
                    for dth in cand:
                        tht = th.copy(); tht[r] = nom_dir[r] + dth
                        P = prev[:, :2] + np.stack([step * np.cos(tht),
                                                    step * np.sin(tht)], 1)
                        P = np.clip(P, 2.0, ext - 2.0)
                        sn_c = gd_sens(F_ex, jnp.asarray(P), gd_base, gd_rho0)
                        prh_c = channel.pr_mean(jnp.asarray(P), F_ex,
                                                cfg.world.extent,
                                                cfg.channel.p0_dbm,
                                                cfg.channel.gamma) + c.fade_mean
                        wg_c, _ = guidance.link_weights(
                            prh_c, c.rssi_var0, cfg.channel.decode_dbm,
                            c.sigma_f, gd_ii, gd_jj)
                        snp_c = np.asarray(sn_c, np.float64)
                        add = np.einsum("l,lk,lj->kj",
                                        np.asarray(wg_c, np.float64),
                                        snp_c, snp_c)
                        S_win = (np.sum(gd_ring, axis=0) if gd_ring
                                 else np.zeros((3, 3)))
                        S_try = S_win + add
                        g = float(np.linalg.eigvalsh(0.5 * (S_try + S_try.T))[0])
                        gs.append(g)
                        if g > best:
                            best, bth = g, tht[r]
                    # Objective flatness across candidates: lambda_min's greedy
                    # increments can be exactly zero, making the pick arbitrary.
                    ga = np.asarray(gs, float)
                    sp = (ga.max() - ga.min()) / max(abs(ga.max()), 1e-30)
                    gd_tie.append((sp, float(abs(bth - nom_dir[r]))))
                    th[r] = bth
                newP = prev[:, :2] + np.stack([step * np.cos(th),
                                               step * np.sin(th)], 1)
                # Clamp POSITION, not heading: redirecting the heading leaves a
                # zero budget wandering, since a fixed-length step toward the
                # waypoint does not land on it.
                off = newP - gd_Xnom[t, :, :2]
                d = np.linalg.norm(off, axis=1)
                scale = np.minimum(1.0, gd_corr / np.maximum(d, 1e-12))
                newP = gd_Xnom[t, :, :2] + off * scale[:, None]
                # Blend heading by the same scale rather than recomputing it from
                # travel direction, which differs from survey heading at wall
                # bounces and would perturb odometry even at scale 0.
                dth_ex = np.arctan2(np.sin(th - gd_Xnom[t, :, 2]),
                                    np.cos(th - gd_Xnom[t, :, 2]))
                th = gd_Xnom[t, :, 2] + scale * dth_ex
                gd_dev.append(float(np.linalg.norm(
                    newP - gd_Xnom[t, :, :2], axis=1).mean()))
                # Realized contribution enters the ring, so the controller
                # re-excites whichever mode is about to age out.
                sn_r = gd_sens(F_ex, jnp.asarray(newP), gd_base, gd_rho0)
                prh_r = channel.pr_mean(jnp.asarray(newP), F_ex,
                                        cfg.world.extent, cfg.channel.p0_dbm,
                                        cfg.channel.gamma) + c.fade_mean
                wg_r, _ = guidance.link_weights(prh_r, c.rssi_var0,
                                                cfg.channel.decode_dbm,
                                                c.sigma_f, gd_ii, gd_jj)
                snp_r = np.asarray(sn_r, np.float64)
                gd_ring.append(np.einsum("l,lk,lj->kj",
                                         np.asarray(wg_r, np.float64),
                                         snp_r, snp_r))
            elif gd_mode == "trajrand":
                pick = gd_rng.integers(0, len(cand), size=n)
                th = nom_th + cand[pick]
                newP = prev[:, :2] + np.stack([step * np.cos(th),
                                               step * np.sin(th)], 1)
            else:
                # One pass of coordinate descent on new voxel coverage.
                th = nom_th.copy()
                for r in range(n):
                    best, bth = -np.inf, th[r]
                    for dth in cand:
                        tht = th.copy(); tht[r] = nom_th[r] + dth
                        P = prev[:, :2] + np.stack([step * np.cos(tht),
                                                    step * np.sin(tht)], 1)
                        P = np.clip(P, 2.0, ext - 2.0)
                        cv = gd_cover(jnp.asarray(P), gd_live)
                        g = float(guidance.diversity_gain(gd_cov, cv))
                        if g > best:
                            best, bth = g, tht[r]
                    th[r] = bth
                newP = prev[:, :2] + np.stack([step * np.cos(th),
                                               step * np.sin(th)], 1)
            newP = np.clip(newP, 2.0, ext - 2.0)
            X_true[t, :, :2] = newP
            X_true[t, :, 2] = th
            gd_cov = gd_cov + gd_cover(jnp.asarray(newP), gd_live)
            if gd_mode != "excite":
                gd_dev.append(float(np.abs(np.arctan2(np.sin(th - nom_th),
                                                      np.cos(th - nom_th))).mean()))
            dp = X_true[t, :, :2] - prev[:, :2]
            ct, st_ = np.cos(prev[:, 2]), np.sin(prev[:, 2])
            U_gt[t - 1, :, 0] = ct * dp[:, 0] + st_ * dp[:, 1]
            U_gt[t - 1, :, 1] = -st_ * dp[:, 0] + ct * dp[:, 1]
            U_gt[t - 1, :, 2] = np.arctan2(np.sin(th - prev[:, 2]),
                                           np.cos(th - prev[:, 2]))
            U_meas[t - 1] = U_gt[t - 1] + odo_noise[t - 1] * np.array(
                [cfg.robots.sigma_trans, cfg.robots.sigma_trans, sr_o])
        pos = X_true[t, :, :2]
        vel = (X_true[t, :, :2] - X_true[t - 1, :, :2]) / dt
        obs, ch_state = ch_step(t, jnp.asarray(pos), jnp.asarray(vel), ch_state,
                                tx, p_tx)
        succ = np.asarray(obs.success)
        aoi += 1.0
        aoi[succ] = 0.0
        dets = det_step(t, jnp.asarray(X_true[t]), jnp.asarray(tau[t]),
                        jnp.asarray(alive[t]))

        # relative-pose payloads on delivered in-range links
        zrel = np.zeros((n, n, 3), np.float32)
        rel_mask = np.zeros((n, n), bool)
        d_ij = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        for i, j in zip(*np.nonzero(succ)):
            if d_ij[i, j] <= cfg.relative.range:
                z = np.asarray(se2.between(jnp.asarray(X_true[t, i]),
                                           jnp.asarray(X_true[t, j])))
                # Indexed, not drawn -- see the CRN note at the pre-draw above.
                noise = rel_noise[t, i, j] * rel_sig
                if rel_outl[t, i, j] < cfg.relative.outlier_rate:
                    noise = noise * cfg.relative.outlier_mag
                zrel[i, j] = z + noise
                rel_mask[i, j] = True

        cur = min(t, W - 1)
        if arch != "A0":
            pos_hat = est.X[cur, :, :2]
            # channel-state prediction field: est.F (A_J), inner field (A_HY),
            # distance-only zeros (A_BL, the published scalar abstraction)
            F_bel = est.F if arch == "A_J" else \
                (inner.F if arch.startswith("A_HY") else jnp.zeros_like(est.F))
            pr_hat = channel.pr_mean(pos_hat, F_bel, ext, cfg.channel.p0_dbm,
                                     cfg.channel.gamma) + c.fade_mean
            plos_hat = channel.p_los(pos_hat, F_bel, ext, 6.0, 1.5)
            belief = reg_update(belief, obs, pr_hat, plos_hat)
            hbar, hvar = regimes.h_moments(belief, cfg.channel)
            if arch.startswith("A_BL"):
                att = np.asarray(obs.attempted)
                if prev_exp_del is not None and att.any():
                    logs["deliv_pred"].append(prev_exp_del[att].astype(np.float32))
                    logs["deliv_real"].append(succ[att])
                if bl_est is None:
                    pr_s = pr_hat[..., None] - jnp.array([0.0, cfg.channel.block_extra_db])
                    p_succ = jax.scipy.stats.norm.cdf(
                        (pr_s - cfg.channel.decode_dbm) / c.sigma_f)
                    exp_del = (belief * p_succ).sum(-1)
                    prev_exp_del = np.asarray(exp_del)
                elif bl_kind == "bernoulli":
                    bl_est.update(att, succ)
                    prev_exp_del = bl_est.exp_del()
                    exp_del = jnp.asarray(prev_exp_del)
                else:
                    pe = np.asarray(pos_hat)
                    bl_est.update(att, succ, np.asarray(obs.rssi_q), pe,
                                  cfg.channel, c.fade_mean, c.sigma_f)
                    prev_exp_del = bl_est.exp_del(pe, cfg.channel,
                                                  c.fade_mean, c.sigma_f)
                    exp_del = jnp.asarray(prev_exp_del)
                if arch == "A_BLF":                  # hard SELECTION gate q in {0,1}
                    # Prop 1 selection clause: drop the lower half by exp_del among
                    # delivered in-range links. A global threshold is vacuous, since
                    # delivering links already carry the highest exp_del.
                    ed = np.asarray(exp_del)
                    rt = np.ones((n, n), np.float32)
                    active = rel_mask & (ed > 0)
                    if active.sum() >= 2:
                        rt[active & (ed <= np.median(ed[active]))] = 0.0
                    rel_trust = jnp.asarray(rt)
                elif arch == "A_BLS":                # schedule-only, no reweight
                    rel_trust = jnp.ones((n, n))
                else:                                # A_BL: continuous reweight
                    rel_trust = 0.5 + 0.5 * exp_del  # scalar interface (Prop 1)
                exp_del_sched = exp_del
                hbar = jnp.zeros((n, n)); hvar = jnp.zeros((n, n))
        else:
            hbar = jnp.zeros((n, n)); hvar = jnp.zeros((n, n))

        # A_HY's inner consumes channel observables though its outer is channel-free.
        see_channel = arch_fns["use_channel"] or arch.startswith("A_HY")
        d_slot = estimator.plain_slot(
            jnp.asarray(U_meas[t - 1]), jnp.asarray(zrel), jnp.asarray(rel_mask),
            obs.rssi_q,
            obs.success if see_channel else jnp.zeros((n, n), bool),
            obs.attempted if see_channel else jnp.zeros((n, n), bool),
            obs.success, hbar, hvar,
            # Transmit power is a known control input, so the estimator gets it;
            # without it the channel factors mis-model the link budget by exactly
            # the power reduction. Offset, not absolute, so the fixed-power path
            # is dp = 0 and byte-identical.
            dp=p_tx - float(cfg.channel.p0_dbm))
        if arch.startswith("A_BL"):
            d_slot = d_slot._replace(rel_trust=rel_trust)
        elif arch.startswith("A_HY"):
            x_pred = jax.vmap(se2.compose)(est.X[min(t - 1, W - 1)], d_slot.u)
            if hy_oracle:                                    # field fixed to truth
                mlam, meta = msg_known(x_pred, d_slot)
            elif arch == "A_HYM":       # state committed once, after relinearization
                _, mlam, meta = once_step(inner, x_pred, d_slot)
            elif arch == "A_HYK":
                _, mlam, meta = oncek_step(inner, x_pred, d_slot)
            else:
                inner, mlam, meta = inner_step(inner, x_pred, d_slot)
            mlam, meta = gproj(mlam, meta, x_pred)           # A_HYG: 9-number message
            d_slot = d_slot._replace(msg_lam=mlam, msg_eta=meta, msg_xlin=x_pred)
        est = (arch_fns["warm"](est, d_slot, t) if t < W
               else arch_fns["slide"](est, d_slot))
        est, telem = arch_fns["gn"](est)
        cur = min(t, W - 1)
        if arch.startswith("A_HY"):
            # Relinearize at the improved pose; the frozen odometry-prediction
            # linearization is what makes A-HY lag.
            if hy_oracle:
                mlam2, meta2 = msg_known(est.X[cur], d_slot)
            elif arch == "A_HYM":
                inner, mlam2, meta2 = once_step(inner, est.X[cur], d_slot)
            elif arch == "A_HYK":
                inner, mlam2, meta2 = oncek_step(inner, est.X[cur], d_slot)
            else:
                mlam2, meta2 = msg_relin(inner, est.X[cur], d_slot)
            mlam2, meta2 = gproj(mlam2, meta2, est.X[cur])
            est = est._replace(win=est.win._replace(
                msg_lam=est.win.msg_lam.at[cur].set(mlam2),
                msg_eta=est.win.msg_eta.at[cur].set(meta2),
                msg_xlin=est.win.msg_xlin.at[cur].set(est.X[cur])))
            est, telem = arch_fns["gn"](est)
        X_hat = est.X[cur]

        # rule=detections fuses local + delivered sets in world frame; ci_tracks
        # and kla instead update locally and exchange after the PHD update.
        zw, Rw, mw = to_world(dets.z, dets.R, dets.mask, X_hat)
        if rule == "detections":
            z_f, R_f, m_f, drops = [], [], [], 0.0
            for i in range(n):
                senders = np.nonzero(succ[:, i])[0]
                if len(senders):
                    zr = jnp.concatenate([zw[s] for s in senders])
                    Rr = jnp.concatenate([Rw[s] for s in senders])
                    mr = jnp.concatenate([mw[s] for s in senders])
                else:
                    zr = jnp.zeros((1, 2)); Rr = jnp.eye(2)[None]
                    mr = jnp.zeros(1, bool)
                zi, Ri, mi, dr = fusion.fuse_sets(zw[i], Rw[i], mw[i], zr, Rr, mr,
                                                  MAX_FUSED)
                z_f.append(zi); R_f.append(Ri); m_f.append(mi); drops += float(dr)
            z_f = jnp.stack(z_f); R_f = jnp.stack(R_f); m_f = jnp.stack(m_f)
        else:
            z_f, R_f, m_f, drops = zw, Rw, mw, 0.0

        if arch_fns["estimate_field"]:
            F_pd = est.F
        elif arch == "A_J" and cfg.estimator.oracle_field:
            F_pd = F_sense
        elif arch.startswith("A_HY"):
            F_pd = inner.F
        else:
            F_pd = jnp.zeros_like(F_sense)
        if backend == "lmb":
            means = np.zeros((n, J, 4), np.float32)
            for i in range(n):
                for k, tr in enumerate(lmb[i].tracks):
                    means[i, k] = tr["m"]
            p_d = np.asarray(jnp.clip(
                pd_comp(X_hat[:, :2], jnp.asarray(means), F_pd), 0.05, 0.98))
            z_np, R_np, m_np = np.asarray(z_f), np.asarray(R_f), np.asarray(m_f)
            tracks = np.zeros((n, ML, 5), np.float32)
            tmask = np.zeros((n, ML), bool)
            for i in range(n):
                lmb[i].predict(dt, cfg.targets.sigma_a, cfg.targets.survival)
                lmb[i].update(z_np[i], R_np[i], m_np[i],
                              p_d[i, :len(lmb[i].tracks)], kappa)
                tracks[i], tmask[i] = lmb[i].extract(ML)
        else:
            p_d = jnp.clip(pd_comp(X_hat[:, :2], phd.m, F_pd), 0.05, 0.98)
            phd = phd_step(phd, z_f, R_f, m_f, p_d)
            if rule == "kla":
                # delivered links carry the senders' post-update PHDs; the
                # receiver's own PHD state continues from the fused result
                phd_np = [np.asarray(f) for f in phd]
                fused = []
                for i in range(n):
                    local = targets.PHD(*(f[i] for f in phd_np))
                    senders = np.nonzero(succ[:, i])[0]
                    if len(senders):
                        local = fusion.kla_fuse(local, [
                            targets.PHD(*(f[s] for f in phd_np))
                            for s in senders])
                    fused.append(local)
                phd = jax.tree.map(lambda *xs: jnp.stack(xs), *fused)
            if rule == "ci_tracks":
                # delivered links carry extracted tracks + covariances; the
                # fused picture is logged, the PHD stays local-only
                e, mk, Pc = phd_extract_cov(phd)
                e, mk, Pc = np.asarray(e), np.asarray(mk), np.asarray(Pc)
                tracks = np.zeros((n, ML, 5), np.float32)
                tmask = np.zeros((n, ML), bool)
                for i in range(n):
                    senders = np.nonzero(succ[:, i])[0]
                    tracks[i], tmask[i] = fusion.ci_fuse_tracks(
                        e[i], mk[i], Pc[i], [e[s] for s in senders],
                        [mk[s] for s in senders], [Pc[s] for s in senders])
            else:
                tracks, tmask = phd_extract(phd)

        age = jnp.where(jnp.asarray(tx).reshape(-1), 0.0, age + 1.0)
        # Uniform pose covariance reduces w_logdet to the same ordering as A0's
        # round-robin. Deliberate: it keeps the arms schedule-matched so
        # differences trace to the coordination variable, not to scheduling.
        pose_cov_diag = jnp.ones(n)
        pr_hat_s = pr_hat if arch != "A0" else jnp.zeros((n, n))
        if gd_on:
            # Reads ESTIMATES only: no true field, no true poses, no realized RSSI.
            P_hat = jnp.asarray(pos_hat_for_sched(X_hat))
            # rho0 is FROZEN for the episode: recomputing it per slot makes the
            # rotation column's scale time-varying, so gd_S would be a sum in a
            # moving basis. Matches diag_alpha's convention.
            rho_h = gd_rho0
            # Same field-estimate rule as the channel-state prediction above;
            # oracle_field arms legitimately hold F_true.
            F_hat = est.F if arch == "A_J" else \
                (inner.F if arch.startswith("A_HY") else F_true)
            sn = gd_sens(F_hat, P_hat, gd_base, rho_h)
            prh_g = channel.pr_mean(P_hat, F_hat, cfg.world.extent,
                                    cfg.channel.p0_dbm, cfg.channel.gamma) + c.fade_mean
            wg, p_del = guidance.link_weights(prh_g, c.rssi_var0,
                                              cfg.channel.decode_dbm, c.sigma_f,
                                              gd_ii, gd_jj)
            bonus = guidance.relative_value(p_del, age / (1.0 + age.max()),
                                            gd_lam_rel, gd_lam_age)
            if gd_mode == "dual":
                # EXPLOIT term, added to the probe objective in gd_pick:
                # lam_route = 0 is pure probing, large is pure exploitation.
                # All three inputs are observable at the estimated field and poses.
                ndet = np.asarray(dets.mask).sum(axis=1).astype(np.float64)
                dv = dualctrl.detection_value(np.asarray(p_del).reshape(n, n),
                                              ndet, aoi)
                bonus = bonus + gd_lam_route * jnp.asarray(dv)
            adj_g = scheduler.conflict_adj(P_hat, n, cfg.channel.conflict_radius)
            if gd_mode == "random":
                # Effort-matched control: same budget and conflict structure, the
                # objective replaced by noise.
                wrand = jnp.asarray(gd_rng.random(n * n))
                chosen = scheduler.greedy_mwis(wrand, adj_g, budget, gd_valid)
            else:
                chosen = gd_pick(sn, wg, adj_g, gd_valid, gd_S, bonus)
            gd_S = guidance.window_update(gd_S, sn, wg, chosen, gd_decay)
            logs["gauge_lmin"][t] = float(jnp.linalg.eigvalsh(
                0.5 * (gd_S + gd_S.T))[0])
            tx = chosen.reshape(n, n)
        else:
            tx = sched_fn(jnp.asarray(pos_hat_for_sched(X_hat)), age, belief,
                          pr_hat_s, pose_cov_diag, exp_del_sched)

        # Power for the attempt set just chosen, read from ESTIMATES on the same
        # rule as above.
        if pw_on:
            F_pw = est.F if arch == "A_J" else \
                (inner.F if arch.startswith("A_HY") else F_true)
            gain_pw = channel.pr_mean(jnp.asarray(pos_hat_for_sched(X_hat)), F_pw,
                                      cfg.world.extent, 0.0, cfg.channel.gamma)
            p_tx = power.setpoint(gain_pw, tx, cfg.channel.decode_dbm, c.sigma_f,
                                  c.fade_mean, pw_target, cfg.channel.p0_dbm,
                                  pw_floor)
            if pw_mode == "uniform":
                p_tx = power.energy_matched(p_tx, tx, cfg.channel.p0_dbm)
        logs["p_tx"][t] = np.asarray(p_tx)

        if os.environ.get("HARNESS_DEBUG") and (t < 40 or t % 50 == 0):
            err = np.asarray(X_hat[:, :2]) - X_true[t, :, :2]
            print(f"  dbg t={t:4d} cm={np.linalg.norm(err.mean(0)):8.2f} "
                  f"|X|max={float(jnp.abs(est.X).max()):9.1f} "
                  f"cost={float(np.asarray(telem[1])[-1]):10.1f} "
                  f"lam={float(np.asarray(telem[2])[-1]):.1e} "
                  f"nrel={int(rel_mask.sum())} nsucc={int(succ.sum())}", flush=True)
        logs["x_est"][t] = np.asarray(X_hat)
        logs["tracks"][t] = np.asarray(tracks)
        logs["tracks_mask"][t] = np.asarray(tmask)
        logs["sched"][t] = np.asarray(tx)
        logs["succ"][t] = succ
        logs["telem"][t] = [float(np.asarray(telem[1])[-1]),
                            float(np.asarray(telem[2])[-1])]
        logs["fused_drop"][t] = drops
        if t % cfg.logging.field_snap_every == 0:
            logs["field_snaps"].append(np.asarray(est.F))

    if gd_on and gd_mode == "excite" and gd_tie:
        _sp = np.array([a for a, _ in gd_tie])
        _of = np.array([b for _, b in gd_tie])
        print(f"  [X9 ties] relative objective spread across candidates: "
              f"median {np.median(_sp):.3e}  frac<1e-6 {np.mean(_sp < 1e-6):.3f}", flush=True)
        print(f"  [X9 ties] chosen |offset|: frac at nominal {np.mean(_of < 1e-9):.3f}  "
              f"median {np.degrees(np.median(_of)):.2f} deg", flush=True)
    wall = time.time() - t0
    out = dict(X_true=X_true, tau=tau, alive=alive, F_true=wld.F_true,
               x_est=logs["x_est"], tracks=logs["tracks"],
               tracks_mask=logs["tracks_mask"], sched=logs["sched"],
               succ=logs["succ"], telem=logs["telem"],
               fused_drop=logs["fused_drop"], p_tx=logs["p_tx"],
               gauge_lmin=logs["gauge_lmin"],
               field_snaps=np.stack(logs["field_snaps"]) if logs["field_snaps"]
               else np.zeros((0,) + wld.F_true.shape),
               deliv_pred=np.concatenate(logs["deliv_pred"])
               if logs["deliv_pred"] else np.zeros(0, np.float32),
               deliv_real=np.concatenate(logs["deliv_real"])
               if logs["deliv_real"] else np.zeros(0, bool),
               wall_per_slot=wall / T, cfg_hash=config_hash(cfg),
               arch=arch, seed=seed, world=kind, regime=regime)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        tag = f"_{bl_kind}" if (arch.startswith("A_BL") and bl_kind != "ge") else ""
        path = p / f"{kind}_{regime}_{arch}_s{seed}{tag}.npz"
        np.savez_compressed(path, **out)
        return path, out
    return None, out


def scheduler_bootstrap(n, budget):
    tx = np.zeros((n, n), bool)
    for k in range(budget):
        tx[(2 * k) % n, (2 * k + 1) % n] = True
    return tx


def pos_hat_for_sched(X_hat):
    return X_hat[:, :2]


pos_hat_for_sched.__doc__ = "Scheduler consumes estimated positions only."
