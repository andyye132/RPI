"""World generation for AETHER v3: attenuation fields, robot trajectories, targets.

All numpy, run once per episode (only the per-slot loop is JAX). Field units are dB
per meter: line_integral(F) over a chord returns the through-loss in dB, so a block
rasterized at L/thickness accrues ~L dB per crossing (spec: urban through-loss
U(10,20) dB per crossing).

Survival convention: the spec gives survival 0.995; slots are 0.1 s, so it is
applied per second (per-slot survival**dt), keeping target count near nominal over a
300 s episode instead of 20 s mean lifetimes.
"""
from typing import NamedTuple

import numpy as np


class Wld(NamedTuple):
    F_true: np.ndarray     # (G,G) dB/m, radio attenuation
    F_sense: np.ndarray    # (G,G) dB/m, sensing occlusion (== F_true unless CH5)
    meta: dict


def _np_rng(key):
    k = np.asarray(key)
    return np.random.default_rng(int(k[0]) << 32 | int(k[1]))


def make_world(key, cfg) -> Wld:
    rng = _np_rng(key)
    G = cfg.shapes.grid
    ext = cfg.world.extent
    vox = ext / G
    F = np.zeros((G, G), np.float32)
    kind = cfg.world.kind
    meta = {"kind": kind}

    if kind == "urban":
        n = rng.integers(cfg.world.urban_blocks[0], cfg.world.urban_blocks[1] + 1)
        blocks = []
        for _ in range(n):
            w, h = rng.uniform(4.0, 10.0, 2)
            x0 = rng.uniform(2.0, ext - 2.0 - w)
            y0 = rng.uniform(2.0, ext - 2.0 - h)
            loss = rng.uniform(*cfg.world.urban_loss_db)
            thick = min(w, h)
            i0, i1 = int(y0 / vox), max(int((y0 + h) / vox), int(y0 / vox) + 1)
            j0, j1 = int(x0 / vox), max(int((x0 + w) / vox), int(x0 / vox) + 1)
            F[i0:i1, j0:j1] = np.maximum(F[i0:i1, j0:j1], loss / thick)
            blocks.append((x0, x0 + w, y0, y0 + h, loss))
        meta["blocks"] = blocks
    elif kind == "terrain":
        # GP heightfield; occluded (high) cells cost the diffraction loss over a
        # typical ~5 m hill chord
        xy = np.stack(np.meshgrid(np.arange(G), np.arange(G)), -1).reshape(-1, 2) * vox
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        Kk = np.exp(-d2 / (2 * 8.0 ** 2)) + 1e-6 * np.eye(G * G)
        h = (np.linalg.cholesky(Kk) @ rng.standard_normal(G * G)).reshape(G, G)
        occ = 1.0 / (1.0 + np.exp(-(h - np.quantile(h, 0.65)) / 0.2))
        F = (cfg.world.terrain_diffraction_db / 5.0 * occ).astype(np.float32)
        meta["height"] = h
    elif kind == "forest":
        for _ in range(cfg.world.forest_scatterers):
            j, i = (rng.uniform(2.0, ext - 2.0, 2) / vox).astype(int)
            F[i, j] = rng.uniform(*cfg.world.forest_loss_db) / vox
    elif kind in ("constant", "stratified", "radial"):
        # Proposition 3's degenerate geometries. Each has a nontrivial stabilizer
        # subalgebra, so the matching gauge mode stays unobservable however well the
        # field is estimated: constant fixes all of SE(2), stratified along n fixes
        # translation along n_perp, radial about the arena centre fixes rotation.
        lvl = float(np.mean(cfg.world.urban_loss_db)) / 5.0
        yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        cx = cy = (G - 1) / 2.0
        if kind == "constant":
            F[:] = lvl
        elif kind == "stratified":
            F[:] = lvl * (1.0 + np.cos(2 * np.pi * yy / (G / 4.0))) / 2.0
        else:
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (G / 2.0)
            F[:] = lvl * (1.0 + np.cos(4 * np.pi * r)) / 2.0
        F = F.astype(np.float32)
    elif kind != "open":
        raise ValueError(f"unknown world kind {kind}")

    return Wld(F_true=F, F_sense=F.copy(), meta=meta)


def _bounce(pos, vel, lo, hi):
    for c in (0, 1):
        under, over = pos[:, c] < lo, pos[:, c] > hi
        vel[under | over, c] *= -1
        pos[:, c] = np.clip(pos[:, c], lo, hi)
    return pos, vel


def make_trajectories(key, cfg):
    """Ground-truth poses X_true (T,N,3) and odometry increments U_gt (T-1,N,3)."""
    rng = _np_rng(key)
    T, N = cfg.time.steps, cfg.shapes.n_robots
    ext, dt, spd = cfg.world.extent, cfg.time.dt, cfg.robots.speed
    kind = cfg.world.trajectory
    X = np.zeros((T, N, 3), np.float32)

    if kind == "convoy":
        # designed failure case: a line, one shared heading, no shadow diversity
        th = rng.uniform(0, 2 * np.pi)
        head = np.array([np.cos(th), np.sin(th)])
        start = np.array([ext / 2, ext / 2]) - head * ext * 0.35
        pos = start[None, :] - head[None, :] * 3.0 * np.arange(N)[:, None]
        vel = np.tile(head * spd, (N, 1))
    else:
        pos = rng.uniform(5, ext - 5, (N, 2))
        ang = rng.uniform(0, 2 * np.pi, N)
        vel = spd * np.stack([np.cos(ang), np.sin(ang)], 1)

    waypoints = rng.uniform(5, ext - 5, (N, 2)) if kind == "waypoint" else None
    if kind == "patrol":
        # rectangular loops: per-robot corner lists visited cyclically
        c0 = rng.uniform(5, ext / 2, (N, 2))
        wh = rng.uniform(8, ext / 2 - 5, (N, 2))
        corners = np.stack([c0, c0 + [1, 0] * wh, c0 + wh, c0 + [0, 1] * wh], 1)
        wp_idx = np.zeros(N, int)

    for t in range(T):
        if kind in ("transect", "convoy"):
            pos = pos + vel * dt
            if kind == "transect":
                pos, vel = _bounce(pos, vel, 2.0, ext - 2.0)
            else:
                pos = np.clip(pos, 2.0, ext - 2.0)
        else:
            tgt = waypoints if kind == "waypoint" else corners[np.arange(N), wp_idx]
            d = tgt - pos
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            arrived = dist[:, 0] < 0.5
            vel = spd * d / np.maximum(dist, 1e-6)
            pos = pos + vel * dt
            if kind == "waypoint":
                waypoints[arrived] = rng.uniform(5, ext - 5, (arrived.sum(), 2))
            else:
                wp_idx[arrived] = (wp_idx[arrived] + 1) % 4
        X[t, :, :2] = pos
        X[t, :, 2] = np.arctan2(vel[:, 1], vel[:, 0])

    # odometry increments in the body frame: u = [R(-th_t)(p_{t+1}-p_t), dtheta]
    dp = X[1:, :, :2] - X[:-1, :, :2]
    th = X[:-1, :, 2]
    c, s = np.cos(th), np.sin(th)
    U = np.zeros((T - 1, N, 3), np.float32)
    U[:, :, 0] = c * dp[:, :, 0] + s * dp[:, :, 1]
    U[:, :, 1] = -s * dp[:, :, 0] + c * dp[:, :, 1]
    dth = X[1:, :, 2] - X[:-1, :, 2]
    U[:, :, 2] = np.arctan2(np.sin(dth), np.cos(dth))
    return X, U


def pe_score(X, wld, cfg):
    """Persistent-excitation diagnostic: link shadow-boundary crossings + heading
    diversity. Makes the convoy failure case attributable (Assumption PE)."""
    from coopsim.field import line_integral_links
    import jax.numpy as jnp
    G = wld.F_true.shape[0]
    sub = X[::10]                                 # every second
    N = X.shape[1]
    blocked_prev, crossings = None, 0
    for t in range(sub.shape[0]):
        li = np.asarray(line_integral_links(jnp.asarray(sub[t, :, :2]),
                                            jnp.asarray(wld.F_true), cfg.world.extent))
        blocked = li > cfg.detection.los_gate_db
        if blocked_prev is not None:
            crossings += int((blocked != blocked_prev)[np.triu_indices(N, 1)].sum())
        blocked_prev = blocked
    th = X[:, :, 2].ravel()
    heading_div = 1.0 - np.hypot(np.cos(th).mean(), np.sin(th).mean())
    return {"crossings": crossings, "heading_diversity": float(heading_div)}


def gen_targets(key, cfg):
    """Targets tau (T,Mt,4) [x,y,vx,vy] and alive (T,Mt) bool. GT label = column."""
    rng = _np_rng(key)
    T, Mt = cfg.time.steps, cfg.shapes.max_targets
    ext, dt = cfg.world.extent, cfg.time.dt
    regime = cfg.targets.regime
    n_nominal = 1 if regime in ("R1", "R3") else 8
    moving = regime in ("R3", "R4")
    p_surv = cfg.targets.survival ** dt

    tau = np.zeros((T, Mt, 4), np.float32)
    alive = np.zeros((T, Mt), bool)
    state = np.zeros((Mt, 4), np.float32)
    live = np.zeros(Mt, bool)
    next_slot = 0

    def spawn(k):
        nonlocal next_slot
        if next_slot >= Mt:
            return
        state[next_slot, :2] = rng.uniform(5, ext - 5, 2)
        if moving:
            ang = rng.uniform(0, 2 * np.pi)
            state[next_slot, 2:] = rng.uniform(0.3, 1.5) * np.array([np.cos(ang), np.sin(ang)])
        live[next_slot] = True
        next_slot += 1

    for _ in range(n_nominal):
        spawn(0)
    for t in range(T):
        if moving:
            acc = rng.normal(0, cfg.targets.sigma_a, (Mt, 2)) * dt
            state[:, 2:] += acc
            spd = np.linalg.norm(state[:, 2:], axis=1, keepdims=True)
            state[:, 2:] *= np.minimum(1.0, cfg.targets.speed_clamp / np.maximum(spd, 1e-6))
            state[:, :2] += state[:, 2:] * dt
            for c in (0, 1):
                out = (state[:, c] < 2) | (state[:, c] > ext - 2)
                state[out, c + 2] *= -1
                state[:, c] = np.clip(state[:, c], 2, ext - 2)
        deaths = live & (rng.random(Mt) > p_surv)
        live[deaths] = False
        while live.sum() < n_nominal and next_slot < Mt:
            spawn(t)
        tau[t], alive[t] = state, live
    return tau, alive
