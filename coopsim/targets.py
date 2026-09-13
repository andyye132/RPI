"""Standard Vo-Ma GM-PHD filter (jit, fixed shapes, per robot; vmap over robots).

Association-free PHD update: misdetection row plus the dense (J x D) candidate grid,
then prune / greedy merge / per-label cap / top-J truncation, all masked at fixed
shape. Components carry an integer label inherited through predict, update, and merge
(heaviest member wins) purely so IDF1 can be scored; there is no identity feedback
(identity_reweight is a no-op hook for the later identity-informed variant).

Conventions: state [x, y, vx, vy] in the world frame; measurements are world-frame
(z, R) positions (fusion.dets_to_world converts range-bearing); merge gate is on
SQUARED Mahalanobis distance (cfg.gmphd.merge_mahal = 4.0, i.e. 2 sigma); survival is
per second, applied as survival**dt per slot (matches world.gen_targets).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

H_POS = jnp.zeros((2, 4)).at[0, 0].set(1.0).at[1, 1].set(1.0)


class PHD(NamedTuple):
    w: jax.Array          # (J,)
    m: jax.Array          # (J,4)
    P: jax.Array          # (J,4,4)
    label: jax.Array      # (J,) int32
    active: jax.Array     # (J,) bool
    next_label: jax.Array # () int32


def init(j_max, robot_id):
    return PHD(w=jnp.zeros(j_max), m=jnp.zeros((j_max, 4)),
               P=jnp.tile(jnp.eye(4), (j_max, 1, 1)),
               label=jnp.zeros(j_max, jnp.int32), active=jnp.zeros(j_max, bool),
               next_label=jnp.int32(robot_id << 20))


def predict(phd: PHD, dt, sigma_a, survival):
    F = jnp.eye(4).at[0, 2].set(dt).at[1, 3].set(dt)
    q = sigma_a ** 2
    Q = q * jnp.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                       [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                       [dt ** 3 / 2, 0, dt ** 2, 0],
                       [0, dt ** 3 / 2, 0, dt ** 2]])
    return phd._replace(w=phd.w * survival ** dt,
                        m=phd.m @ F.T,
                        P=F @ phd.P @ F.T + Q)


def identity_reweight(phd: PHD):
    """No-op hook: the identity-informed variant attaches here later."""
    return phd


def update(phd: PHD, z, R, mask, p_d, kappa, cfg_gmphd, v_init_std=1.5):
    """z (D,2), R (D,2,2) world-frame, mask (D,), p_d (J,) per component."""
    J, D = phd.w.shape[0], z.shape[0]
    eta = phd.m @ H_POS.T                                        # (J,2)
    S = H_POS @ phd.P @ H_POS.T + R[:, None] + 1e-6 * jnp.eye(2)  # (D,J,2,2)
    Sinv = jnp.linalg.inv(S)
    diff = z[:, None, :] - eta[None, :, :]                       # (D,J,2)
    mahal = jnp.einsum("dji,djik,djk->dj", diff, Sinv, diff)
    logq = -0.5 * mahal - 0.5 * jnp.linalg.slogdet(S)[1] - jnp.log(2 * jnp.pi)
    q = jnp.exp(jnp.clip(logq, -60, 60)) * phd.active[None, :]   # (D,J)

    w_cand = phd.w[None, :] * p_d[None, :] * q                   # (D,J)
    denom = kappa + w_cand.sum(1)                                # (D,)
    w_cand = jnp.where(mask[:, None], w_cand / denom[:, None], 0.0)

    K = jnp.einsum("jab,djbc->djac", phd.P @ H_POS.T, Sinv)      # (D,J,4,2)
    m_upd = phd.m[None] + jnp.einsum("djab,djb->dja", K, diff)   # (D,J,4)
    P_upd = jnp.broadcast_to(phd.P, (D, J, 4, 4)) - jnp.einsum(
        "djab,bc,jce->djae", K, H_POS, phd.P)

    # births at measurements poorly explained by existing components
    assoc_mass = w_cand.sum(1)
    birth_score = jnp.where(mask, -assoc_mass, jnp.inf)
    b_idx = jnp.argsort(birth_score)[:cfg_gmphd.max_births]
    b_ok = mask[b_idx] & (assoc_mass[b_idx] < cfg_gmphd.birth_weight)
    b_m = jnp.zeros((cfg_gmphd.max_births, 4)).at[:, :2].set(z[b_idx])
    b_P = jnp.tile(jnp.diag(jnp.array([0.5 ** 2, 0.5 ** 2,
                                       v_init_std ** 2, v_init_std ** 2])),
                   (cfg_gmphd.max_births, 1, 1))
    b_label = phd.next_label + jnp.arange(cfg_gmphd.max_births, dtype=jnp.int32)

    # stack: misdetection row, candidate grid, births
    w_all = jnp.concatenate([phd.w * (1.0 - p_d), w_cand.reshape(-1),
                             jnp.where(b_ok, cfg_gmphd.birth_weight, 0.0)])
    m_all = jnp.concatenate([phd.m, m_upd.reshape(-1, 4), b_m])
    P_all = jnp.concatenate([phd.P, P_upd.reshape(-1, 4, 4), b_P])
    lab_all = jnp.concatenate([phd.label, jnp.tile(phd.label, D), b_label])
    act_all = jnp.concatenate([phd.active, jnp.tile(phd.active, D), b_ok])
    nl = phd.next_label + b_ok.sum().astype(jnp.int32)
    return w_all, m_all, P_all, lab_all, act_all, nl


def reduce(w, m, P, label, active, next_label, cfg_gmphd, j_max):
    """Prune, greedy merge (squared-Mahalanobis gate), per-label cap, truncate to J."""
    valid = active & (w > cfg_gmphd.prune)
    Pinv = jnp.linalg.inv(P + 1e-6 * jnp.eye(4))

    def body(i, carry):
        absorbed, ow, om, oP, olab, oact = carry
        wa = jnp.where(absorbed, -1.0, w)
        j = jnp.argmax(wa)
        seed_ok = wa[j] > 0
        diff = m - m[j]
        mah = jnp.einsum("ci,cij,cj->c", diff, Pinv, diff)
        take = (~absorbed) & (mah <= cfg_gmphd.merge_mahal) & seed_ok
        wt = jnp.where(take, w, 0.0)
        ws = wt.sum()
        mbar = (wt[:, None] * m).sum(0) / jnp.maximum(ws, 1e-12)
        spread = m - mbar
        Pbar = (wt[:, None, None] * (P + spread[:, :, None] * spread[:, None, :])
                ).sum(0) / jnp.maximum(ws, 1e-12)
        ow = ow.at[i].set(jnp.where(seed_ok, ws, 0.0))
        om = om.at[i].set(mbar)
        oP = oP.at[i].set(jnp.where(seed_ok, Pbar, jnp.eye(4)))
        olab = olab.at[i].set(label[j])
        oact = oact.at[i].set(seed_ok)
        return absorbed | take, ow, om, oP, olab, oact

    init_carry = (~valid, jnp.zeros(j_max), jnp.zeros((j_max, 4)),
                  jnp.tile(jnp.eye(4), (j_max, 1, 1)),
                  jnp.zeros(j_max, jnp.int32), jnp.zeros(j_max, bool))
    _, ow, om, oP, olab, oact = jax.lax.fori_loop(0, j_max, body, init_carry)

    # cap per label: keep the top cap_per_label by weight within each label
    same = (olab[:, None] == olab[None, :]) & oact[None, :] & oact[:, None]
    heavier = (ow[None, :] > ow[:, None]) | ((ow[None, :] == ow[:, None])
                                             & (jnp.arange(j_max)[None, :] < jnp.arange(j_max)[:, None]))
    rank = (same & heavier).sum(1)
    oact = oact & (rank < cfg_gmphd.cap_per_label)
    return PHD(w=jnp.where(oact, ow, 0.0), m=om, P=oP, label=olab,
               active=oact, next_label=next_label)


def step(phd: PHD, z, R, mask, p_d, kappa, cfg_gmphd, dt, sigma_a, survival, j_max):
    phd = predict(phd, dt, sigma_a, survival)
    phd = identity_reweight(phd)
    stacked = update(phd, z, R, mask, p_d, kappa, cfg_gmphd)
    return reduce(*stacked, cfg_gmphd, j_max)


def extract(phd: PHD, extract_thresh, max_labels):
    """Best component per label with w > threshold -> (max_labels, 5), mask."""
    est, mask, _ = extract_with_cov(phd, extract_thresh, max_labels)
    return est, mask


def extract_with_cov(phd: PHD, extract_thresh, max_labels):
    """extract plus the selected components' covariances (max_labels,4,4)."""
    ok = phd.active & (phd.w > extract_thresh)
    same = phd.label[:, None] == phd.label[None, :]
    heavier = (phd.w[None, :] > phd.w[:, None]) | (
        (phd.w[None, :] == phd.w[:, None])
        & (jnp.arange(phd.w.shape[0])[None, :] < jnp.arange(phd.w.shape[0])[:, None]))
    best = ok & ~((same & heavier & ok[None, :]).any(1))
    order = jnp.argsort(jnp.where(best, -phd.w, jnp.inf))[:max_labels]
    est = jnp.concatenate([phd.m[order], phd.label[order, None].astype(jnp.float32)], -1)
    return est, best[order], phd.P[order]


def cardinality(phd: PHD):
    return jnp.where(phd.active, phd.w, 0.0).sum()
