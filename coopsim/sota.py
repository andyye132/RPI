"""SOTA pose-estimator rows: invariant-EKF cooperative localization and Gaussian
belief propagation (Robot-Web style), each standalone (odometry + relative only;
the drift theorem says they must drift) and hybridized (AETHER channel factors
attached, oracle field; estimator-independence of the bound).

InEKF: retraction-based error-state EKF on SE(2)^N (X = X_hat o exp(xi)), which for
group-affine odometry and between-measurements is the right-invariant EKF; exact
discrete Jacobians via jacfwd through the retraction. Outliers handled by chi-square
gating (the published configuration; no DCS).

GBP: per-slot synchronous Gaussian belief propagation over current poses with prior
factors from the propagated previous marginals, linearized relative factors, and
(hybridized) linearized channel unaries. Fixed iteration count; at convergence it
matches the centralized solve, which is the published claim.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coopsim import factors, se2


class EKF(NamedTuple):
    X: jax.Array          # (N,3)
    P: jax.Array          # (3N,3N)


def _prop_jacs(x, u):
    F = jax.jacfwd(lambda xi_t: se2.local(se2.compose(x, u),
                                          se2.compose(se2.retract(x, xi_t), u)))(
        jnp.zeros(3))
    G = jax.jacfwd(lambda w: se2.local(se2.compose(x, u),
                                       se2.compose(x, se2.compose(u, w))))(
        jnp.zeros(3))
    return F, G


def ekf_predict(s: EKF, U, q_diag):
    N = s.X.shape[0]
    Fs, Gs = jax.vmap(_prop_jacs)(s.X, U)
    X_new = jax.vmap(se2.compose)(s.X, U)
    Fb = jax.scipy.linalg.block_diag(*[Fs[i] for i in range(N)])
    Q = jax.scipy.linalg.block_diag(*[Gs[i] @ jnp.diag(q_diag) @ Gs[i].T
                                      for i in range(N)])
    return EKF(X=X_new, P=Fb @ s.P @ Fb.T + Q)


def _rel_jac(X, i, j, z):
    def r(xi):
        Xi = se2.retract(X[i], xi[:3])
        Xj = se2.retract(X[j], xi[3:])
        return se2.log(se2.between(z, se2.between(Xi, Xj)))
    r0 = r(jnp.zeros(6))
    H6 = jax.jacfwd(r)(jnp.zeros(6))
    return r0, H6


def ekf_update_rel(s: EKF, i, j, z, r_diag, gate=11.34):
    N = s.X.shape[0]
    r0, H6 = _rel_jac(s.X, i, j, z)
    H = jnp.zeros((3, 3 * N))
    H = H.at[:, 3 * i:3 * i + 3].set(H6[:, :3]).at[:, 3 * j:3 * j + 3].set(H6[:, 3:])
    S = H @ s.P @ H.T + jnp.diag(r_diag)
    mahal = r0 @ jnp.linalg.solve(S, r0)
    K = s.P @ H.T @ jnp.linalg.inv(S)
    ok = mahal < gate
    dx = jnp.where(ok, -(K @ r0), 0.0)
    X_new = jax.vmap(se2.retract)(s.X, dx.reshape(N, 3))
    P_new = jnp.where(ok, (jnp.eye(3 * N) - K @ H) @ s.P, s.P)
    return EKF(X=X_new, P=P_new)


def ekf_update_scalar(s: EKF, h_fn, z, var, gate=9.0, step_cap=0.3):
    """Generic scalar measurement z = h(X) + v for the hybridized channel factors,
    with the same bounded-influence guards the joint estimator uses: Huber variance
    inflation on the normalized innovation and a per-update step cap (the RSSI
    likelihood is multi-modal in position; unguarded filters basin-hop)."""
    N = s.X.shape[0]
    def r(xi):
        Xp = jax.vmap(se2.retract)(s.X, xi.reshape(N, 3))
        return h_fn(Xp) - z
    r0 = r(jnp.zeros(3 * N))
    H = jax.jacfwd(r)(jnp.zeros(3 * N))[None, :]
    S0 = (H @ s.P @ H.T)[0, 0] + var
    huber = jnp.maximum(1.0, jnp.abs(r0) / (2.0 * jnp.sqrt(S0)))
    var_r = var * huber ** 2
    S = (H @ s.P @ H.T)[0, 0] + var_r
    ok = r0 ** 2 / S < gate
    K = (s.P @ H.T / S)[:, 0]
    dx = jnp.where(ok, -K * r0, 0.0)
    step = jnp.sqrt((dx.reshape(N, 3)[:, :2] ** 2).sum(-1)).max()
    dx = dx * jnp.minimum(1.0, step_cap / jnp.maximum(step, 1e-9))
    X_new = jax.vmap(se2.retract)(s.X, dx.reshape(N, 3))
    P_new = jnp.where(ok, s.P - jnp.outer(K, K) * S, s.P)
    return EKF(X=X_new, P=P_new)


# ---- Gaussian belief propagation (per-slot, synchronous, linearized) ----

class GBPFactor(NamedTuple):
    i: int
    j: int
    Lam: jax.Array        # (6,6)
    eta: jax.Array        # (6,)


def gbp_solve(prior_lam, prior_eta, pair_factors, n, iters=20):
    """prior_lam (N,3,3), prior_eta (N,3); pair_factors list of GBPFactor.
    Returns marginal means (N,3) in the linearization tangent."""
    msgs_lam = [jnp.zeros((3, 3)) for _ in range(2 * len(pair_factors))]
    msgs_eta = [jnp.zeros(3) for _ in range(2 * len(pair_factors))]
    for _ in range(iters):
        new_lam, new_eta = [], []
        for k, f in enumerate(pair_factors):
            for side, (a, b) in enumerate(((f.i, f.j), (f.j, f.i))):
                # belief at b minus this factor's message to b
                bl = prior_lam[b] + sum(msgs_lam[2 * m + s] for m, g in
                                        enumerate(pair_factors) for s, tgt in
                                        enumerate((g.i, g.j)) if tgt == b
                                        and not (m == k and s == 1 - side))
                be = prior_eta[b] + sum(msgs_eta[2 * m + s] for m, g in
                                        enumerate(pair_factors) for s, tgt in
                                        enumerate((g.i, g.j)) if tgt == b
                                        and not (m == k and s == 1 - side))
                ia = slice(0, 3) if a == f.i else slice(3, 6)
                ib = slice(3, 6) if a == f.i else slice(0, 3)
                Laa = f.Lam[ia, ia]
                Lab = f.Lam[ia, ib]
                Lbb = f.Lam[ib, ib] + bl
                sol = jnp.linalg.solve(Lbb + 1e-9 * jnp.eye(3),
                                       jnp.concatenate(
                                           [Lab.T, (f.eta[ib] + be)[:, None]], 1))
                new_lam.append(Laa - Lab @ sol[:, :3])
                new_eta.append(f.eta[ia] - Lab @ sol[:, 3])
        msgs_lam, msgs_eta = new_lam, new_eta
    means, lams = [], []
    for v in range(n):
        L = prior_lam[v] + sum(msgs_lam[2 * m + s] for m, g in
                               enumerate(pair_factors) for s, tgt in
                               enumerate((g.i, g.j)) if tgt == v)
        e = prior_eta[v] + sum(msgs_eta[2 * m + s] for m, g in
                               enumerate(pair_factors) for s, tgt in
                               enumerate((g.i, g.j)) if tgt == v)
        means.append(jnp.linalg.solve(L + 1e-9 * jnp.eye(3), e))
        lams.append(L)
    return jnp.stack(means), jnp.stack(lams)
