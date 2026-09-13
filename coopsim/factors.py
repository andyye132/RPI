"""Factor residuals and Jacobians for the joint pose-field estimator (jit, vmapped).

Convention: residual e = h(v) - z, normal equations A += J'WJ, b -= J'We, so the GN
step is v += solve(A, b). Pose factors use SE(2) log-residuals (jacfwd through se2);
channel factors share one emission Jacobian per attempted link between the RSSI factor
and the probit outcome factor (iterated Laplace pseudo-measurement).
"""
import jax
import jax.numpy as jnp

from coopsim import se2
from coopsim.field import emission


def odom_factors(X, U):
    """X (W,N,3), U (W-1,N,3) measured body-frame increments.
    Returns e (M,3), Ja, Jb (M,3,3) with M = (W-1)*N, factor m = (t, i)."""
    h = lambda a, b, u: se2.log(se2.between(se2.compose(a, u), b))
    def one(a, b, u):
        return h(a, b, u), jax.jacfwd(h, 0)(a, b, u), jax.jacfwd(h, 1)(a, b, u)
    A = X[:-1].reshape(-1, 3)
    B = X[1:].reshape(-1, 3)
    Z = U.reshape(-1, 3)
    return jax.vmap(one)(A, B, Z)


def relpose_factors(X, Z):
    """X (W,N,3), Z (W,N,N,3) measured relative poses i->j.
    Returns e (M,3), Ji, Jj (M,3,3), M = W*N*N (mask applied by caller)."""
    h = lambda xi, xj, z: se2.log(se2.between(z, se2.between(xi, xj)))
    def one(xi, xj, z):
        return h(xi, xj, z), jax.jacfwd(h, 0)(xi, xj, z), jax.jacfwd(h, 1)(xi, xj, z)
    W, N = X.shape[:2]
    Xi = jnp.broadcast_to(X[:, :, None, :], (W, N, N, 3)).reshape(-1, 3)
    Xj = jnp.broadcast_to(X[:, None, :, :], (W, N, N, 3)).reshape(-1, 3)
    return jax.vmap(one)(Xi, Xj, Z.reshape(-1, 3))


def dcs_weights(e, W_diag, phi):
    """Dynamic covariance scaling: s = min(1, 2 phi / (phi + chi2)) per factor."""
    chi2 = (e ** 2 * W_diag[None, :]).sum(-1)
    return jnp.minimum(1.0, 2.0 * phi / (phi + chi2))


def emission_factors(X, F, extent, p0, gamma):
    """Emission value and Jacobians for every (t, i, j). X (W,N,3), F (G,G).
    Returns pr (M,), Ji, Jj (M,2) position Jacobians, JF (M, G*G), M = W*N*N."""
    W, N = X.shape[:2]
    P = X[..., :2]
    Pi = jnp.broadcast_to(P[:, :, None, :], (W, N, N, 2)).reshape(-1, 2)
    Pj = jnp.broadcast_to(P[:, None, :, :], (W, N, N, 2)).reshape(-1, 2)
    def one(pi, pj):
        val, grads = jax.value_and_grad(
            lambda a, b, Ff: emission(a, b, Ff, extent, P0=p0, gamma=gamma),
            argnums=(0, 1, 2))(pi, pj, F)
        return val, grads[0], grads[1], grads[2].reshape(-1)
    return jax.vmap(one)(Pi, Pj)


def probit_pseudo(pr_lin, success, theta, sigma_f):
    """Iterated-Laplace Gaussian pseudo-measurement for a success/failure bit.
    Returns (e_pseudo, w): A += w J'J, b -= w J' e_pseudo. For l(x) = log Phi(x):
    l' = phi/Phi, l'' = -l'(x + l')."""
    s = jnp.where(success, 1.0, -1.0)
    m = s * (pr_lin - theta) / sigma_f
    lprime = jnp.exp(jax.scipy.stats.norm.logpdf(m) - jax.scipy.stats.norm.logcdf(m))
    g = s * lprime / sigma_f                       # d logPhi / d pr
    w = jnp.maximum(lprime * (m + lprime) / sigma_f ** 2, 1e-8)
    return -g / w, w


def truncation_correction(pr_lin, theta, sigma):
    """Success-conditioned RSSI truncation at the decode threshold.
    Returns (shift, jfac): E[rssi | success] = pr_lin + shift with shift =
    sigma * phi(s)/Phi(s) at margin s = (pr_lin - theta)/sigma, and the first-order
    Jacobian factor jfac = 1 - r(s + r) in (0, 1] that vanishes for
    truncation-dominated observations."""
    s = (pr_lin - theta) / sigma
    r = jnp.exp(jax.scipy.stats.norm.logpdf(s) - jax.scipy.stats.norm.logcdf(s))
    shift = sigma * r
    jfac = jnp.clip(1.0 - r * (s + r), 0.0, 1.0)
    return shift, jfac


def gauge_generators(X, slot_mask):
    """Three scene gauge modes over the valid window poses: tx, ty, rotation about
    the centroid. X (W,N,3). Returns Gg (3, W*N*3)."""
    W, N = X.shape[:2]
    valid = jnp.broadcast_to(slot_mask[:, None], (W, N)).reshape(-1)   # (W*N,)
    px, py = X[..., 0].reshape(-1), X[..., 1].reshape(-1)
    cx = (px * valid).sum() / jnp.maximum(valid.sum(), 1.0)
    cy = (py * valid).sum() / jnp.maximum(valid.sum(), 1.0)
    zeros = jnp.zeros_like(px)
    ones = jnp.where(valid, 1.0, 0.0)
    g_tx = jnp.stack([ones, zeros, zeros], -1).reshape(-1)
    g_ty = jnp.stack([zeros, ones, zeros], -1).reshape(-1)
    g_rot = jnp.stack([-(py - cy) * ones, (px - cx) * ones, ones], -1).reshape(-1)
    return jnp.stack([g_tx, g_ty, g_rot])
