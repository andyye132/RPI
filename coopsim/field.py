"""Differentiable radio-attenuation field model (AETHER).

A static log-attenuation field F over a 2D arena. Shadowing on a link is the
Fresnel-weighted line integral of F between the two endpoints. Differentiable in both
endpoints and in F (jax.grad / jacfwd), so the same function is the simulator shadowing
model (on F_true) and the estimator measurement model (on F_hat).
"""
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

K = 64                       # samples along the segment
FRESNEL_K = 8.0             # sharpness of the along-path Fresnel emphasis


def _safe_norm(v):
    # jnp.linalg.norm has a NaN gradient at exactly zero (0/0); this form does not
    return jnp.sqrt((v ** 2).sum() + 1e-12)


def line_integral(xi, xj, F, extent):
    """Fresnel-weighted line integral of F along segment xi->xj. xi,xj: (2,). F: (H,W)."""
    Hh, Ww = F.shape
    u = jnp.linspace(0.0, 1.0, K)
    pts = xi[None, :] * (1.0 - u)[:, None] + xj[None, :] * u[:, None]     # (K,2)
    gx = pts[:, 0] / extent * (Ww - 1)
    gy = pts[:, 1] / extent * (Hh - 1)
    Fs = map_coordinates(F, [gy, gx], order=1, mode="constant", cval=0.0)  # (K,)
    rho = jnp.abs(2.0 * u - 1.0)                                          # 0 mid, 1 at ends
    w = jax.nn.sigmoid(FRESNEL_K * (1.0 - rho ** 2))
    seg = _safe_norm(xj - xi)
    return jnp.sum(w * Fs) * seg / jnp.sum(w)


def make_field(extent=50.0, res=40, obstacles=None, atten=12.0):
    """Rasterize rectangular attenuators (Manhattan boxes) into a log-attenuation grid."""
    import numpy as np
    F = np.zeros((res, res), np.float32)
    if obstacles is None:
        obstacles = [(12, 18, 8, 34), (28, 34, 14, 40), (20, 26, 4, 12)]  # (x0,x1,y0,y1) meters
    for (x0, x1, y0, y1) in obstacles:
        i0, i1 = int(y0 / extent * res), int(y1 / extent * res)
        j0, j1 = int(x0 / extent * res), int(x1 / extent * res)
        F[i0:i1, j0:j1] = atten
    return jnp.asarray(F)


# emission model: received power (dB) for a link, given poses, field, regime offset
def emission(xi, xj, F, extent, P0=0.0, gamma=2.2, h_s=0.0):
    d = _safe_norm(xj - xi)
    return P0 - 10.0 * gamma * jnp.log10(d) - line_integral(xi, xj, F, extent) + h_s


def line_integral_links(X, F, extent):
    """Line integrals for all ordered robot pairs. X: (N,2) positions. Returns (N,N)."""
    f = lambda xi, xj: line_integral(xi, xj, F, extent)
    return jax.vmap(lambda xi: jax.vmap(lambda xj: f(xi, xj))(X))(X)


# field prior helpers (estimator side): Huber-TV smoothness + per-voxel Gaussian prior
def tv_residuals(F):
    """First-difference residuals: vertical (G-1,W) and horizontal (H,G-1)."""
    return F[1:, :] - F[:-1, :], F[:, 1:] - F[:, :-1]


def huber_weights(r, delta=3.0):
    """IRLS weights for the Huber penalty: 1 inside |r|<=delta, delta/|r| outside."""
    return jnp.minimum(1.0, delta / jnp.maximum(jnp.abs(r), 1e-9))
