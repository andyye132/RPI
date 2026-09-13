"""Gauge-aware link scheduling: choose which chords to measure so the accumulated
channel information stays well conditioned on the gauge subspace.

The schedule, not the trajectory, is the actuator. Theorem 1 bounds the gauge
error below for any estimator treating the schedule and delivery pattern as
exogenous; a scheduler driven by channel observables steps outside that
hypothesis, and it leaves trajectories and fading keys untouched so paired CRN
survives.

The objective is NOT tr(Sigma^G): that block is dominated by the sliding-window
prior, and adding task factors lowers it through the Schur complement with no
gauge information collected. Built only from channel-factor blocks restricted to
the gauge subspace,

    S_k = sum_{s in window} G_n(x_s)^T M^c_s G_n(x_s)          (3x3, PSD)

the schedule maximizes the predicted increment to logdet(S + eps I). logdet
rather than lambda_min, whose greedy increments are frequently exactly zero,
leaving no gradient to follow; lambda_min stays the reported diagnostic.

A link's gauge sensitivity is three autodiff directions through the same
`field.emission` the estimator uses,

    s_ij = d/dxi emission(g_xi p_i, g_xi p_j, Fbar) |_{xi=0}    (3,)

at the CURRENT FIELD ESTIMATE and ESTIMATED poses. No true field, no true poses,
no realized RSSI.
"""
import jax
import jax.numpy as jnp

from coopsim import field as fieldmod


def gauge_dirs(P, base, rho0, n):
    """Column-normalized gauge generators restricted to position, at fixed base.

    Returns (N,2,3): for robot i, column k, the planar velocity induced by unit
    gauge coordinate k. Translations are 1/sqrt(N); rotation is about `base`
    scaled by rho0. A FIXED base is required: recentring on the team centroid
    each call gives a fixed se(2) element time-varying coordinates, so a window
    sum would not be a sum in one basis.
    """
    Q = P - base
    d = jnp.zeros((n, 2, 3))
    d = d.at[:, 0, 0].set(1.0 / jnp.sqrt(n))
    d = d.at[:, 1, 1].set(1.0 / jnp.sqrt(n))
    d = d.at[:, 0, 2].set(-Q[:, 1] / rho0)
    d = d.at[:, 1, 2].set(Q[:, 0] / rho0)
    return d


def make_link_sens(ext, p0, gamma, n, ii, jj):
    """Jitted (F, P, base, rho0) -> (L,3) gauge sensitivity per DIRECTED link.

    Indexing matches scheduler.link_index: l = i*n + j, L = n*n, so the result
    lines up elementwise with the conflict adjacency and candidate mask the
    scheduler already builds. Self-links (i == j) are degenerate and are zeroed;
    the caller's `cand` mask excludes them anyway.

    Row l is d/dxi of link l's emission along the three gauge directions, built
    by the chain rule from the endpoint gradients so one autodiff pass serves all
    three columns.
    """
    same = jnp.asarray(ii == jj)

    def one(Ff, P, base, rho0):
        g2 = jax.grad(lambda a, q: fieldmod.emission(a, q, Ff, ext, P0=p0,
                                                     gamma=gamma),
                      argnums=(0, 1))
        Pi, Pj = P[ii], P[jj]
        gi, gj = jax.vmap(g2)(Pi, Pj)                    # (L,2) each
        d = gauge_dirs(P, base, rho0, n)                 # (N,2,3)
        # chain rule: both endpoints move together under a scene-wide gauge shift
        s = jnp.einsum("lc,lck->lk", gi, d[ii]) + jnp.einsum("lc,lck->lk", gj, d[jj])
        return jnp.where(same[:, None], 0.0, s)
    return jax.jit(one)


def make_chord_coverage(ext, grid, n, ii, jj, nk=24):
    """Jitted (P, live) -> (V,) voxel visitation weight for the team's chord set.

    Field reconstruction is a tomography problem: the voxel Gram is W = A^T R^-1 A
    with A the chord-to-voxel weight rows, and the field is recoverable only where
    chords cross. Measured, the chord operator has rank 1394 of 1600 voxels at
    N = 8, so most of the map is under-determined and the estimated-field error
    is 4x the supplied-field floor. That deficiency is the thing a trajectory can
    actually move, and unlike the gauge it is NOT blocked by Proposition 3: the
    joint (X,F) gauge makes the FRAME unrecoverable, it does not make the field
    shape unrecoverable.

    Returns, per voxel, the Fresnel-weighted mass this configuration's chords
    deposit. Accumulated over time this is the diagonal of the voxel Gram, which
    is the cheap surrogate for its conditioning.
    """
    u = jnp.linspace(0.0, 1.0, nk)
    rho = jnp.abs(2.0 * u - 1.0)
    wk = jax.nn.sigmoid(8.0 * (1.0 - rho ** 2))
    wk = wk / wk.sum()

    def one(P, live):
        Pi, Pj = P[ii], P[jj]                                   # (L,2)
        pts = Pi[:, None, :] * (1.0 - u)[None, :, None] \
            + Pj[:, None, :] * u[None, :, None]                 # (L,nk,2)
        gx = jnp.clip(pts[..., 0] / ext * (grid - 1), 0, grid - 1)
        gy = jnp.clip(pts[..., 1] / ext * (grid - 1), 0, grid - 1)
        idx = (jnp.round(gy).astype(jnp.int32) * grid
               + jnp.round(gx).astype(jnp.int32))               # (L,nk)
        seg = jnp.linalg.norm(Pj - Pi, axis=-1)                 # (L,)
        mass = (live * seg)[:, None] * wk[None, :]              # (L,nk)
        return jnp.zeros(grid * grid).at[idx.reshape(-1)].add(mass.reshape(-1))

    return jax.jit(one)


def diversity_gain(cov_prev, cov_new, floor=1e-3):
    """Marginal value of adding cov_new to the accumulated coverage cov_prev.

    Concave in coverage, so a voxel already crossed many times is worth little
    and an unvisited one is worth a lot. This is what makes the objective seek
    DIVERSITY rather than simply more chord length: sum log(1 + c) is maximized
    by spreading mass across voxels, whereas sum c is maximized by flying
    anywhere at all. The distinction is exactly what the effort-matched random
    arm is there to test.
    """
    return jnp.sum(jnp.log1p((cov_prev + cov_new) / floor)
                   - jnp.log1p(cov_prev / floor))


def link_weights(pr_hat, rssi_var0, decode_dbm, sigma_f, ii, jj):
    """Per-directed-link Fisher weight on the gauge sensitivity s_l.

    Two pathways contribute, and measurement showed the second dominates: on this
    configuration the decode bit carries 0.89 of the total gauge information and
    the RSSI residual only 0.11, because most attempts fail and a failed attempt
    still constrains the field along its Fresnel zone.

      RSSI, on delivered links      1 / sigma_rssi^2
      decode bit, on every attempt  [phi(z)/sigma_f]^2 / (p (1-p)),  z = (pr - theta)/sigma_f

    Both act on the SAME direction s_l, since each is a function of received
    power and dP_r/dxi = s_l, so only the scalar weight differs. The Bernoulli
    term is maximized at p = 1/2, i.e. at links right at the decode threshold:
    the most informative link to schedule is a marginal one, not a strong one.
    That is the opposite of what a delivery-maximizing scheduler does.
    """
    z = (pr_hat - decode_dbm) / sigma_f
    p = 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    phi = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2.0 * jnp.pi)
    w_bit = (phi / sigma_f) ** 2 / (p * (1.0 - p))
    w_rssi = p / rssi_var0                      # expected delivery times precision
    return (w_bit + w_rssi).reshape(-1), p.reshape(-1)


def relative_value(p, age, lam_rel, lam_age):
    """Value of a link BEYOND its gauge information.

    A delivered packet carries two things: the channel measurement, priced by
    link_weights, and the relative-pose measurement, which is what pins the
    team's shape. Pricing only the first is actively harmful here. The decode-bit
    Fisher term peaks at p = 1/2, so a gauge-only objective prefers marginal
    links; those deliver least often, and the relative-measurement graph is
    already starved at ~0.5 usable measurements per slot across 28 pairs, so
    starving it further costs more shape error than the gauge information is
    worth. Measured: gauge-only selection dropped delivery 0.010 -> 0.008 and
    raised BOTH error components.

    `lam_rel` prices expected delivery (the relative measurement arrives only if
    the packet does) and `lam_age` restores the fairness that round-robin
    provides for free, keeping every pair visited rather than concentrating on a
    few informative chords.
    """
    return lam_rel * p + lam_age * age


def greedy_gauge_links(sens, wgt, adj, cand, budget, S_prev, eps=1e-9, bonus=None):
    """Greedy maximization of logdet(S_prev + sum_chosen w_m s_m s_m^T + eps I)
    subject to the MWIS conflict structure already used by the scheduler.

    sens (L,3) gauge sensitivity, wgt (L,) expected delivery weight, adj (L,L)
    conflict adjacency, cand (L,) admissible mask, S_prev (3,3) accumulated.
    Returns (L,) bool chosen.
    """
    L = sens.shape[0]

    def body(_, carry):
        chosen, blocked, S = carry
        Sinv = jnp.linalg.inv(S + eps * jnp.eye(3))
        # rank-one logdet increment: log(1 + w * s^T Sinv s)
        q = jnp.einsum("lk,kj,lj->l", sens, Sinv, sens)
        gain = jnp.log1p(jnp.maximum(wgt * q, 0.0))
        if bonus is not None:
            gain = gain + bonus          # delivery and fairness, see relative_value
        w = jnp.where(cand & ~blocked & ~chosen, gain, -jnp.inf)
        k = jnp.argmax(w)
        take = w[k] > -jnp.inf
        chosen = chosen.at[k].set(chosen[k] | take)
        blocked = jnp.where(take, blocked | adj[k], blocked)
        S = S + jnp.where(take, 1.0, 0.0) * wgt[k] * jnp.outer(sens[k], sens[k])
        return chosen, blocked, S

    init = (jnp.zeros(L, bool), jnp.zeros(L, bool), S_prev)
    chosen, _, _ = jax.lax.fori_loop(0, budget, body, init)
    return chosen


def window_update(S_prev, sens, wgt, chosen, decay):
    """Trailing-window accumulation of the realized gauge Gram. `decay` is the
    geometric forgetting factor standing in for a hard window, so the state is
    3x3 rather than a ring buffer of slots."""
    add = jnp.einsum("l,lk,lj->kj", jnp.where(chosen, wgt, 0.0), sens, sens)
    return decay * S_prev + add
