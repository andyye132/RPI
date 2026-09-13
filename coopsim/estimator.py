"""Sliding-window Gauss-Newton over [window poses | attenuation field] (jit).

The estF fix, replacing the old ad-hoc per-slot field gradient step: the field is a
state block with a proper prior (per-voxel Gaussian + Huber-TV via IRLS), solved
jointly with the poses through the cross information block, under LM damping that
rejects cost-increasing steps (divergence becomes stall at worst), with the field
eliminated by Schur complement each iteration. Poses leaving the window are
marginalized onto [next-oldest slot, field], so channel information accumulates in the
persistent prior instead of being rediscovered per slot.

Variable layout: pose block index p(t, i) = t*N + i, flat pose dim P = W*N*3 with slot
0 occupying rows [0, 3N); field flattened row-major, dim G*G. The prior (Lam_p, mu_p)
lives over [slot-0 poses | field]. Pose parameters are treated as vectors with angle
wrap on theta residuals; factor residuals themselves use SE(2) log maps (factors.py).
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coopsim import factors, se2
from coopsim.field import huber_weights

STEP_CAP = 1.0    # max position update per pose per GN solve (m); robots move
                  # 0.15 m/slot, so estimates never need to teleport


class Win(NamedTuple):
    U: jax.Array         # (W-1,N,3) odometry increments between window slots
    Zrel: jax.Array      # (W,N,N,3)
    rel_mask: jax.Array  # (W,N,N)
    rel_trust: jax.Array # (W,N,N) A-BL scalar reweighting (ones otherwise)
    rssi: jax.Array      # (W,N,N)
    rssi_mask: jax.Array # (W,N,N)
    att: jax.Array       # (W,N,N) attempted
    succ: jax.Array      # (W,N,N)
    hbar: jax.Array      # (W,N,N) regime mean offset
    hvar: jax.Array      # (W,N,N) regime variance inflation
    msg_lam: jax.Array   # (W,3N,3N) A-HY per-slot pose-information message
    msg_eta: jax.Array   # (W,3N)
    msg_xlin: jax.Array  # (W,N,3) message linearization poses
    dp: jax.Array        # (W,N) transmit power OFFSET in dB from c.p0, per TX


class Est(NamedTuple):
    X: jax.Array         # (W,N,3)
    F: jax.Array         # (G,G)
    Lam_p: jax.Array     # (3N+G2, 3N+G2)
    mu_p: jax.Array      # (3N+G2,)
    win: Win
    slot_mask: jax.Array # (W,)
    lam_lm: jax.Array    # ()


def empty_win(W, N):
    z = jnp.zeros((W, N, N))
    return Win(U=jnp.zeros((W - 1, N, 3)), Zrel=jnp.zeros((W, N, N, 3)),
               rel_mask=jnp.zeros((W, N, N), bool), rel_trust=jnp.ones((W, N, N)),
               rssi=z, rssi_mask=jnp.zeros((W, N, N), bool),
               att=jnp.zeros((W, N, N), bool), succ=jnp.zeros((W, N, N), bool),
               hbar=z, hvar=z,
               msg_lam=jnp.zeros((W, 3 * N, 3 * N)), msg_eta=jnp.zeros((W, 3 * N)),
               msg_xlin=jnp.zeros((W, N, 3)), dp=jnp.zeros((W, N)))


def init(X0, F0, W, sigma0=0.5):
    N = X0.shape[0]
    G2 = F0.size
    Lam_p = jnp.zeros((3 * N + G2, 3 * N + G2))
    Lam_p = Lam_p.at[:3 * N, :3 * N].set(jnp.eye(3 * N) / sigma0 ** 2)
    mu_p = jnp.concatenate([X0.reshape(-1), F0.reshape(-1)])
    X = jnp.tile(X0[None], (W, 1, 1))
    mask = jnp.zeros(W, bool).at[0].set(True)
    return Est(X=X, F=F0, Lam_p=Lam_p, mu_p=mu_p, win=empty_win(W, N),
               slot_mask=mask, lam_lm=jnp.asarray(1e-3))


class SlotData(NamedTuple):
    u: jax.Array         # (N,3)
    zrel: jax.Array      # (N,N,3)
    rel_mask: jax.Array  # (N,N)
    rssi: jax.Array      # (N,N)
    rssi_mask: jax.Array # (N,N)
    att: jax.Array       # (N,N)
    succ: jax.Array      # (N,N)
    hbar: jax.Array      # (N,N)
    hvar: jax.Array      # (N,N)
    rel_trust: jax.Array # (N,N)
    msg_lam: jax.Array   # (3N,3N)
    msg_eta: jax.Array   # (3N,)
    msg_xlin: jax.Array  # (N,3)
    dp: jax.Array        # (N,) transmit power OFFSET in dB from c.p0


def plain_slot(u, zrel, rel_mask, rssi, rssi_mask, att, succ, hbar, hvar,
               dp=None):
    n = u.shape[0]
    return SlotData(u=u, zrel=zrel, rel_mask=rel_mask, rssi=rssi,
                    rssi_mask=rssi_mask, att=att, succ=succ, hbar=hbar, hvar=hvar,
                    rel_trust=jnp.ones((n, n)),
                    msg_lam=jnp.zeros((3 * n, 3 * n)), msg_eta=jnp.zeros(3 * n),
                    msg_xlin=jnp.zeros((n, 3)),
                    dp=jnp.zeros(n) if dp is None else dp)


def _dp_links(dp):
    """Per-transmitter power offset -> per-LINK, flattened to emission ordering.

    `emission_factors` returns (W,N,N) flattened in (w,i,j) order, and transmit
    power belongs to the transmitter i, so the offset is broadcast across
    receivers j. Works for dp of shape (W,N) and (N,) alike.

    Transmit power enters `field.emission` purely additively and does not depend
    on pose or field, so it shifts the predicted received power and leaves every
    Jacobian untouched. That is why this is an offset added after the fact rather
    than a parameter threaded through the autodiff, and why dp = 0 reproduces the
    fixed-power estimator exactly.
    """
    return jnp.broadcast_to(dp[..., :, None],
                            dp.shape + (dp.shape[-1],)).reshape(-1)


def _wrap3(e):
    return e.at[..., 2].set(jnp.arctan2(jnp.sin(e[..., 2]), jnp.cos(e[..., 2])))


def _insert(win: Win, d: SlotData, idx):
    return Win(U=win.U.at[idx - 1].set(d.u),
               Zrel=win.Zrel.at[idx].set(d.zrel),
               rel_mask=win.rel_mask.at[idx].set(d.rel_mask),
               rel_trust=win.rel_trust.at[idx].set(d.rel_trust),
               rssi=win.rssi.at[idx].set(d.rssi),
               rssi_mask=win.rssi_mask.at[idx].set(d.rssi_mask),
               att=win.att.at[idx].set(d.att), succ=win.succ.at[idx].set(d.succ),
               hbar=win.hbar.at[idx].set(d.hbar), hvar=win.hvar.at[idx].set(d.hvar),
               msg_lam=win.msg_lam.at[idx].set(d.msg_lam),
               msg_eta=win.msg_eta.at[idx].set(d.msg_eta),
               msg_xlin=win.msg_xlin.at[idx].set(d.msg_xlin),
               dp=win.dp.at[idx].set(d.dp))


def _roll(win: Win, d: SlotData):
    r = lambda a: jnp.roll(a, -1, axis=0)
    win = Win(*[r(f) for f in win])
    return _insert(win, d, win.U.shape[0])          # new slot at index W-1


class _Consts(NamedTuple):
    extent: float
    p0: float
    gamma: float
    theta: float          # decode threshold
    sigma_f: float        # effective outcome scale: shadowing + fading (dB)
    fade_mean: float      # E[10 log10 X], X ~ Gamma(m, 1/m); negative for small m
    w_odom: jax.Array     # (3,)
    w_rel: jax.Array      # (3,)
    dcs_phi: float
    rssi_var0: float      # fading + shadowing + quantization variance (dB^2)
    f_prior_var: float
    tv_lambda: float
    tv_delta: float
    gauge_prior: float


def make_consts(cfg):
    from scipy.special import digamma, polygamma
    import numpy as np
    m = getattr(cfg.channel.nakagami_m, cfg.world.kind)
    scale = 10.0 / np.log(10.0)
    fade_mean = scale * (digamma(m) - np.log(m))
    fade_var = scale ** 2 * polygamma(1, m) / max(int(cfg.channel.get("burst_k", 1)), 1)
    sr = jnp.deg2rad(cfg.robots.sigma_rot_deg)
    rr = jnp.deg2rad(cfg.relative.sigma_rot_deg)
    return _Consts(
        extent=cfg.world.extent, p0=cfg.channel.p0_dbm, gamma=cfg.channel.gamma,
        theta=cfg.channel.decode_dbm,
        sigma_f=float(np.sqrt(cfg.channel.sigma_f_db ** 2 + fade_var)),
        fade_mean=float(fade_mean),
        w_odom=jnp.array([1 / cfg.robots.sigma_trans ** 2,
                          1 / cfg.robots.sigma_trans ** 2, 1 / sr ** 2]),
        w_rel=jnp.array([1 / cfg.relative.sigma_trans ** 2,
                         1 / cfg.relative.sigma_trans ** 2, 1 / rr ** 2]),
        dcs_phi=cfg.relative.dcs_phi,
        rssi_var0=float(fade_var + cfg.channel.sigma_f_db ** 2
                        + cfg.channel.rssi_quant_db ** 2 / 12.0),
        f_prior_var=cfg.field.prior_sigma_db ** 2,
        tv_lambda=cfg.field.tv_lambda, tv_delta=cfg.field.tv_delta_db,
        gauge_prior=cfg.estimator.gauge_prior)


def _tv_indices(G):
    k = jnp.arange(G * G).reshape(G, G)
    v1, v2 = k[:-1, :].reshape(-1), k[1:, :].reshape(-1)
    h1, h2 = k[:, :-1].reshape(-1), k[:, 1:].reshape(-1)
    return jnp.concatenate([v1, h1]), jnp.concatenate([v2, h2])


def _assemble(X, F, est: Est, c: _Consts, use_channel, estimate_field,
              use_msgs=False):
    """Normal equations A, b over [poses | field] at linearization (X, F)."""
    W, N = X.shape[:2]
    WN, P, G2 = W * N, W * N * 3, F.size
    slot_ok = est.slot_mask
    win = est.win

    Axx = jnp.zeros((WN, WN, 3, 3))
    bx = jnp.zeros((WN, 3))
    Axf = jnp.zeros((WN, 3, G2))
    Aff = jnp.zeros((G2, G2))
    bf = jnp.zeros(G2)

    # odometry
    e_o, Ja, Jb = factors.odom_factors(X, win.U)
    m_ok = (slot_ok[:-1] & slot_ok[1:])[:, None].repeat(N, 1).reshape(-1)
    Wo = c.w_odom[None, :] * m_ok[:, None]
    t_idx = jnp.arange(W - 1)[:, None].repeat(N, 1).reshape(-1)
    i_idx = jnp.arange(N)[None, :].repeat(W - 1, 0).reshape(-1)
    ia, ib = t_idx * N + i_idx, (t_idx + 1) * N + i_idx
    JaW = Ja * Wo[:, :, None]
    JbW = Jb * Wo[:, :, None]
    Axx = Axx.at[ia, ia].add(jnp.einsum("mki,mkj->mij", JaW, Ja))
    Axx = Axx.at[ib, ib].add(jnp.einsum("mki,mkj->mij", JbW, Jb))
    Hab = jnp.einsum("mki,mkj->mij", JaW, Jb)
    Axx = Axx.at[ia, ib].add(Hab)
    Axx = Axx.at[ib, ia].add(Hab.transpose(0, 2, 1))
    bx = bx.at[ia].add(-jnp.einsum("mki,mk->mi", JaW, e_o))
    bx = bx.at[ib].add(-jnp.einsum("mki,mk->mi", JbW, e_o))

    # relative poses with DCS (rel_trust = A-BL scalar reweighting, ones otherwise)
    e_r, Ji, Jj = factors.relpose_factors(X, win.Zrel)
    pair_ok = (win.rel_mask & slot_ok[:, None, None]).reshape(-1)
    s_dcs = factors.dcs_weights(e_r, c.w_rel, c.dcs_phi)
    Wr = c.w_rel[None, :] * (pair_ok * s_dcs * win.rel_trust.reshape(-1))[:, None]
    tt = jnp.arange(W)[:, None, None]
    ii = jnp.arange(N)[None, :, None]
    jj = jnp.arange(N)[None, None, :]
    ia_r = jnp.broadcast_to(tt * N + ii, (W, N, N)).reshape(-1)
    ib_r = jnp.broadcast_to(tt * N + jj, (W, N, N)).reshape(-1)
    JiW = Ji * Wr[:, :, None]
    JjW = Jj * Wr[:, :, None]
    Axx = Axx.at[ia_r, ia_r].add(jnp.einsum("mki,mkj->mij", JiW, Ji))
    Axx = Axx.at[ib_r, ib_r].add(jnp.einsum("mki,mkj->mij", JjW, Jj))
    Hij = jnp.einsum("mki,mkj->mij", JiW, Jj)
    Axx = Axx.at[ia_r, ib_r].add(Hij)
    Axx = Axx.at[ib_r, ia_r].add(Hij.transpose(0, 2, 1))
    bx = bx.at[ia_r].add(-jnp.einsum("mki,mk->mi", JiW, e_r))
    bx = bx.at[ib_r].add(-jnp.einsum("mki,mk->mi", JjW, e_r))

    if use_channel:
        pr, Jpi, Jpj, JF = factors.emission_factors(X, F, c.extent, c.p0, c.gamma)
        pad = lambda J2: jnp.concatenate([J2, jnp.zeros((J2.shape[0], 1))], -1)
        Ji3, Jj3 = pad(Jpi), pad(Jpj)
        pr_lin = pr + _dp_links(win.dp) + win.hbar.reshape(-1) + c.fade_mean

        att_ok = (win.att & slot_ok[:, None, None]).reshape(-1)
        rssi_ok = (win.rssi_mask & slot_ok[:, None, None]).reshape(-1)
        # success-conditioned truncation: observed RSSI is Gaussian truncated at the
        # decode threshold; shift the mean by the inverse Mills ratio and scale the
        # Jacobian by the truncation information factor (kills near-threshold rows)
        shift, jfac = factors.truncation_correction(pr_lin, c.theta, c.sigma_f)
        e_rs = pr_lin + shift - win.rssi.reshape(-1)
        w_rs = jnp.where(rssi_ok, 1.0 / (c.rssi_var0 + win.hvar.reshape(-1)), 0.0)
        e_ob, w_ob = factors.probit_pseudo(pr_lin, win.succ.reshape(-1),
                                           c.theta, c.sigma_f)
        w_ob = jnp.where(att_ok, w_ob, 0.0)
        # bounded influence: mis-tracked regimes give confidently-wrong residuals
        w_rs = w_rs * huber_weights(e_rs * jnp.sqrt(w_rs), 2.0)
        w_ob = w_ob * huber_weights(e_ob * jnp.sqrt(w_ob), 2.0)
        ones = jnp.ones_like(jfac)

        for e_c, w_c, f_c in ((e_rs, w_rs, jfac), (e_ob, w_ob, ones)):
            Ji3c, Jj3c = Ji3 * f_c[:, None], Jj3 * f_c[:, None]
            Haa = jnp.einsum("m,mi,mj->mij", w_c, Ji3c, Ji3c)
            Hbb = jnp.einsum("m,mi,mj->mij", w_c, Jj3c, Jj3c)
            Hab = jnp.einsum("m,mi,mj->mij", w_c, Ji3c, Jj3c)
            Axx = Axx.at[ia_r, ia_r].add(Haa)
            Axx = Axx.at[ib_r, ib_r].add(Hbb)
            Axx = Axx.at[ia_r, ib_r].add(Hab)
            Axx = Axx.at[ib_r, ia_r].add(Hab.transpose(0, 2, 1))
            bx = bx.at[ia_r].add(-(w_c * e_c)[:, None] * Ji3c)
            bx = bx.at[ib_r].add(-(w_c * e_c)[:, None] * Jj3c)
            if estimate_field:
                JFc = JF * f_c[:, None]
                Axf = Axf.at[ia_r].add(jnp.einsum("m,mi,mk->mik", w_c, Ji3c, JFc))
                Axf = Axf.at[ib_r].add(jnp.einsum("m,mi,mk->mik", w_c, Jj3c, JFc))
                Aff = Aff + (JFc * w_c[:, None]).T @ JFc
                bf = bf - (w_c * e_c) @ JFc

    if estimate_field:
        # per-voxel Gaussian prior (mean 0) + Huber-TV smoothness (IRLS)
        Ff = F.reshape(-1)
        Aff = Aff + jnp.eye(G2) / c.f_prior_var
        bf = bf - Ff / c.f_prior_var
        k1, k2 = _tv_indices(F.shape[0])
        e_tv = Ff[k2] - Ff[k1]
        w_tv = c.tv_lambda * huber_weights(e_tv, c.tv_delta)
        Aff = Aff.at[k1, k1].add(w_tv).at[k2, k2].add(w_tv)
        Aff = Aff.at[k1, k2].add(-w_tv).at[k2, k1].add(-w_tv)
        bf = bf.at[k1].add(w_tv * e_tv).at[k2].add(-w_tv * e_tv)

    # flatten pose block, add prior over [slot 0 | field] and gauge conditioning
    A_flat = Axx.transpose(0, 2, 1, 3).reshape(P, P)
    b_flat = bx.reshape(-1)
    n3 = 3 * N

    if use_msgs:
        # A-HY: per-slot Gaussian pose-information messages (eq. P-HY)
        for t in range(W):
            sl = slice(t * n3, (t + 1) * n3)
            ok = slot_ok[t]
            de = _wrap3(X[t] - win.msg_xlin[t]).reshape(-1)
            A_flat = A_flat.at[sl, sl].add(jnp.where(ok, 1.0, 0.0) * win.msg_lam[t])
            b_flat = b_flat.at[sl].add(
                jnp.where(ok, 1.0, 0.0) * (win.msg_eta[t] - win.msg_lam[t] @ de))
    e_p = jnp.concatenate([_wrap3(X[0] - est.mu_p[:n3].reshape(N, 3)).reshape(-1),
                           F.reshape(-1) - est.mu_p[n3:]])
    g_p = est.Lam_p @ e_p
    A_flat = A_flat.at[:n3, :n3].add(est.Lam_p[:n3, :n3])
    b_flat = b_flat.at[:n3].add(-g_p[:n3])
    Gg = factors.gauge_generators(X, slot_ok)
    A_flat = A_flat + c.gauge_prior * Gg.T @ Gg
    if estimate_field:
        Axf_flat = Axf.reshape(P, G2).at[:n3, :].add(est.Lam_p[:n3, n3:])
        Aff = Aff + est.Lam_p[n3:, n3:]
        bf = bf - g_p[n3:]
    else:
        Axf_flat = Axf.reshape(P, G2)
    return A_flat, Axf_flat, Aff, b_flat, bf


def _cost(X, F, est: Est, c: _Consts, use_channel, estimate_field, use_msgs=False):
    """Weighted robust cost at (X, F) with the same weighting scheme as assembly."""
    W, N = X.shape[:2]
    win, slot_ok = est.win, est.slot_mask
    e_o, _, _ = factors.odom_factors(X, win.U)
    m_ok = (slot_ok[:-1] & slot_ok[1:])[:, None].repeat(N, 1).reshape(-1)
    cost = (m_ok * (e_o ** 2 * c.w_odom[None, :]).sum(-1)).sum()

    e_r, _, _ = factors.relpose_factors(X, win.Zrel)
    pair_ok = (win.rel_mask & slot_ok[:, None, None]).reshape(-1)
    s_dcs = factors.dcs_weights(e_r, c.w_rel, c.dcs_phi)
    cost += (pair_ok * s_dcs * win.rel_trust.reshape(-1)
             * (e_r ** 2 * c.w_rel[None, :]).sum(-1)).sum()

    if use_msgs:
        n3m = 3 * N
        for t in range(W):
            de = _wrap3(X[t] - win.msg_xlin[t]).reshape(-1)
            ok = jnp.where(slot_ok[t], 1.0, 0.0)
            cost += ok * (de @ win.msg_lam[t] @ de - 2.0 * win.msg_eta[t] @ de)

    if use_channel:
        pr, _, _, _ = factors.emission_factors(X, F, c.extent, c.p0, c.gamma)
        pr_lin = pr + _dp_links(win.dp) + win.hbar.reshape(-1) + c.fade_mean
        rssi_ok = (win.rssi_mask & slot_ok[:, None, None]).reshape(-1)
        att_ok = (win.att & slot_ok[:, None, None]).reshape(-1)
        shift, _ = factors.truncation_correction(pr_lin, c.theta, c.sigma_f)
        w_rs = jnp.where(rssi_ok, 1.0 / (c.rssi_var0 + win.hvar.reshape(-1)), 0.0)
        cost += (w_rs * (pr_lin + shift - win.rssi.reshape(-1)) ** 2).sum()
        s = jnp.where(win.succ.reshape(-1), 1.0, -1.0)
        m = s * (pr_lin - c.theta) / c.sigma_f
        cost += -2.0 * (att_ok * jax.scipy.stats.norm.logcdf(m)).sum()

    n3 = 3 * N
    e_p = jnp.concatenate([_wrap3(X[0] - est.mu_p[:n3].reshape(N, 3)).reshape(-1),
                           F.reshape(-1) - est.mu_p[n3:]])
    cost += e_p @ est.Lam_p @ e_p
    if estimate_field:
        Ff = F.reshape(-1)
        cost += (Ff ** 2).sum() / c.f_prior_var
        k1, k2 = _tv_indices(F.shape[0])
        e_tv = jnp.abs(Ff[k2] - Ff[k1])
        hub = jnp.where(e_tv <= c.tv_delta, 0.5 * e_tv ** 2,
                        c.tv_delta * (e_tv - 0.5 * c.tv_delta))
        cost += 2.0 * c.tv_lambda * hub.sum()
    return cost


def _solve(A, Axf, Aff, bx, bf, lam, estimate_field):
    P = A.shape[0]
    A_d = A + lam * jnp.diag(jnp.diag(A)) + 1e-8 * jnp.eye(P)
    if not estimate_field:
        return jnp.linalg.solve(A_d, bx), jnp.zeros(Aff.shape[0])
    G2 = Aff.shape[0]
    Aff_d = Aff + lam * jnp.diag(jnp.diag(Aff)) + 1e-6 * jnp.eye(G2)
    cf = jax.scipy.linalg.cho_factor(Aff_d)
    Y = jax.scipy.linalg.cho_solve(cf, Axf.T)          # (G2, P)
    S = A_d - Axf @ Y
    rhs = bx - Axf @ jax.scipy.linalg.cho_solve(cf, bf)
    dx = jnp.linalg.solve(S, rhs)
    df = jax.scipy.linalg.cho_solve(cf, bf - Y @ dx)
    return dx, df


def gn_update(est: Est, c: _Consts, use_channel, estimate_field, iters=4,
              use_msgs=False):
    W, N = est.X.shape[:2]

    def body(carry, _):
        X, F, lam = carry
        A, Axf, Aff, bx, bf = _assemble(X, F, est, c, use_channel, estimate_field,
                                        use_msgs)
        cost0 = _cost(X, F, est, c, use_channel, estimate_field, use_msgs)

        def attempt(lam_a):
            dx, df = _solve(A, Axf, Aff, bx, bf, lam_a, estimate_field)
            # trust region: no pose moves more than STEP_CAP per solve; scaling the
            # whole step preserves the GN direction and forbids basin hopping
            dxs = dx.reshape(W, N, 3)
            step = jnp.sqrt((dxs[..., :2] ** 2).sum(-1)).max()
            scale = jnp.minimum(1.0, STEP_CAP / jnp.maximum(step, 1e-9))
            Xc = _wrap3(X + scale * dxs)
            Fc = jnp.clip(F + scale * df.reshape(F.shape), 0.0, 30.0) \
                if estimate_field else F
            return Xc, Fc, _cost(Xc, Fc, est, c, use_channel, estimate_field,
                                 use_msgs)

        X1, F1, c1 = attempt(lam)
        X2, F2, c2 = attempt(lam * 10.0)
        use1 = c1 <= cost0
        use2 = (~use1) & (c2 <= cost0)
        Xn = jnp.where(use1, X1, jnp.where(use2, X2, X))
        Fn = jnp.where(use1, F1, jnp.where(use2, F2, F))
        lam_n = jnp.where(use1, jnp.maximum(lam * 0.7, 1e-5),
                          jnp.where(use2, lam * 10.0, lam * 100.0))
        lam_n = jnp.minimum(lam_n, 1e6)
        cn = jnp.where(use1, c1, jnp.where(use2, c2, cost0))
        return (Xn, Fn, lam_n), (cost0, cn, lam_n)

    (Xf, Ff, lamf), telem = jax.lax.scan(body, (est.X, est.F, est.lam_lm),
                                         None, length=iters)
    return est._replace(X=Xf, F=Ff, lam_lm=lamf), telem


def marginalize(est: Est, c: _Consts, use_channel, estimate_field, use_msgs=False):
    """Schur-marginalize slot 0 onto [slot 1 | field] using only the factors that
    touch slot 0 (prior, odom 0->1, relative in slot 0, channel in slot 0, and the
    slot-0 message in A-HY)."""
    W, N = est.X.shape[:2]
    n3, G2 = 3 * N, est.F.size
    X, F, win = est.X, est.F, est.win

    # A over [s0 (3N), s1 (3N), F (G2)]
    A = jnp.zeros((2 * n3 + G2, 2 * n3 + G2))
    g = jnp.zeros(2 * n3 + G2)

    e_p = jnp.concatenate([_wrap3(X[0] - est.mu_p[:n3].reshape(N, 3)).reshape(-1),
                           F.reshape(-1) - est.mu_p[n3:]])
    gp = est.Lam_p @ e_p
    A = A.at[:n3, :n3].add(est.Lam_p[:n3, :n3])
    A = A.at[:n3, 2 * n3:].add(est.Lam_p[:n3, n3:])
    A = A.at[2 * n3:, :n3].add(est.Lam_p[n3:, :n3])
    A = A.at[2 * n3:, 2 * n3:].add(est.Lam_p[n3:, n3:])
    g = g.at[:n3].add(gp[:n3]).at[2 * n3:].add(gp[n3:])

    # odometry slot0 -> slot1
    def block3(Jm, Wd):
        return jnp.einsum("mki,mk,mkj->mij", Jm, Wd, Jm)
    e_o, Ja, Jb = jax.vmap(lambda a, b, u: (
        se2.log(se2.between(se2.compose(a, u), b)),
        jax.jacfwd(lambda aa: se2.log(se2.between(se2.compose(aa, u), b)))(a),
        jax.jacfwd(lambda bb: se2.log(se2.between(se2.compose(a, u), bb)))(b)))(
        X[0], X[1], win.U[0])
    Wo = jnp.broadcast_to(c.w_odom, (N, 3))
    for r in range(N):
        s0, s1 = slice(3 * r, 3 * r + 3), slice(n3 + 3 * r, n3 + 3 * r + 3)
        A = A.at[s0, s0].add(Ja[r].T * Wo[r] @ Ja[r])
        A = A.at[s1, s1].add(Jb[r].T * Wo[r] @ Jb[r])
        A = A.at[s0, s1].add(Ja[r].T * Wo[r] @ Jb[r])
        A = A.at[s1, s0].add((Ja[r].T * Wo[r] @ Jb[r]).T)
        g = g.at[s0].add(Ja[r].T @ (Wo[r] * e_o[r]))
        g = g.at[s1].add(Jb[r].T @ (Wo[r] * e_o[r]))

    # relative + channel factors within slot 0
    X0 = X[0:1]
    e_r, Ji, Jj = factors.relpose_factors(X0, win.Zrel[0:1])
    pair_ok = win.rel_mask[0].reshape(-1)
    s_dcs = factors.dcs_weights(e_r, c.w_rel, c.dcs_phi)
    Wr = c.w_rel[None, :] * (pair_ok * s_dcs * win.rel_trust[0].reshape(-1))[:, None]
    ii = jnp.arange(N)[:, None].repeat(N, 1).reshape(-1)
    jj = jnp.arange(N)[None, :].repeat(N, 0).reshape(-1)

    def scatter_pair(A, g, ia, ib, Jia, Jib, Wm, em, off_a=0, off_b=0):
        JiWm = Jia * Wm[:, :, None]
        JjWm = Jib * Wm[:, :, None]
        idx_a, idx_b = off_a + ia * 3, off_b + ib * 3
        blk = lambda Ha: Ha.reshape(-1, 3, 3)
        for m_arr, i_arr, j_arr in (
                (jnp.einsum("mki,mkj->mij", JiWm, Jia), idx_a, idx_a),
                (jnp.einsum("mki,mkj->mij", JjWm, Jib), idx_b, idx_b),
                (jnp.einsum("mki,mkj->mij", JiWm, Jib), idx_a, idx_b),
                (jnp.einsum("mki,mkj->mij", JjWm, Jia), idx_b, idx_a)):
            rows = (i_arr[:, None] + jnp.arange(3)[None, :])
            cols = (j_arr[:, None] + jnp.arange(3)[None, :])
            A = A.at[rows[:, :, None], cols[:, None, :]].add(m_arr)
        g = g.at[(idx_a[:, None] + jnp.arange(3)[None, :]).reshape(-1)].add(
            jnp.einsum("mki,mk->mi", JiWm, em).reshape(-1))
        g = g.at[(idx_b[:, None] + jnp.arange(3)[None, :]).reshape(-1)].add(
            jnp.einsum("mki,mk->mi", JjWm, em).reshape(-1))
        return A, g

    A, g = scatter_pair(A, g, ii, jj, Ji, Jj, Wr, e_r)

    if use_msgs:
        de = _wrap3(X[0] - win.msg_xlin[0]).reshape(-1)
        A = A.at[:n3, :n3].add(win.msg_lam[0])
        g = g.at[:n3].add(win.msg_lam[0] @ de - win.msg_eta[0])

    if use_channel:
        pr, Jpi, Jpj, JF = factors.emission_factors(X0, F, c.extent, c.p0, c.gamma)
        pad = lambda J2: jnp.concatenate([J2, jnp.zeros((J2.shape[0], 1))], -1)
        Ji3, Jj3 = pad(Jpi), pad(Jpj)
        pr_lin = pr + _dp_links(win.dp[0]) + win.hbar[0].reshape(-1) + c.fade_mean
        rssi_ok = win.rssi_mask[0].reshape(-1)
        att_ok = win.att[0].reshape(-1)
        shift, jfac = factors.truncation_correction(pr_lin, c.theta, c.sigma_f)
        e_rs = pr_lin + shift - win.rssi[0].reshape(-1)
        w_rs = jnp.where(rssi_ok, 1.0 / (c.rssi_var0 + win.hvar[0].reshape(-1)), 0.0)
        e_ob, w_ob = factors.probit_pseudo(pr_lin, win.succ[0].reshape(-1),
                                           c.theta, c.sigma_f)
        w_ob = jnp.where(att_ok, w_ob, 0.0)
        w_rs = w_rs * huber_weights(e_rs * jnp.sqrt(w_rs), 2.0)
        w_ob = w_ob * huber_weights(e_ob * jnp.sqrt(w_ob), 2.0)
        ones = jnp.ones_like(jfac)
        for e_c, w_c, f_c in ((e_rs, w_rs, jfac), (e_ob, w_ob, ones)):
            Ji3c, Jj3c = Ji3 * f_c[:, None], Jj3 * f_c[:, None]
            JFc = JF * f_c[:, None]
            Wc3 = jnp.stack([w_c] * 3, -1)
            A, g = scatter_pair(A, g, ii, jj, Ji3c[:, None, :], Jj3c[:, None, :],
                                Wc3[:, :1], e_c[:, None])
            if estimate_field:
                idx_a, idx_b = ii * 3, jj * 3
                cols = 2 * n3 + jnp.arange(G2)[None, None, :]
                for J3, idx in ((Ji3c, idx_a), (Jj3c, idx_b)):
                    rows = (idx[:, None] + jnp.arange(3)[None, :])[:, :, None]
                    contrib = jnp.einsum("m,mi,mk->mik", w_c, J3, JFc)
                    A = A.at[rows, cols].add(contrib)
                    A = A.at[cols.transpose(0, 2, 1), rows.transpose(0, 2, 1)].add(
                        contrib.transpose(0, 2, 1))
                Aff_c = (JFc * w_c[:, None]).T @ JFc
                A = A.at[2 * n3:, 2 * n3:].add(Aff_c)
                g = g.at[2 * n3:].add((w_c * e_c) @ JFc)

    # Schur eliminate slot 0. The marginal mean is the CURRENT estimate (standard
    # sliding-window practice); solving the near-singular joint for a mean
    # correction injects garbage on barely-observed voxels and is what blew up the
    # first estF attempt.
    A = A + 1e-9 * jnp.eye(A.shape[0])
    A00, A0k = A[:n3, :n3], A[:n3, n3:]
    Akk = A[n3:, n3:]
    sol = jnp.linalg.solve(A00, A0k)
    Lam_new = Akk - A0k.T @ sol
    Lam_new = 0.5 * (Lam_new + Lam_new.T)
    mu_new = jnp.concatenate([X[1].reshape(-1), F.reshape(-1)])
    if not estimate_field:
        keep = jnp.zeros_like(Lam_new).at[:n3, :n3].set(Lam_new[:n3, :n3])
        Lam_new = keep
        mu_new = mu_new.at[n3:].set(est.mu_p[n3:])
    return Lam_new, mu_new


def push_warmup(est: Est, d: SlotData, idx):
    """Insert slot at position idx (1..W-1) during warmup; no marginalization."""
    W, N = est.X.shape[:2]
    x_new = jax.vmap(se2.compose)(est.X[idx - 1], d.u)
    return est._replace(X=est.X.at[idx].set(x_new),
                        win=_insert(est.win, d, idx),
                        slot_mask=est.slot_mask.at[idx].set(True))


def push_slide(est: Est, d: SlotData, c: _Consts, use_channel, estimate_field,
               use_msgs=False):
    """Marginalize slot 0, roll the window, insert the new slot at W-1."""
    W, N = est.X.shape[:2]
    Lam_new, mu_new = marginalize(est, c, use_channel, estimate_field, use_msgs)
    X = jnp.roll(est.X, -1, axis=0)
    x_new = jax.vmap(se2.compose)(X[W - 2], d.u)
    return est._replace(X=X.at[W - 1].set(x_new),
                        Lam_p=Lam_new, mu_p=mu_new,
                        win=_roll(est.win, d))


class InnerField(NamedTuple):
    """A-HY inner state: accumulated channel information on the field."""
    Lam_F: jax.Array     # (G2,G2)
    F: jax.Array         # (G,G)


def init_inner(F0):
    return InnerField(Lam_F=jnp.zeros((F0.size, F0.size)), F=F0)


def _slot_channel_blocks(X_slot, F, d: SlotData, c: _Consts):
    """Channel-factor normal-equation blocks for ONE slot at poses X_slot (N,3)."""
    N = X_slot.shape[0]
    n3, G2 = 3 * N, F.size
    pr, Jpi, Jpj, JF = factors.emission_factors(X_slot[None], F, c.extent, c.p0,
                                                c.gamma)
    pad = lambda J2: jnp.concatenate([J2, jnp.zeros((J2.shape[0], 1))], -1)
    Ji3, Jj3 = pad(Jpi), pad(Jpj)
    pr_lin = pr + _dp_links(d.dp) + d.hbar.reshape(-1) + c.fade_mean
    shift, jfac = factors.truncation_correction(pr_lin, c.theta, c.sigma_f)
    e_rs = pr_lin + shift - d.rssi.reshape(-1)
    w_rs = jnp.where(d.rssi_mask.reshape(-1),
                     1.0 / (c.rssi_var0 + d.hvar.reshape(-1)), 0.0)
    e_ob, w_ob = factors.probit_pseudo(pr_lin, d.succ.reshape(-1), c.theta, c.sigma_f)
    w_ob = jnp.where(d.att.reshape(-1), w_ob, 0.0)
    ones = jnp.ones_like(jfac)

    ii = jnp.arange(N)[:, None].repeat(N, 1).reshape(-1)
    jj = jnp.arange(N)[None, :].repeat(N, 0).reshape(-1)
    Axx = jnp.zeros((N, N, 3, 3))
    bx = jnp.zeros((N, 3))
    Axf = jnp.zeros((N, 3, G2))
    Aff = jnp.zeros((G2, G2))
    bf = jnp.zeros(G2)
    for e_c, w_c, f_c in ((e_rs, w_rs, jfac), (e_ob, w_ob, ones)):
        Ji3c, Jj3c = Ji3 * f_c[:, None], Jj3 * f_c[:, None]
        JFc = JF * f_c[:, None]
        Axx = Axx.at[ii, ii].add(jnp.einsum("m,mi,mj->mij", w_c, Ji3c, Ji3c))
        Axx = Axx.at[jj, jj].add(jnp.einsum("m,mi,mj->mij", w_c, Jj3c, Jj3c))
        Hij = jnp.einsum("m,mi,mj->mij", w_c, Ji3c, Jj3c)
        Axx = Axx.at[ii, jj].add(Hij)
        Axx = Axx.at[jj, ii].add(Hij.transpose(0, 2, 1))
        bx = bx.at[ii].add(-(w_c * e_c)[:, None] * Ji3c)
        bx = bx.at[jj].add(-(w_c * e_c)[:, None] * Jj3c)
        Axf = Axf.at[ii].add(jnp.einsum("m,mi,mk->mik", w_c, Ji3c, JFc))
        Axf = Axf.at[jj].add(jnp.einsum("m,mi,mk->mik", w_c, Jj3c, JFc))
        Aff = Aff + (JFc * w_c[:, None]).T @ JFc
        bf = bf - (w_c * e_c) @ JFc
    return (Axx.transpose(0, 2, 1, 3).reshape(n3, n3), Axf.reshape(n3, G2),
            Aff, bx.reshape(-1), bf)


def hybrid_inner_step(inner: InnerField, X_slot, d: SlotData, c: _Consts):
    """A-HY inner: field-only update + the Schur pose-information message (P-HY).
    Returns (inner', msg_lam (3N,3N), msg_eta (3N,))."""
    G = inner.F.shape[0]
    G2 = G * G
    Axx, Axf, Aff_s, bx, bf = _slot_channel_blocks(X_slot, inner.F, d, c)
    Ff = inner.F.reshape(-1)
    bf_p = -Ff / c.f_prior_var
    Aff_p = jnp.eye(G2) / c.f_prior_var
    k1, k2 = _tv_indices(G)
    e_tv = Ff[k2] - Ff[k1]
    w_tv = c.tv_lambda * huber_weights(e_tv, c.tv_delta)
    Aff_p = Aff_p.at[k1, k1].add(w_tv).at[k2, k2].add(w_tv)
    Aff_p = Aff_p.at[k1, k2].add(-w_tv).at[k2, k1].add(-w_tv)
    bf_p = bf_p.at[k1].add(w_tv * e_tv).at[k2].add(-w_tv * e_tv)

    Aff_tot = Aff_s + Aff_p + inner.Lam_F + 1e-6 * jnp.eye(G2)
    cf = jax.scipy.linalg.cho_factor(Aff_tot)
    Y = jax.scipy.linalg.cho_solve(cf, Axf.T)              # (G2, 3N)
    msg_lam = Axx - Axf @ Y
    msg_lam = 0.5 * (msg_lam + msg_lam.T)
    msg_eta = bx - Axf @ jax.scipy.linalg.cho_solve(cf, bf)
    dF = jax.scipy.linalg.cho_solve(cf, bf + bf_p)
    F_new = jnp.clip(inner.F + dF.reshape(G, G), 0.0, 30.0)
    return InnerField(Lam_F=inner.Lam_F + Aff_s, F=F_new), msg_lam, msg_eta


def hybrid_message_known(F, X_slot, d: SlotData, c: _Consts):
    """A-HY message when the field is KNOWN: the channel factors' pose information at
    the fixed field, with NO field marginalization (the Prop-2 confirmation regime).
    Marginalizing a known field, as the Schur form does with finite field info, would
    wrongly subtract real pose information."""
    Axx, _, _, bx, _ = _slot_channel_blocks(X_slot, F, d, c)
    return Axx, bx


def gauge_project(msg_lam, msg_eta, X_slot):
    """A-HYG interface: the channel level ships only the gauge block of the message,
    Lam_g = Gs^T Lam Gs (6 numbers) and eta_g = Gs^T eta (3), and the task level
    rebuilds Lam_proj = Gs (Gs^T Gs)^-1 Lam_g (Gs^T Gs)^-1 Gs^T (and eta likewise).
    That is the exact restriction of the message to range(Gs) -- symmetric, PSD when
    Lam is, and lossless if Lam already lives on the gauge subspace (Prop 2 says at
    most these three directions are ever recoverable). Columns of Gs are normalized
    because the rotation generator scales with the arena; the normalization cancels
    (Gs -> Gs S leaves Gs (Gs^T Gs)^-1 Gs^T unchanged), and with the rotation taken
    about the centroid the normalized columns are orthonormal, so Gs^T Gs = I."""
    Gs = factors.gauge_generators(X_slot[None], jnp.ones(1, bool)).T     # (3N,3)
    Gs = Gs / jnp.linalg.norm(Gs, axis=0, keepdims=True)
    Gp = jnp.linalg.solve(Gs.T @ Gs, Gs.T)                               # (3,3N)
    lam_g = Gs.T @ msg_lam @ Gs
    lam_g = 0.5 * (lam_g + lam_g.T)
    return Gp.T @ lam_g @ Gp, Gp.T @ (Gs.T @ msg_eta)


def rand_project(msg_lam, msg_eta, X_slot, key):
    """A-HYR: the rate-matched control for A-HYG.

    Ships the same nine numbers, a symmetric 3x3 block plus a 3-vector, but on a
    RANDOM three-dimensional subspace of the pose space instead of the gauge
    subspace. Identical bandwidth, identical algebra, identical reconstruction;
    the only difference is which subspace the theory says to keep.

    Without this arm, "nine numbers recover most of the effect" is consistent
    with nine numbers of anything being enough. With it, the comparison isolates
    Proposition 2's content: at most three directions are ever recoverable from
    the channel, and they are the gauge directions, so a rate-constrained
    interface should spend its budget there and nowhere else.

    The subspace is drawn once per slot from a fixed key stream, orthonormalized
    so the projector is exactly comparable to the gauge one.
    """
    n3 = msg_lam.shape[0]
    A = jax.random.normal(key, (n3, 3))
    Qm, _ = jnp.linalg.qr(A)                      # (3N,3) orthonormal columns
    lam_r = Qm.T @ msg_lam @ Qm
    lam_r = 0.5 * (lam_r + lam_r.T)
    return Qm @ lam_r @ Qm.T, Qm @ (Qm.T @ msg_eta)


def hybrid_message(inner: InnerField, X_slot, d: SlotData, c: _Consts):
    """Schur pose-information message at X_slot using the current inner field, with
    NO field update. For relinearizing A-HY's message at the outer estimator's
    improved pose (the frozen-prediction linearization is what makes A-HY lag A-J)."""
    G2 = inner.F.size
    Axx, Axf, Aff_s, bx, bf = _slot_channel_blocks(X_slot, inner.F, d, c)
    Aff_tot = Aff_s + jnp.eye(G2) / c.f_prior_var + inner.Lam_F + 1e-6 * jnp.eye(G2)
    cf = jax.scipy.linalg.cho_factor(Aff_tot)
    msg_lam = Axx - Axf @ jax.scipy.linalg.cho_solve(cf, Axf.T)
    msg_lam = 0.5 * (msg_lam + msg_lam.T)
    msg_eta = bx - Axf @ jax.scipy.linalg.cho_solve(cf, bf)
    return msg_lam, msg_eta


class InnerOnce(NamedTuple):
    """A-HYM's accumulated current-pose surrogate, with 3N pose coordinates.

    These arrays identify current-pose coordinates across slots; they do not
    retain a trajectory or its pose-field prior correlations. Before PSD clipping
    and with a fixed quadratic, Lam_F/C/K and Bx/Bf/u telescope this surrogate's
    precision and information vector. They are not an exact moving-window joint
    Gaussian marginal, including in the translation directions.
    """
    Lam_F: jax.Array     # (G2,G2)  sum_{s<=t} Aff_s
    F: jax.Array         # (G,G)
    C: jax.Array         # (3N,G2)  sum_{s<=t} Axf_s
    K: jax.Array         # (3N,3N)  C_t (Lam_F,t + prior)^-1 C_t^T
    Bx: jax.Array        # (3N,)    sum_{s<=t} bx_s
    Bf: jax.Array        # (G2,)    sum_{s<=t} bf_s
    u: jax.Array         # (3N,)    Bx_t - C_t (Lam_F,t + prior)^-1 Bf_t


def init_inner_once(F0, n):
    return InnerOnce(Lam_F=jnp.zeros((F0.size, F0.size)), F=F0,
                     C=jnp.zeros((3 * n, F0.size)), K=jnp.zeros((3 * n, 3 * n)),
                     Bx=jnp.zeros(3 * n), Bf=jnp.zeros(F0.size),
                     u=jnp.zeros(3 * n))


def _psd_part(A):
    w, V = jnp.linalg.eigh(0.5 * (A + A.T))
    return (V * jnp.maximum(w, 0.0)) @ V.T


def _pinv_apply(A, v):
    """A^+ v with a relative eigenvalue cut; A is gauge-rank-deficient by Prop. 2."""
    w, V = jnp.linalg.eigh(0.5 * (A + A.T))
    keep = w > 1e-8 * jnp.maximum(w.max(), 1e-30)
    return V @ (jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0) * (V.T @ v))


def hybrid_once(inner: InnerOnce, X_slot, d: SlotData, c: _Consts):
    """A-HYM: difference accumulated field elimination on a 3N-pose surrogate.

    The surrogate is T_t = sum Axx_s - C_t (Lam_F,t + prior)^-1 C_t^T,
    C_t = sum Axf_s. Before clipping, the precision increments telescope to this
    expression. The information-vector increments telescope analogously. Frozen
    replay can compare its restriction to a repeated translation perturbation;
    that is not equality of trajectory posterior means or covariances. Poses at
    distinct times remain distinct random variables even when their translation
    generators agree.

    The implemented field update changes linearizations and TV prior weights, and
    PSD clipping can change the telescoped precision. Exact trajectory or sliding
    window messages instead require every retained pose coordinate, the correlated
    retained prior, and replacement of affected matrix/vector contributions. A
    replacement correction need not be PSD. See diag_paper_gaussian for bounded
    checks of that specified Gaussian construction and failure contrasts.
    """
    G = inner.F.shape[0]
    G2 = G * G
    Axx, Axf, Aff_s, bx, bf = _slot_channel_blocks(X_slot, inner.F, d, c)
    Ff = inner.F.reshape(-1)
    bf_p = -Ff / c.f_prior_var
    Aff_p = jnp.eye(G2) / c.f_prior_var
    k1, k2 = _tv_indices(G)
    e_tv = Ff[k2] - Ff[k1]
    w_tv = c.tv_lambda * huber_weights(e_tv, c.tv_delta)
    Aff_p = Aff_p.at[k1, k1].add(w_tv).at[k2, k2].add(w_tv)
    Aff_p = Aff_p.at[k1, k2].add(-w_tv).at[k2, k1].add(-w_tv)
    bf_p = bf_p.at[k1].add(w_tv * e_tv).at[k2].add(-w_tv * e_tv)

    Lam_F = inner.Lam_F + Aff_s
    Aff_tot = Lam_F + Aff_p + 1e-6 * jnp.eye(G2)
    cf = jax.scipy.linalg.cho_factor(Aff_tot)
    C = inner.C + Axf
    K = C @ jax.scipy.linalg.cho_solve(cf, C.T)
    K = 0.5 * (K + K.T)
    msg_lam = _psd_part(Axx - (K - inner.K))
    # eta telescopes exactly as lam does: accumulate the raw pose and field
    # information vectors, form the joint marginal vector u_t, ship its increment.
    Bx = inner.Bx + bx
    Bf = inner.Bf + bf
    u = Bx - C @ jax.scipy.linalg.cho_solve(cf, Bf)
    msg_eta = u - inner.u
    dF = jax.scipy.linalg.cho_solve(cf, bf + bf_p)
    F_new = jnp.clip(inner.F + dF.reshape(G, G), 0.0, 30.0)
    return (InnerOnce(Lam_F=Lam_F, F=F_new, C=C, K=K, Bx=Bx, Bf=Bf, u=u),
            msg_lam, msg_eta)


def hybrid_once_gauge(inner: InnerOnce, X_slot, d: SlotData, c: _Consts):
    """A-HYK: apply the marginalize-once correction ONLY on the gauge subspace.

    hybrid_once accumulates sum_s Axx_s across slots, which identifies each slot's
    pose block with the next. That identification is exact only for directions whose
    generator does not depend on pose, i.e. the gauge directions; robot i's heading at
    slot 5 and at slot 500 are different variables. It is also exactly the gauge block
    where the per-slot pipeline double counts (Fig. 2a). So keep A-HY's per-slot
    message everywhere and substitute the telescoped increment on range(G) alone.
    """
    G = inner.F.shape[0]
    G2 = G * G
    Axx, Axf, Aff_s, bx, bf = _slot_channel_blocks(X_slot, inner.F, d, c)
    Ff = inner.F.reshape(-1)
    bf_p = -Ff / c.f_prior_var
    Aff_p = jnp.eye(G2) / c.f_prior_var
    k1, k2 = _tv_indices(G)
    e_tv = Ff[k2] - Ff[k1]
    w_tv = c.tv_lambda * huber_weights(e_tv, c.tv_delta)
    Aff_p = Aff_p.at[k1, k1].add(w_tv).at[k2, k2].add(w_tv)
    Aff_p = Aff_p.at[k1, k2].add(-w_tv).at[k2, k1].add(-w_tv)
    bf_p = bf_p.at[k1].add(w_tv * e_tv).at[k2].add(-w_tv * e_tv)

    Lam_F = inner.Lam_F + Aff_s
    Aff_tot = Lam_F + Aff_p + 1e-6 * jnp.eye(G2)
    cf = jax.scipy.linalg.cho_factor(Aff_tot)

    lam_s = Axx - Axf @ jax.scipy.linalg.cho_solve(cf, Axf.T)   # A-HY's per-slot msg
    eta_s = bx - Axf @ jax.scipy.linalg.cho_solve(cf, bf)

    C = inner.C + Axf
    K = 0.5 * (lambda M: M + M.T)(C @ jax.scipy.linalg.cho_solve(cf, C.T))
    Bx, Bf = inner.Bx + bx, inner.Bf + bf
    u = Bx - C @ jax.scipy.linalg.cho_solve(cf, Bf)
    lam_inc = Axx - (K - inner.K)                                # telescoped increment
    eta_inc = u - inner.u

    Gs = factors.gauge_generators(X_slot[None], jnp.ones(1, bool)).T     # (3N,3)
    Gs = Gs / jnp.linalg.norm(Gs, axis=0, keepdims=True)                 # orthonormal
    P = Gs @ Gs.T
    Q = jnp.eye(P.shape[0]) - P
    # Block-diagonal in the gauge/complement split. Substituting the corrected
    # (smaller) gauge block into lam_s while leaving the cross blocks P lam_s Q
    # intact violates the Schur condition, so the matrix goes indefinite and the
    # PSD projection puts the correction straight back: measured, that reverted
    # 82 percent of the removal on t_x and 96 percent on t_y, with a worst
    # eigenvalue of -0.92. Compressing onto the two subspaces instead is PSD by
    # construction, needs no projection, and gives exactly lam_inc on range(G)
    # and lam_s on its complement. The price is the gauge-to-pose cross
    # information, which this form drops.
    msg_lam = P @ lam_inc @ P + Q @ lam_s @ Q
    msg_lam = 0.5 * (msg_lam + msg_lam.T)
    msg_eta = Q @ eta_s + P @ eta_inc

    dF = jax.scipy.linalg.cho_solve(cf, bf + bf_p)
    F_new = jnp.clip(inner.F + dF.reshape(G, G), 0.0, 30.0)
    return (InnerOnce(Lam_F=Lam_F, F=F_new, C=C, K=K, Bx=Bx, Bf=Bf, u=u),
            msg_lam, msg_eta)
