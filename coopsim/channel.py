"""Physical channel for AETHER v3 (jit, (N,N) link matrices).

Received power = P0 - 10 gamma log10(d) - line_integral(F_true) + fading + blockage
+ shadowing noise. Nakagami-m small-scale fading with Doppler-coherent temporal
correlation: a latent per-link AR(1) standard normal (rho = J0(2 pi f_D dt)) is mapped
through the exact Nakagami dB inverse CDF (precomputed lookup table), giving the exact
marginal with the right coherence. Blockage is a 2-state Gilbert-Elliott chain whose
entry probability is conditioned on line-of-sight through F_true. Success requires the
decode threshold and SINR against concurrent transmitters. Noise floor is set so that
an interference-free link at the decode threshold sits exactly at the SINR threshold.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from coopsim.field import line_integral_links

C_LIGHT = 3e8


class ChannelState(NamedTuple):
    u_fade: jax.Array      # (N,N) latent AR(1) standard normal
    ge_bad: jax.Array      # (N,N) bool, blockage chain state


class ChannelObs(NamedTuple):
    attempted: jax.Array   # (N,N) bool
    success: jax.Array     # (N,N) bool
    rssi_q: jax.Array      # (N,N) quantized dBm, valid where success


def nakagami_lut(m, n=1024):
    """dB fading values on a uniform probability grid (numpy, once per episode)."""
    from scipy.stats import gamma as gamma_dist
    p = (np.arange(n) + 0.5) / n
    return jnp.asarray(10.0 * np.log10(gamma_dist.ppf(p, a=m, scale=1.0 / m)))


def fade_stats(m):
    """Mean and variance of the Nakagami-m dB fading offset."""
    from scipy.special import digamma, polygamma
    s = 10.0 / np.log(10.0)
    return float(s * (digamma(m) - np.log(m))), float(s ** 2 * polygamma(1, m))


def bessel_j0(x):
    """Abramowitz-Stegun 9.4.1 / 9.4.3 polynomial approximation, |err| < 1e-7."""
    ax = jnp.abs(x)
    y = (x / 3.0) ** 2
    small = (1.0 - 2.2499997 * y + 1.2656208 * y ** 2 - 0.3163866 * y ** 3
             + 0.0444479 * y ** 4 - 0.0039444 * y ** 5 + 0.0002100 * y ** 6)
    z = 3.0 / jnp.maximum(ax, 3.0)
    f0 = (0.79788456 - 0.00000077 * z - 0.00552740 * z ** 2 - 0.00009512 * z ** 3
          + 0.00137237 * z ** 4 - 0.00072805 * z ** 5 + 0.00014476 * z ** 6)
    th0 = (jnp.maximum(ax, 3.0) - 0.78539816 - 0.04166397 * z - 0.00003954 * z ** 2
           + 0.00262573 * z ** 3 - 0.00054125 * z ** 4 - 0.00029333 * z ** 5
           + 0.00013558 * z ** 6)
    large = f0 * jnp.cos(th0) / jnp.sqrt(jnp.maximum(ax, 3.0))
    return jnp.where(ax <= 3.0, small, large)


def init_state(key, n):
    return ChannelState(u_fade=jax.random.normal(key, (n, n)),
                        ge_bad=jnp.zeros((n, n), bool))


def pr_mean(pos, F, extent, p0_dbm, gamma):
    """Mean received power (N,N) dBm from path loss + field shadowing.

    p0_dbm is the 1 m reference power. A scalar is the fixed-power radio; an
    (N,) vector is per-transmitter power control, indexed by the ROW, since
    pr[i,j] is what i's transmission delivers at j. Passing 0.0 returns the link
    GAIN alone, which is what the power controller needs.
    """
    d = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) + 1e-6
    p0 = jnp.asarray(p0_dbm)
    p0 = p0[:, None] if p0.ndim == 1 else p0
    return p0 - 10.0 * gamma * jnp.log10(d) - line_integral_links(pos, F, extent)


def p_los(pos, F, extent, gate_db, scale_db):
    """Soft line-of-sight probability (N,N), shared by radio and sensing."""
    li = line_integral_links(pos, F, extent)
    return jax.nn.sigmoid((gate_db - li) / scale_db)


def step(t, keys, pos, vel, F_true, state, tx, lut, cfg_ch, extent, dt, p_tx=None):
    """One slot. pos (N,2), vel (N,2), tx (N,N) bool. Returns (obs, state').

    p_tx (N,) per-transmitter power in dBm, or None for the fixed-power radio at
    cfg_ch.p0_dbm. The SINR sum below already reads each transmitter's power out
    of pr, so interference stays correct under power control with no change.
    """
    k_fade = jax.random.fold_in(keys.fading, t)
    k_shadow = jax.random.fold_in(keys.fading, t + 1_000_000)
    k_block = jax.random.fold_in(keys.blockage, t)
    n = pos.shape[0]

    v_rel = jnp.linalg.norm(vel[:, None, :] - vel[None, :, :], axis=-1)
    f_d = v_rel * cfg_ch.f_c_hz / C_LIGHT
    rho = jnp.clip(bessel_j0(2.0 * jnp.pi * f_d * dt), -0.999, 0.999)
    u = rho * state.u_fade + jnp.sqrt(1.0 - rho ** 2) * jax.random.normal(k_fade, (n, n))
    k_burst = int(getattr(cfg_ch, "burst_k", 1))
    if k_burst > 1:
        # frequency-hopped burst of k repeats: mean of k independent fades (CLT),
        # temporal correlation kept through the shared latent
        fade_db = cfg_ch._fade_mu + jnp.sqrt(cfg_ch._fade_var / k_burst) * u
    else:
        p = jax.scipy.stats.norm.cdf(u)
        fade_db = jnp.interp(p * lut.shape[0] - 0.5, jnp.arange(lut.shape[0]), lut)

    plos = p_los(pos, F_true, extent, 6.0, 1.5)
    p_enter = cfg_ch.ge_p_bad_los + (cfg_ch.ge_p_bad_nlos - cfg_ch.ge_p_bad_los) * (1.0 - plos)
    p_bad = jnp.where(state.ge_bad, cfg_ch.ge_persist, p_enter)
    ge_bad = jax.random.uniform(k_block, (n, n)) < p_bad

    pr = (pr_mean(pos, F_true, extent,
                  cfg_ch.p0_dbm if p_tx is None else p_tx, cfg_ch.gamma)
          + fade_db
          - jnp.where(ge_bad, cfg_ch.block_extra_db, 0.0)
          + cfg_ch.sigma_f_db * jax.random.normal(k_shadow, (n, n)))

    # SINR at each receiver against concurrent transmitters (noise floor such that
    # decode threshold == SINR threshold when interference-free)
    p_lin = 10.0 ** (pr / 10.0)
    tx_any = tx.any(axis=1)                                   # (N,) transmitting
    interference = (p_lin * tx_any[:, None]).sum(axis=0)[None, :] - jnp.where(tx, p_lin, 0.0)
    noise_lin = 10.0 ** ((cfg_ch.decode_dbm - cfg_ch.sinr_db) / 10.0)
    sinr_db = 10.0 * jnp.log10(jnp.where(tx, p_lin, 1e-30) / (noise_lin + interference))

    success = tx & (pr > cfg_ch.decode_dbm) & (sinr_db > cfg_ch.sinr_db)
    rssi_q = jnp.round(pr / cfg_ch.rssi_quant_db) * cfg_ch.rssi_quant_db
    return (ChannelObs(attempted=tx, success=success, rssi_q=rssi_q),
            ChannelState(u_fade=u, ge_bad=ge_bad))
