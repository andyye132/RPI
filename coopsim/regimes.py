"""Estimator-side per-link regime bank (jit, vmapped over the (N,N) link grid).

Two states per link: 0 = clear, 1 = blocked. Transitions are geometry-conditioned on
the line-of-sight probability under the CURRENT field posterior (an observation model
conditioned on the state estimate, not a forecaster of an exogenous process). Soft
marginalization hands the factors a mean regime offset h_bar and its variance h_var.
"""
import jax
import jax.numpy as jnp

from coopsim.channel import p_los


def init_belief(n):
    return jnp.tile(jnp.array([0.9, 0.1]), (n, n, 1))


def build_transition(plos_hat, cfg_ch):
    """(N,N,2,2) row-stochastic transition, conditioned on estimated p_LoS."""
    p_enter = cfg_ch.ge_p_bad_los + (cfg_ch.ge_p_bad_nlos - cfg_ch.ge_p_bad_los) * (1.0 - plos_hat)
    persist = jnp.full_like(plos_hat, cfg_ch.ge_persist)
    row_good = jnp.stack([1.0 - p_enter, p_enter], -1)
    row_bad = jnp.stack([1.0 - persist, persist], -1)
    return jnp.stack([row_good, row_bad], -2)


def forward_step(belief, trans):
    return jnp.einsum("ijk,ijkl->ijl", belief, trans)


def forecast_ahead(belief, trans, k):
    return jax.lax.fori_loop(0, k, lambda _, b: forward_step(b, trans), belief)


def obs_likelihood(obs, pr_hat, cfg_ch, rssi_var, sigma_eff=None):
    """(N,N,2) likelihood of this slot's channel observables per regime.
    pr_hat must already include the fading dB mean; sigma_eff the fading-inflated
    outcome scale (falls back to the shadowing sigma)."""
    sig = cfg_ch.sigma_f_db if sigma_eff is None else sigma_eff
    pr_s = pr_hat[..., None] - jnp.array([0.0, cfg_ch.block_extra_db])   # (N,N,2)
    p_succ = jax.scipy.stats.norm.cdf((pr_s - cfg_ch.decode_dbm) / sig)
    lik = jnp.where(obs.success[..., None], p_succ, 1.0 - p_succ)
    rssi_lik = jnp.exp(-0.5 * (obs.rssi_q[..., None] - pr_s) ** 2 / rssi_var)
    lik = lik * jnp.where(obs.success[..., None], rssi_lik + 1e-9, 1.0)
    return jnp.where(obs.attempted[..., None], lik, 1.0)


def update(belief, obs, pr_hat, plos_hat, cfg_ch, rssi_var, sigma_eff=None):
    b = forward_step(belief, build_transition(plos_hat, cfg_ch))
    b = b * obs_likelihood(obs, pr_hat, cfg_ch, rssi_var, sigma_eff)
    return b / jnp.maximum(b.sum(-1, keepdims=True), 1e-12)


def h_moments(belief, cfg_ch):
    """Soft-marginalized regime offset: mean h_bar and variance h_var, (N,N) each."""
    p_bad = belief[..., 1]
    h_bar = -cfg_ch.block_extra_db * p_bad
    h_var = cfg_ch.block_extra_db ** 2 * p_bad * (1.0 - p_bad)
    return h_bar, h_var
