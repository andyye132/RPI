"""Transmit power as an information actuator.

The decode bit carries 0.89 of the total gauge information and its Bernoulli
Fisher term

    w_bit = [phi(z)/sigma_f]^2 / (p (1-p)),      z = (pr - decode_dbm) / sigma_f

is maximized at p = 1/2, i.e. on links at the decode threshold. Transmit power is
otherwise a fixed scalar (cfg.channel.p0_dbm), so each link's p is whatever
geometry hands it; this module places it deliberately. The setpoint is derived,
not tuned: for a target decode probability p_target,

    pr_target = decode_dbm + sigma_f * Phi^-1(p_target)

and the power putting link (i,j) there is pr_target minus its predicted gain.
p_target = 0.5 is information-optimal; p_target -> 1 recovers the fixed-power
radio. Sweeping it traces the frontier against the relative-measurement graph,
which wants reliable links instead.

Power may only be LOWERED -- the cap is the existing p0_dbm -- so no comparison
is granted extra energy. Computed from the ESTIMATED field and poses only.
"""
import jax
import jax.numpy as jnp


def setpoint(gain_hat, chosen, decode_dbm, sigma_f, fade_mean, p_target,
             p_cap, p_floor):
    """Per-transmitter power (N,) dBm placing each scheduled link at p_target.

    gain_hat (N,N) predicted link gain in dB EXCLUDING transmit power, i.e.
    channel.pr_mean evaluated at p0 = 0 on the estimated poses and field.
    chosen (N,N) bool, this slot's schedule.

    `scheduler.conflict_adj` blocks two links that share a node in any role, so
    a transmitter has at most one chosen link and the row sum below selects that
    link's requirement rather than mixing several. Nodes that do not transmit are
    left at the cap; channel.step masks them out of the interference sum anyway.
    """
    z = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p_target - 1.0)
    pr_target = decode_dbm + sigma_f * z
    p_req = pr_target - fade_mean - gain_hat
    sel = chosen.astype(p_req.dtype)
    p_row = (sel * p_req).sum(axis=1)
    return jnp.where(chosen.any(axis=1),
                     jnp.clip(p_row, p_floor, p_cap), p_cap)


def energy_matched(p_dbm, chosen, p_cap):
    """Control arm: the same radiated energy, spread uniformly.

    Replaces each transmitting node's power by the LINEAR mean over transmitting
    nodes, so total energy per slot is preserved while the per-link placement is
    destroyed. Without this arm an info-arm win is explained just as well by the
    lower power LEVEL as by putting each link at its Fisher-optimal point, and
    the campaign's other controllers all failed for want of exactly this kind of
    matched control.
    """
    tx_any = chosen.any(axis=1)
    lin = 10.0 ** (p_dbm / 10.0)
    k = jnp.maximum(tx_any.sum(), 1)
    mean_lin = jnp.where(tx_any, lin, 0.0).sum() / k
    out = 10.0 * jnp.log10(jnp.maximum(mean_lin, 1e-30))
    return jnp.where(tx_any, jnp.minimum(out, p_cap), p_cap)
