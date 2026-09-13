"""Level-2 channel-model ladder rungs for the A_BL scalar interface (tab:l2).

Both inners emit per-link expected-delivery scalars exp_del (N,N) consumed
identically to the Gilbert-Elliott bank: rel_trust = 0.5 + 0.5*exp_del and the
delivery scheduler weight. Independent Bernoulli is the floor (no structure);
Gudmundson is spatially correlated shadowing without an explicit field
(kriging-lite over link-midpoint residuals). Numpy, per slot, outside jit.
"""
import numpy as np
from scipy.special import ndtr


class BernoulliInner:
    """Per-link EWMA of observed delivery on attempted links."""

    def __init__(self, n, alpha=0.05, init=0.5):
        self.alpha = alpha
        self.p = np.full((n, n), init)

    def update(self, att, succ):
        self.p = np.where(att, (1.0 - self.alpha) * self.p
                          + self.alpha * succ.astype(np.float64), self.p)

    def exp_del(self):
        return self.p.copy()


def _pathloss(pos, cfg_ch):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) + 1e-6
    return cfg_ch.p0_dbm - 10.0 * cfg_ch.gamma * np.log10(d)


class GudmundsonInner:
    """Spatially correlated shadowing: buffered (midpoint, rssi residual) pairs
    from successful attempts, predicted per link by distance- and age-weighted
    mean (weights exp(-d/d_c) * exp(-age/tau))."""

    def __init__(self, buf_len=300, d_c=5.0, tau=200.0):
        self.buf_len, self.d_c, self.tau = buf_len, d_c, tau
        self.mids = np.zeros((0, 2))
        self.res = np.zeros(0)
        self.t_obs = np.zeros(0)
        self.t = 0

    def update(self, att, succ, rssi, pos_est, cfg_ch, fade_mean, sigma_eff):
        self.t += 1
        i, j = np.nonzero(att & succ)
        if i.size == 0:
            return
        pl = _pathloss(pos_est, cfg_ch)[i, j]
        r = rssi[i, j] - pl - fade_mean
        m = 0.5 * (pos_est[i] + pos_est[j])
        self.mids = np.concatenate([self.mids, m])[-self.buf_len:]
        self.res = np.concatenate([self.res, r])[-self.buf_len:]
        self.t_obs = np.concatenate(
            [self.t_obs, np.full(i.size, float(self.t))])[-self.buf_len:]

    def exp_del(self, pos_est, cfg_ch, fade_mean, sigma_eff):
        pr = _pathloss(pos_est, cfg_ch) + fade_mean
        if self.res.size:
            mids = 0.5 * (pos_est[:, None, :] + pos_est[None, :, :])
            dd = np.linalg.norm(mids[:, :, None, :] - self.mids[None, None],
                                axis=-1)
            w = np.exp(-dd / self.d_c) \
                * np.exp(-(self.t - self.t_obs) / self.tau)
            pr = pr + (w * self.res).sum(-1) / np.maximum(w.sum(-1), 1e-9)
        return ndtr((pr - cfg_ch.decode_dbm) / sigma_eff)
