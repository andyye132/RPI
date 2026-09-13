"""GOSPA metric (Rahmathullah 2017) for point sets and episode series.

Assignment core lifted from ModTrack's local GOSPA backend: LAP on min(d, c)^p,
then pairs at distance >= c counted as missed + false rather than assigned. The
decomposition follows the alpha = 2 convention, where cutting a pair (cost
2 * c^p / alpha) equals assigning it at the cap (cost c^p), so the LAP stays exact.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def gospa(gt, est, c=5.0, p=2, alpha=2):
    """GOSPA between two point sets. gt (n,>=2), est (m,>=2), positions are the
    first 2 columns. Returns dict(gospa, loc, missed, false, n_assigned); loc,
    missed, false are the powered components (they sum to gospa**p)."""
    gt = np.asarray(gt, np.float64)
    est = np.asarray(est, np.float64)
    n = gt.shape[0] if gt.size else 0
    m = est.shape[0] if est.size else 0
    cutoff = c ** p / alpha
    if n == 0 and m == 0:
        return {"gospa": 0.0, "loc": 0.0, "missed": 0.0, "false": 0.0, "n_assigned": 0}
    if n == 0 or m == 0:
        missed, false = cutoff * n, cutoff * m
        return {"gospa": (missed + false) ** (1.0 / p), "loc": 0.0,
                "missed": missed, "false": false, "n_assigned": 0}
    d = np.linalg.norm(gt[:, None, :2] - est[None, :, :2], axis=2)
    row, col = linear_sum_assignment(np.minimum(d, c) ** p)
    ok = d[row, col] < c
    loc = float((d[row, col][ok] ** p).sum())
    n_assigned = int(ok.sum())
    missed = cutoff * (n - n_assigned)
    false = cutoff * (m - n_assigned)
    return {"gospa": (loc + missed + false) ** (1.0 / p), "loc": loc,
            "missed": missed, "false": false, "n_assigned": n_assigned}


def gospa_series(tau, alive, est, est_mask, c=5.0, p=2, alpha=2):
    """Per-frame GOSPA over an episode. tau (T,Mt,4), alive (T,Mt) bool,
    est (T,ML,>=2), est_mask (T,ML) bool. Returns dict of (T,) arrays
    (gospa, loc, missed, false, card_err); card_err = |n_alive - n_est|."""
    T = tau.shape[0]
    out = {k: np.zeros(T) for k in ("gospa", "loc", "missed", "false", "card_err")}
    finite = np.isfinite(est[..., :2]).all(-1)      # guard: drop numerically bad rows
    est_mask = est_mask & finite
    for t in range(T):
        g = gospa(tau[t, alive[t]], est[t, est_mask[t]], c=c, p=p, alpha=alpha)
        for k in ("gospa", "loc", "missed", "false"):
            out[k][t] = g[k]
        out["card_err"][t] = abs(int(alive[t].sum()) - int(est_mask[t].sum()))
    return out
