"""Gauge decomposition of multi-robot localization error: the common
(unobservable) translation mode vs relative shape error vs raw ATE."""
import numpy as np


def gauge_decompose(true_xy, est_xy):
    """true_xy, est_xy (T,N,2). Returns dict of (T,) arrays: common = norm of
    the mean error over robots, relative = RMS of the error about that mean,
    ate = RMS of the raw error."""
    e = np.asarray(est_xy, np.float64) - np.asarray(true_xy, np.float64)
    mean_e = e.mean(axis=1, keepdims=True)
    common = np.linalg.norm(mean_e[:, 0], axis=1)
    relative = np.sqrt((np.linalg.norm(e - mean_e, axis=2) ** 2).mean(axis=1))
    ate = np.sqrt((np.linalg.norm(e, axis=2) ** 2).mean(axis=1))
    return {"common": common, "relative": relative, "ate": ate}


def gauge_modes(X_true, X_est):
    """Per-mode gauge error for Proposition 3's degeneracies. X_* (T,N,3) poses.
    Projects the pose error onto the three scene generators at the true poses,
    translation in x and y and rotation about the team centroid, each normalized,
    and returns the coefficient along each. A field whose stabilizer contains a
    mode leaves that coefficient free to drift while the others stay bounded."""
    Xt, Xe = np.asarray(X_true, np.float64), np.asarray(X_est, np.float64)
    T, N = Xt.shape[:2]
    e = np.stack([Xe[..., 0] - Xt[..., 0], Xe[..., 1] - Xt[..., 1],
                  np.arctan2(np.sin(Xe[..., 2] - Xt[..., 2]),
                             np.cos(Xe[..., 2] - Xt[..., 2]))], -1)
    px, py = Xt[..., 0], Xt[..., 1]
    cx, cy = px.mean(1, keepdims=True), py.mean(1, keepdims=True)
    zed, one = np.zeros_like(px), np.ones_like(px)
    gens = [np.stack([one, zed, zed], -1), np.stack([zed, one, zed], -1),
            np.stack([-(py - cy), px - cx, one], -1)]
    out = {}
    for name, g in zip(("tx", "ty", "rot"), gens):
        gn = g / np.linalg.norm(g.reshape(T, -1), axis=1)[:, None, None]
        out[name] = np.abs((e.reshape(T, -1) * gn.reshape(T, -1)).sum(1))
    return out
