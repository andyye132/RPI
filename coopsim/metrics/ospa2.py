"""OSPA(2) metric over tracks (Beard 2020): OSPA on track sets, with a base
distance that is a windowed time average of the capped per-frame distance."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _est_tracks(est, est_labels, est_mask):
    """Track view of labeled estimates: positions (T,L,2) and existence (T,L)
    per unique label. Duplicate labels in a frame resolve to the first slot."""
    labels = np.unique(est_labels[est_mask]) if est_mask.any() else np.empty(0, int)
    T = est.shape[0]
    pos = np.zeros((T, len(labels), 2))
    exists = np.zeros((T, len(labels)), bool)
    for k, lab in enumerate(labels):
        sel = est_mask & (est_labels == lab)
        has = sel.any(axis=1)
        exists[:, k] = has
        idx = sel.argmax(axis=1)
        pos[has, k] = est[np.nonzero(has)[0], idx[has], :2]
    return pos, exists


def ospa2_series(tau, alive, est, est_labels, est_mask, window=100, c=5.0, p=2):
    """OSPA(2) at each frame over the trailing window [t-window+1, t].
    tau (T,Mt,4), alive (T,Mt) bool, est (T,ML,>=2), est_labels (T,ML) int,
    est_mask (T,ML) bool. GT track = column of tau where alive; est track =
    unique label. Base distance between a GT and an est track = mean over the
    window frames where either exists of min(d, c), with c charged where
    exactly one exists. Returns (T,) array."""
    T = tau.shape[0]
    gpos, gex = tau[:, :, :2].astype(np.float64), np.asarray(alive, bool)
    epos, eex = _est_tracks(est, est_labels, est_mask)
    out = np.zeros(T)
    for t in range(T):
        w0 = max(0, t - window + 1)
        gi = np.nonzero(gex[w0:t + 1].any(axis=0))[0]
        ei = np.nonzero(eex[w0:t + 1].any(axis=0))[0]
        ng, ne = len(gi), len(ei)
        if ng == 0 and ne == 0:
            continue
        if ng == 0 or ne == 0:
            out[t] = c
            continue
        Pg, Eg = gpos[w0:t + 1][:, gi], gex[w0:t + 1][:, gi]
        Pe, Ee = epos[w0:t + 1][:, ei], eex[w0:t + 1][:, ei]
        d = np.minimum(np.linalg.norm(Pg[:, :, None] - Pe[:, None], axis=3), c)
        both = Eg[:, :, None] & Ee[:, None]
        either = Eg[:, :, None] | Ee[:, None]
        base = np.where(both, d, np.where(either, c, 0.0)).sum(0) / either.sum(0)
        row, col = linear_sum_assignment(base ** p)
        n_max = max(ng, ne)
        total = (base[row, col] ** p).sum() + c ** p * (n_max - min(ng, ne))
        out[t] = (total / n_max) ** (1.0 / p)
    return out
