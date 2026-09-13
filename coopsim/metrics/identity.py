"""Identity metrics for labeled tracks: IDF1 (Ristani 2016), ID switches,
fragmentations, cardinality error."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from coopsim.metrics.ospa2 import _est_tracks


def _greedy_match(dist, thresh):
    d = dist.copy()
    pairs = []
    while d.size:
        r, c = np.unravel_index(np.argmin(d), d.shape)
        if d[r, c] > thresh:
            break
        pairs.append((r, c))
        d[r, :] = np.inf
        d[:, c] = np.inf
    return pairs


def identity_metrics(tau, alive, est, est_labels, est_mask, thresh=2.0):
    """IDF1, ID switches, fragmentations, mean cardinality error.

    GT identity = column of tau; predicted identity = est label. IDF1 matches
    identities globally by total frames within thresh; idsw counts per-frame
    greedy-match label changes per GT track against its last matched label;
    frag counts gap starts while a track is alive and was matched at its
    previous alive frame. Returns dict(idf1, idsw, frag, card_err)."""
    tau = np.asarray(tau, np.float64)
    alive = np.asarray(alive, bool)
    T, Mt = alive.shape
    epos, eex = _est_tracks(est, est_labels, est_mask)
    L = eex.shape[1]

    if L:
        d = np.linalg.norm(tau[:, :, None, :2] - epos[:, None], axis=3)
        overlap = (alive[:, :, None] & eex[:, None] & (d <= thresh)).sum(0)
        row, col = linear_sum_assignment(-overlap)
        idtp = int(overlap[row, col].sum())
    else:
        idtp = 0
    idfp = int(eex.sum()) - idtp
    idfn = int(alive.sum()) - idtp
    den = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / den if den else 1.0

    idsw = 0
    frag = 0
    last_lab = np.full(Mt, -1, int)
    prev_matched = np.zeros(Mt, bool)
    for t in range(T):
        gi = np.nonzero(alive[t])[0]
        ej = np.nonzero(eex[t])[0]
        matched_now = np.zeros(Mt, bool)
        if len(gi) and len(ej):
            dt = np.linalg.norm(tau[t, gi][:, None, :2] - epos[t, ej][None], axis=2)
            for a, b in _greedy_match(dt, thresh):
                g, k = gi[a], ej[b]
                matched_now[g] = True
                if last_lab[g] >= 0 and last_lab[g] != k:
                    idsw += 1
                last_lab[g] = k
        frag += int((prev_matched[gi] & ~matched_now[gi]).sum())
        prev_matched[gi] = matched_now[gi]

    card_err = float(np.abs(alive.sum(1) - np.asarray(est_mask, bool).sum(1)).mean())
    return {"idf1": float(idf1), "idsw": idsw, "frag": frag, "card_err": card_err}
