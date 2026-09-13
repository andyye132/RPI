"""Loader for the Utah SPAN RTI datasets (npatwari/rti, GPL, Wilson-Patwari).

Two real 2.4 GHz TelosB experiments with surveyed node positions, used to validate
AETHER's emission model and field-estimation machinery on real radio without hardware:
  - ten-node square (empty area): 10 nodes, 4 channels, for the path-loss / emission fit
  - basement: 10 nodes, 8 channels, a person walking a known 21-pivot path, for
    real-data tomography (the E4 protocol analog)

Row format: whitespace ints, one RSS per (tx, rx, channel) directed link, trailing
timestamp (ms). Link column = ch*P + tx_enum*(n-1) + rx_enum, minus 1 if rx>tx, with
P = n*(n-1) (rss.py::linkNumForTxRxChLists). RSS is dBm; any value > -10 means no
packet (missing).
"""
from pathlib import Path

import numpy as np

from coopsim.paths import RTI as ROOT


def _pair_index(a, b, n):
    idx = a * (n - 1) + b
    return idx - 1 if b > a else idx


def load_rss(data_file, coord_file, n=10):
    """Returns coords (n,2), and channel-averaged undirected RSS series
    rss (T, n, n) in dBm with NaN for missing, plus timestamps (T,)."""
    coords = np.loadtxt(coord_file)
    n = coords.shape[0]
    P = n * (n - 1)
    rows = []
    ts = []
    for line in Path(data_file).read_text().splitlines():
        v = line.split()
        if not v:
            continue
        vals = np.array([int(x) for x in v[:-1]])
        ts.append(float(v[-1]))
        channels = vals.size // P
        vals = vals[:channels * P].reshape(channels, P).astype(np.float64)
        vals[vals > -10] = np.nan                       # missing
        # directed link RSS averaged over channels
        d = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                col = _pair_index(a, b, n)
                d[a, b] = np.nanmean(vals[:, col])
        rows.append(d)
    rss = np.array(rows)
    # symmetrize to undirected (average the two directions where both present)
    und = np.nanmean(np.stack([rss, rss.transpose(0, 2, 1)]), axis=0)
    return coords, und, np.array(ts)


def load_path(path_file, pivot_file):
    """Ground-truth walked polyline: pivot coords in visit order."""
    pivots = np.loadtxt(pivot_file)
    order = np.loadtxt(path_file).astype(int)
    return pivots[order]


def point_to_polyline(p, poly):
    """Min distance from point p to the polyline through poly (K,2)."""
    best = np.inf
    for k in range(len(poly) - 1):
        a, b = poly[k], poly[k + 1]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best
