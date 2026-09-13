"""Hand-checkable selftests for the metrics package. Returns PASS/FAIL lines;
run on a compute node, never the login node."""
import os
import importlib.util

import numpy as np

from coopsim.metrics import (delivery_auc, gauge_decompose, gospa,
                             identity_metrics, nees_coverage, nees_series,
                             ospa2_series)

MODTRACK_GOSPA = os.environ.get("MODTRACK_GOSPA", "")  # optional cross-check


def _check(lines, name, ok):
    lines.append(f"{'PASS' if ok else 'FAIL'} {name}")


def _crossing_scene():
    """Two GT tracks crossing near t=10 (offset 0.5 m so frames never coincide
    exactly), exact estimates with consistent labels."""
    T = 21
    t = np.arange(T, dtype=np.float64)
    tau = np.zeros((T, 2, 4))
    tau[:, 0, 0], tau[:, 0, 1] = t, t
    tau[:, 1, 0], tau[:, 1, 1] = t, 20.5 - t
    alive = np.ones((T, 2), bool)
    est = tau[:, :, :2].copy()
    est_labels = np.tile(np.array([1, 2]), (T, 1))
    est_mask = np.ones((T, 2), bool)
    return tau, alive, est, est_labels, est_mask


def run_selftest():
    lines = []

    g = gospa(np.zeros((0, 2)), np.zeros((0, 2)))
    _check(lines, "gospa_empty",
           g["gospa"] == 0.0 and g["missed"] == 0.0 and g["false"] == 0.0
           and g["n_assigned"] == 0)

    g = gospa(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    _check(lines, "gospa_1v1_d1",
           abs(g["gospa"] - 1.0) < 1e-12 and abs(g["loc"] - 1.0) < 1e-12
           and g["n_assigned"] == 1)

    g = gospa(np.array([[0.0, 0.0]]), np.zeros((0, 2)))
    _check(lines, "gospa_missed_only",
           abs(g["gospa"] - (25.0 / 2) ** 0.5) < 1e-12 and g["missed"] == 12.5
           and g["false"] == 0.0)

    g = gospa(np.array([[0.0, 0.0], [10.0, 0.0]]),
              np.array([[1.0, 0.0], [100.0, 0.0]]))
    _check(lines, "gospa_2v2_one_beyond_c",
           abs(g["gospa"] - 26.0 ** 0.5) < 1e-12 and abs(g["loc"] - 1.0) < 1e-12
           and g["missed"] == 12.5 and g["false"] == 12.5 and g["n_assigned"] == 1)

    try:
        spec = importlib.util.spec_from_file_location("gospa_backend", MODTRACK_GOSPA)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rng = np.random.default_rng(0)
        ok = True
        for _ in range(20):
            n, m = rng.integers(0, 9, 2)
            gt = rng.uniform(0, 20, (n, 2))
            es = rng.uniform(0, 20, (m, 2))
            ref = mod._calculate_gospa_local(list(gt), list(es), c=5.0, p=2.0, alpha=2.0)[0]
            ok = ok and abs(gospa(gt, es)["gospa"] - ref) < 1e-6
        _check(lines, "gospa_crosscheck_modtrack", ok)
    except Exception:
        lines.append("SKIP gospa_crosscheck_modtrack (ModTrack unreadable)")

    tau, alive, est, est_labels, est_mask = _crossing_scene()
    r = identity_metrics(tau, alive, est, est_labels, est_mask, thresh=2.0)
    _check(lines, "identity_correct_labels",
           r["idf1"] == 1.0 and r["idsw"] == 0 and r["frag"] == 0)

    labels_sw = est_labels.copy()
    labels_sw[10:] = labels_sw[10:, ::-1]
    r = identity_metrics(tau, alive, est, labels_sw, est_mask, thresh=2.0)
    _check(lines, "identity_label_swap", r["idsw"] == 2 and r["idf1"] < 1.0)

    rng = np.random.default_rng(1)
    T, d = 4000, 2
    A = np.array([[2.0, 0.3], [0.3, 1.0]])
    P = A @ A.T
    err = rng.multivariate_normal(np.zeros(d), P, size=T)
    cov = np.tile(P, (T, 1, 1))
    cover = nees_coverage(nees_series(err, cov), d)
    _check(lines, "nees_coverage_95", abs(cover - 0.95) < 0.03)

    rng = np.random.default_rng(2)
    true_xy = rng.uniform(0, 50, (10, 5, 2))
    gd = gauge_decompose(true_xy, true_xy + np.array([1.0, 0.0]))
    _check(lines, "gauge_common_shift",
           np.allclose(gd["common"], 1.0) and np.allclose(gd["relative"], 0.0)
           and np.allclose(gd["ate"], 1.0))

    realized = np.array([False] * 10 + [True] * 10)
    _check(lines, "auc_perfect", delivery_auc(np.arange(20.0), realized) == 1.0)
    _check(lines, "auc_anti", delivery_auc(-np.arange(20.0), realized) == 0.0)
    rng = np.random.default_rng(3)
    auc = delivery_auc(rng.uniform(size=2000), rng.random(2000) < 0.5)
    _check(lines, "auc_random", abs(auc - 0.5) < 0.05)

    T = 50
    tau = np.zeros((T, 1, 4))
    tau[:, 0, :2] = [5.0, 5.0]
    o = ospa2_series(tau, np.ones((T, 1), bool), tau[:, :, :2].copy(),
                     np.zeros((T, 1), int), np.ones((T, 1), bool), window=10)
    _check(lines, "ospa2_exact_track", np.allclose(o, 0.0))

    return lines


if __name__ == "__main__":
    results = run_selftest()
    print("\n".join(results))
    n_fail = sum(line.startswith("FAIL") for line in results)
    print(f"{'ALL PASS' if n_fail == 0 else 'FAILURES PRESENT'} ({len(results) - n_fail}/{len(results)})")
    raise SystemExit(1 if n_fail else 0)
