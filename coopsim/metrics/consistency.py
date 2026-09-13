"""Filter consistency: NEES series and chi-square coverage."""
import numpy as np
from scipy.stats import chi2


def nees_series(err, cov):
    """NEES per step: err (T,d), cov (T,d,d). Returns (T,). Uses a batched
    solve, no explicit inverse."""
    err = np.asarray(err, np.float64)
    sol = np.linalg.solve(np.asarray(cov, np.float64), err[..., None])[..., 0]
    return np.einsum("td,td->t", err, sol)


def nees_coverage(nees, d):
    """Fraction of NEES values inside the two-sided 95 percent chi-square
    envelope for d dof."""
    lo, hi = chi2.ppf(0.025, d), chi2.ppf(0.975, d)
    nees = np.asarray(nees, np.float64)
    return float(np.mean((nees >= lo) & (nees <= hi)))
