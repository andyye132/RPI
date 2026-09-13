"""Does the safety dichotomy hold on the quantity the theorem is actually about?

The paper's separation proposition claims inter-agent separation constraints stay feasible while
map-referenced obstacle constraints do not, because the first is gauge-annihilating
and the second is gauge-covariant. Two quantities decide it, and the paper's logged
"relative" metric is NEITHER:

  ABSOLUTE   e_abs(t) = rms_i || p_i_hat - p_i_true ||
             drives the obstacle margin. Gauge-covariant, expected to grow.

  SEPARATION e_sep(t) = rms_{i<j} | ||p_i_hat - p_j_hat|| - ||p_i_true - p_j_true|| |
             drives the inter-agent margin. This is the BASELINE-PROJECTED error,
             exactly gauge-invariant on all three generators because the constraint
             normal n_ij annihilates E_ij G. Expected to stay bounded.

The logged `relative` metric is rms scatter about the team mean, which is not
e_sep: it retains rotational and shape modes that barely move pairwise distance.
Fitting sqrt(t) to it gives shape growing FASTER than the common mode, which
would kill the dichotomy. That fit is on the wrong object, so this runner
computes both correct quantities directly from the saved poses.

Reads npz written by harness.run_episode: x_est (T,N,3) and the truth it was
scored against. Env: DICH_GLOB (default the supplied-field A_J runs).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim.paths import RUNS


def rates(e, t_s, lo=0.25):
    """Least-squares k in e = k*sqrt(t), fit over the last (1-lo) of the episode
    so the transient does not dominate. Returns k and the final-third mean.

    WARNING, and the reason `growth` below exists: this fit has NO INTERCEPT, so
    it is a level readout, not a growth rate. A perfectly CONSTANT signal at
    level L returns k = L * sum(sqrt t)/sum(t), which on this fit window is
    L * 0.0719 -- and that reproduces the previously reported separation rate
    0.1154 to 0.6% from the observed level 1.5965 m. Do not quote k as evidence
    that anything grows. Kept only because the published numbers were computed
    with it and must remain reproducible.
    """
    m = t_s > lo * t_s[-1]
    x, y = np.sqrt(t_s[m]), e[m]
    k = float((x * y).sum() / max((x * x).sum(), 1e-30))
    f = float(e[t_s > (2.0 / 3.0) * t_s[-1]].mean())
    return k, f


def growth(e, t_s, lo=0.25):
    """Growth of e against sqrt(t) WITH an intercept, plus a shape check.

    Returns (b, level, frac) where b is the slope on sqrt(t) after allowing a
    constant, level is the mean over the fit window, and frac is the fraction of
    the zero-intercept fit's implied rise that is actually realized across the
    window (observed third3-third1 divided by what k would require). frac near 0
    means the signal is flat and k is reporting its level.

    This is the statistic the bounded-vs-growing question must be decided on.
    """
    m = t_s > lo * t_s[-1]
    x, y = np.sqrt(t_s[m]), e[m]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    k = float((x * y).sum() / max((x * x).sum(), 1e-30))
    yb, tb = np.array_split(y, 3), np.array_split(t_s[m], 3)
    req = k * (np.sqrt(tb[2].mean()) - np.sqrt(tb[0].mean()))
    obs = yb[2].mean() - yb[0].mean()
    return float(coef[1]), float(y.mean()), float(obs / req) if req else np.nan


def main():
    pat = os.environ.get("DICH_GLOB", str(RUNS / "urban_R1_A_J_s*.npz"))
    files = sorted(Path(pat).parent.glob(Path(pat).name))
    if not files:
        print(f"no files matching {pat}")
        return
    print(f"dichotomy check on {len(files)} runs matching {Path(pat).name}", flush=True)

    ka, ks, fa, fs, ba, bs = [], [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        keys = set(d.files)
        if "x_est" not in keys:
            print(f"  {f.name}: no x_est (keys: {sorted(keys)[:8]}...)")
            continue
        Xe = np.asarray(d["x_est"], np.float64)
        tk = next((k for k in ("x_true", "X_true", "truth", "x_gt") if k in keys), None)
        if tk is None:
            print(f"  {f.name}: no truth array (keys: {sorted(keys)})")
            continue
        Xt = np.asarray(d[tk], np.float64)
        T = min(len(Xe), len(Xt))
        Pe, Pt = Xe[:T, :, :2], Xt[:T, :, :2]
        t_s = np.arange(T) * 0.1

        e_abs = np.sqrt(((Pe - Pt) ** 2).sum(-1)).mean(1)

        de = np.linalg.norm(Pe[:, :, None, :] - Pe[:, None, :, :], axis=-1)
        dt_ = np.linalg.norm(Pt[:, :, None, :] - Pt[:, None, :, :], axis=-1)
        iu = np.triu_indices(Pe.shape[1], 1)
        e_sep = np.abs(de[:, iu[0], iu[1]] - dt_[:, iu[0], iu[1]]).mean(1)

        a, af = rates(e_abs, t_s)
        s, sf = rates(e_sep, t_s)
        ba.append(growth(e_abs, t_s)); bs.append(growth(e_sep, t_s))
        ka.append(a); ks.append(s); fa.append(af); fs.append(sf)
        print(f"  {f.name}: abs rate {a:.4f} (final {af:.2f} m)   "
              f"sep rate {s:.4f} (final {sf:.2f} m)", flush=True)

    if not ka:
        print("nothing computed")
        return
    ka, ks = np.array(ka), np.array(ks)
    print("\n=== RESULT ===", flush=True)
    print(f"  n runs                    {len(ka)}", flush=True)
    print(f"  ABSOLUTE   rate {ka.mean():.4f} +/- {ka.std(ddof=1):.4f}  "
          f"final-third mean {np.mean(fa):.2f} m", flush=True)
    print(f"  SEPARATION rate {ks.mean():.4f} +/- {ks.std(ddof=1):.4f}  "
          f"final-third mean {np.mean(fs):.2f} m", flush=True)
    ratio = ka.mean() / max(ks.mean(), 1e-30)
    print(f"  ratio abs/sep             {ratio:.2f}", flush=True)
    print("  ^ ZERO-INTERCEPT fits: these are LEVEL readouts, not growth rates,"
          "\n    and the verdict below does not use them. See rates().", flush=True)

    bam, bsm = np.array(ba), np.array(bs)

    def rep(tag, g):
        b, lv, fr = g[:, 0], g[:, 1], g[:, 2]
        se = b.std(ddof=1) / np.sqrt(len(b))
        t = b.mean() / se if se > 0 else float("nan")
        print(f"  {tag:<11}slope {b.mean():+.4f} +/- {se:.4f}  t = {t:+.2f}   "
              f"level {lv.mean():.3f} m   realized/implied rise {fr.mean():.2f}",
              flush=True)
        return b.mean(), t

    print("\n=== GROWTH, fitted WITH an intercept (the deciding statistic) ===",
          flush=True)
    b_abs, t_abs = rep("ABSOLUTE", bam)
    b_sep, t_sep = rep("SEPARATION", bsm)
    gratio = b_abs / max(abs(b_sep), 1e-30)
    print(f"  growth ratio abs/sep      {gratio:.2f}", flush=True)

    print("\n  Dichotomy needs separation BOUNDED (slope ~ 0) while absolute grows.",
          flush=True)
    if abs(t_sep) < 2.0 and t_abs > 2.0 and gratio >= 5.0:
        print("  VERDICT: HOLDS - absolute grows, separation does not", flush=True)
    elif abs(t_sep) < 2.0 and t_abs > 2.0:
        print(f"  VERDICT: NOT ESTABLISHED - absolute grows (t={t_abs:+.2f}) while"
              f" separation does\n           not (t={t_sep:+.2f}), but the growth "
              f"ratio {gratio:.2f} is under 5. The evidence\n           neither "
              f"supports nor refutes the dichotomy; it cannot resolve it.",
              flush=True)
    elif abs(t_sep) >= 2.0:
        print(f"  VERDICT: FAILS - separation genuinely grows (t={t_sep:+.2f})",
              flush=True)
    else:
        print(f"  VERDICT: INCONCLUSIVE - absolute does not grow either "
              f"(t={t_abs:+.2f})", flush=True)
    print("DONE diag_dichotomy", flush=True)


if __name__ == "__main__":
    main()
