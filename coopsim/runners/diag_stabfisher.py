"""Direct test of Prop. 2 (prop:stab): the STABILIZER-NULL of the channel Fisher.

Measures the object the proposition constrains -- channel-factor information
restricted to the gauge subspace -- so no estimator, no error, and no shape
confound. The range term -10 gamma log10 d is exactly gauge-null at every
linearization point (J skew), which is why the confound drops out analytically;
check U1 verifies it numerically.

PARAMETERIZATION (canonical, base point at the world ORIGIN, t-independent):
  xi = (v_x, v_y, omega),  flow on u:  du = v + omega J u,  J = [[0,-1],[1,0]].
  Per robot i at p_i:  L_i = [[1,0,-p_iy],[0,1,p_ix],[0,0,1]].
  per link (i<j):      a = (grad_i pr)^T L_i + (grad_j pr)^T L_j          in R^3
  S = sum_t sum_{i<j} w a a^T  (3x3 PSD)    B = sum_t sum_i L_i^T L_i  (3x3 SPD)
  Readouts from the GENERALIZED problem S v = mu B v, invariant under every
  nonsingular reparameterization xi -> T xi.
NOT the per-slot centroid basis of run_gauge_fisher: rotation about the moving
centroid gives a t-dependent basis, so a fixed se(2) element has drifting
coordinates and sum_t of rank-2 terms is generically full rank (kills radial).

PRIMARY weights are w == 1 on all N(N-1)/2 pairs at F_true/X_true: the claim is a
property of the Jacobian, and the converse hypothesis R^{-1} > 0 is violated by
realized weights (zeroed on undelivered links). Realized weights are reported in
section D as an operational secondary, never as the confirmatory number.

Predicted stabilizer elements:
  stratified F(<u,n>), n = (cos tn, sin tn) -> xi = (-sin tn, cos tn, 0)
  radial about c                            -> xi = (c_y, -c_x, 1)
  constant -> all of R^3 (S == 0)           urban / warped placebo -> none

Run: SF_SEEDS=8 SF_STEPS=3000 python -u -m coopsim.runners.diag_stabfisher
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")      # must precede `import jax`
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax.scipy.ndimage import map_coordinates

from coopsim import channel, estimator, world
from coopsim.config import crn_keys, load_config
from coopsim.field import emission
from coopsim.runners.run_m1_plateau import make_sched

SEEDS = int(os.environ.get("SF_SEEDS", "8"))
STEPS = int(os.environ.get("SF_STEPS", "3000"))
EVERY = int(os.environ.get("SF_EVERY", "10"))
GRID = int(os.environ.get("SF_GRID", "40"))              # repo grid, main table
SGRID = int(os.environ.get("SF_SWEEP_GRID", "160"))      # sweeps: off-axis strata and
#   the radial bowl are NOT exactly null on a Cartesian raster (only axis-aligned
#   strata are, by bilinear separability), so the swept arms need h*k small.
PERIOD = float(os.environ.get("SF_PERIOD_M", "100.0"))   # one half-period across arena
NANG = int(os.environ.get("SF_ANGLES", "12"))
REFINE = [int(x) for x in os.environ.get("SF_REFINE", "40,80,160,320").split(",")]
REALIZED = os.environ.get("SF_REALIZED", "1") == "1"
TOL_NULL = float(os.environ.get("SF_TOL_NULL", "1e-6"))   # C1
TOL_FULL = float(os.environ.get("SF_TOL_FULL", "1e-2"))   # C4/C5
TOL_ANG = float(os.environ.get("SF_TOL_ANG", "2.0"))      # C2, degrees
TOL_EPS = float(os.environ.get("SF_TOL_EPS", "0.05"))     # radial rasterization gate
CRN_TAG = os.environ.get("SF_CRN", "SF")   # fixed, INDEPENDENT of the field kind

# ------------------------------------------------------------------ fields
# Built here rather than in world.py: the swept designs need the stratification
# normal, the radial centre and the grid resolution as knobs, and world.py hardwires
# all three. Same dB/m level convention as world.py:70 (mean(urban_loss_db)/5).


def _grid_xy(G, ext):
    yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")   # row<->y, col<->x
    return xx / (G - 1) * ext, yy / (G - 1) * ext


def f_stratified(G, ext, lvl, tn, period):
    X, Y = _grid_xy(G, ext)
    s = np.cos(tn) * X + np.sin(tn) * Y
    return (lvl * (1.0 + np.cos(2 * np.pi * s / period)) / 2.0).astype(np.float64)


def f_radial(G, ext, lvl, c, period):
    X, Y = _grid_xy(G, ext)
    r = np.hypot(X - c[0], Y - c[1])
    return (lvl * (1.0 + np.cos(2 * np.pi * r / period)) / 2.0).astype(np.float64)


def f_warped(G, ext, lvl, tn, period, amp=6.0, wl=25.0):
    """Placebo: strata warped by a sin ripple. Same gradient statistics, no symmetry."""
    X, Y = _grid_xy(G, ext)
    s = np.cos(tn) * X + np.sin(tn) * Y
    p = -np.sin(tn) * X + np.cos(tn) * Y
    s = s + amp * np.sin(2 * np.pi * p / wl)
    return (lvl * (1.0 + np.cos(2 * np.pi * s / period)) / 2.0).astype(np.float64)


def f_repo(kind, cfg, keys):
    cfg.world.kind = kind
    return np.asarray(world.make_world(keys.world_gen, cfg).F_true, np.float64)


# ------------------------------------------------- interpolant-gradient leakage gate
_GRAD_CACHE = {}


def _interp_grad(F, ext, pts):
    G = F.shape[0]
    fn = _GRAD_CACHE.get((G, ext))
    if fn is None:
        def val(Ff, p):
            return map_coordinates(Ff, [jnp.array([p[1] / ext * (G - 1)]),
                                        jnp.array([p[0] / ext * (G - 1)])],
                                   order=1, mode="constant", cval=0.0)[0]
        fn = jax.jit(jax.vmap(jax.grad(val, argnums=1), in_axes=(None, 0)))
        _GRAD_CACHE[(G, ext)] = fn
    return np.asarray(fn(jnp.asarray(F), jnp.asarray(pts)))


def eps_leak(F, ext, flow, npts=20000):
    """RMS component of grad F_interp ALONG the claimed stabilizer flow, relative to
    RMS |grad F_interp|. Exactly 0 iff the rasterized field really has the symmetry.
    flow(pts) -> (M,2) unit vectors, or None if no symmetry is claimed."""
    if flow is None:
        return float("nan")
    rng = np.random.default_rng(12345)
    pts = rng.uniform(2.0, ext - 2.0, (npts, 2))
    v = flow(pts)
    keep = np.isfinite(v).all(1)
    pts, v = pts[keep], v[keep]
    g = _interp_grad(jnp.asarray(F), ext, pts)
    den = np.sqrt((g ** 2).sum(1).mean())
    if den == 0.0:
        return 0.0
    return float(np.sqrt(((g * v).sum(1) ** 2).mean()) / den)


def flow_trans(xi):
    v = np.asarray(xi[:2], float)
    v = v / np.linalg.norm(v)
    return lambda pts: np.broadcast_to(v, pts.shape).copy()


def flow_rot(c):
    def f(pts):
        d = pts - np.asarray(c, float)
        r = np.linalg.norm(d, axis=1, keepdims=True)
        out = np.stack([-d[:, 1], d[:, 0]], 1) / np.where(r < 2.0, np.nan, r)
        return out
    return f


# ------------------------------------------------------------------ the instrument
_SB_CACHE = {}


def make_SB(ext, p0, gamma, n, base=(0.0, 0.0)):
    """vmapped (S,B) over a batch of slots. All N(N-1)/2 pairs, unit weights.
    F is a runtime argument so the kernel compiles once per (n, base, ext)."""
    key = (ext, p0, gamma, n, base)
    if key in _SB_CACHE:
        return _SB_CACHE[key]
    iu, ju = np.triu_indices(n, 1)          # i<j only: no self-pairs, no double count
    iu, ju = jnp.asarray(iu), jnp.asarray(ju)
    b = jnp.asarray(base, jnp.float64)

    def one(Ff, P):
        g2 = jax.grad(lambda a, q: emission(a, q, Ff, ext, P0=p0, gamma=gamma),
                      argnums=(0, 1))
        Pi, Pj = P[iu], P[ju]
        gi, gj = jax.vmap(g2)(Pi, Pj)
        Qi, Qj = Pi - b, Pj - b
        a2 = (-gi[:, 0] * Qi[:, 1] + gi[:, 1] * Qi[:, 0]
              - gj[:, 0] * Qj[:, 1] + gj[:, 1] * Qj[:, 0])
        A = jnp.concatenate([gi + gj, a2[:, None]], -1)              # (M,3)
        Q = P - b
        sx, sy, sq = Q[:, 0].sum(), Q[:, 1].sum(), (Q ** 2).sum()
        B = jnp.array([[1.0 * n, 0.0, -sy], [0.0, 1.0 * n, sx],
                       [-sy, sx, sq + n]])
        return A.T @ A, B

    fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    _SB_CACHE[key] = fn
    return fn


def accumulate(F, X, ext, p0, gamma, base=(0.0, 0.0), chunk=64, every=None):
    """S,B accumulated over the episode's slots (float64)."""
    n = X.shape[1]
    fn = make_SB(ext, p0, gamma, n, base)
    Fj = jnp.asarray(F, jnp.float64)
    ts = np.arange(1, len(X), EVERY if every is None else every)
    S = np.zeros((3, 3))
    B = np.zeros((3, 3))
    for k in range(0, len(ts), chunk):
        P = jnp.asarray(X[ts[k:k + chunk], :, :2], jnp.float64)
        Sc, Bc = fn(Fj, P)
        S += np.asarray(Sc, np.float64).sum(0)
        B += np.asarray(Bc, np.float64).sum(0)
    return 0.5 * (S + S.T), 0.5 * (B + B.T)


def spectrum(S, B):
    mu, V = scipy.linalg.eigh(0.5 * (S + S.T), 0.5 * (B + B.T))
    return mu[::-1].copy(), V[:, ::-1].copy()          # descending; V is B-orthonormal


def readouts(S, B, xi_pred):
    mu, V = spectrum(S, B)
    mu1 = float(mu[0])
    r = float(mu[2] / mu1) if mu1 > 0 else float("nan")
    p = np.clip(mu, 0, None)
    p = p / p.sum() if p.sum() > 0 else p
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if p.sum() > 0 else 0.0
    ang = float("nan")
    if xi_pred is not None and mu1 > 0:
        v3, x = V[:, 2], np.asarray(xi_pred, float)
        num = abs(v3 @ B @ x)
        den = np.sqrt((v3 @ B @ v3) * (x @ B @ x))
        ang = float(np.degrees(np.arccos(np.clip(num / den, 0, 1))))
    return mu, V, r, erank, ang


def trans_circle(S, B):
    """Continuous variant: q(phi) = xi^T S xi / xi^T B xi on the unit circle of PURE
    TRANSLATIONS xi = (cos phi, sin phi, 0). Returns (phi_min_deg, q_min/mu1, curve).
    phi_min is the exact 2x2 generalized eigen-angle; the curve is for the figure."""
    mu2, V2 = scipy.linalg.eigh(S[:2, :2], B[:2, :2])
    phi = float(np.degrees(np.arctan2(V2[1, 0], V2[0, 0])) % 180.0)
    mu1 = float(spectrum(S, B)[0][0])
    grid = np.radians(np.arange(0, 180, 15.0))
    xi = np.stack([np.cos(grid), np.sin(grid), np.zeros_like(grid)], 1)
    q = np.einsum("ki,ij,kj->k", xi, S, xi) / np.einsum("ki,ij,kj->k", xi, B, xi)
    if mu1 <= 0:                       # S == 0 (constant field): ratios undefined
        return phi, float("nan"), np.full_like(q, np.nan)
    return phi, float(mu2[0] / mu1), q / mu1


# ------------------------------------------------------------------ episode geometry
def geometry(cfg, seed):
    """Trajectories under a FIXED CRN tag: byte-identical across every field arm."""
    keys = crn_keys(CRN_TAG, "P3", seed)
    X, _ = world.make_trajectories(keys.target_gen, cfg)
    return keys, np.asarray(X, np.float64)


def arm(name, label, Fmk, xi_pred, flow, verdict, Xs, cfg, note=""):
    """Run one field arm over all seeds and print its row + per-seed spread."""
    ext, p0, gam = cfg.world.extent, cfg.channel.p0_dbm, cfg.channel.gamma
    rs, angs, mus, smax = [], [], [], []
    Spool = np.zeros((3, 3))
    Bpool = np.zeros((3, 3))
    for X in Xs:
        F = Fmk()
        S, B = accumulate(F, X, ext, p0, gam)
        Spool += S
        Bpool += B
        mu, _, r, _, a = readouts(S, B, xi_pred)
        rs.append(r); angs.append(a); mus.append(mu[0]); smax.append(np.abs(S).max())
    mu, V, r, erank, ang = readouts(Spool, Bpool, xi_pred)
    phi, qmin, curve = trans_circle(Spool, Bpool)
    el = eps_leak(Fmk(), ext, flow)
    ok = verdict(r, ang, el, np.array(smax))
    print(f"{name:<18}{label:<12}{el:>9.2e}{mu[0]:>11.3e}{mu[1]:>11.3e}"
          f"{mu[2]:>11.3e}{r:>11.2e}{erank:>7.2f}"
          f"{('--' if np.isnan(ang) else f'{ang:.3f}'):>10}{phi:>9.2f}"
          f"   {ok}", flush=True)
    print(f"{'':<18}per-seed r: " + " ".join(f"{x:.1e}" for x in rs)
          + (f" | per-seed ang(deg): " + " ".join(f"{x:.2f}" for x in angs)
             if not np.isnan(angs[0]) else "")
          + f" | max|S|={max(smax):.3e} cond(B)={np.linalg.cond(Bpool):.2e}"
          + (f"  {note}" if note else ""), flush=True)
    print(f"{'':<18}q(phi)/mu1 on the translation circle, phi=0..165 deg step 15: "
          + " ".join(f"{x:.2e}" for x in curve), flush=True)
    return dict(name=name, S=Spool, B=Bpool, mu=mu, V=V, r=r, ang=ang, eps=el,
                phi=phi, ok=ok, smax=max(smax))


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    cfg.world.trajectory = "transect"
    cfg.time.steps = STEPS
    cfg.shapes.grid = GRID
    cfg.channel.ge_p_bad_los = 0.0
    cfg.channel.ge_p_bad_nlos = 0.0
    n, ext = cfg.shapes.n_robots, cfg.world.extent
    p0, gam = cfg.channel.p0_dbm, cfg.channel.gamma
    lvl = float(np.mean(cfg.world.urban_loss_db)) / 5.0
    ctr = (ext / 2.0, ext / 2.0)
    tn0 = np.pi / 2.0                                   # world.py's stratified normal
    x64 = jnp.zeros(1).dtype == jnp.float64

    print("=" * 118, flush=True)
    print("prop:stab STABILIZER-FISHER TEST -- generalized spectrum of (S, B), "
          "S = gauge-restricted channel Fisher", flush=True)
    print(f"N={n} grid={GRID} extent={ext} steps={STEPS} every={EVERY} "
          f"seeds={SEEDS} crn_tag={CRN_TAG!r} lvl={lvl:.3f} dB/m  x64={x64}",
          flush=True)
    print("primary weights: w==1 on all N(N-1)/2 pairs at F_true/X_true "
          "(proof makes the null weight-independent; converse needs R^-1 > 0)",
          flush=True)
    print("=" * 118, flush=True)

    keys0, X0 = geometry(cfg, 0)
    Xs = [geometry(cfg, s)[1] for s in range(SEEDS)]

    # ---------------------------------------------------------- U: unit tests
    print("\n[U] INSTRUMENT UNIT TESTS (all must pass before any row is interpreted)",
          flush=True)
    S_ref = accumulate(f_repo("urban", cfg, keys0), X0, ext, p0, gam)[0]
    ref = np.abs(S_ref).max()
    Sz, Bz = accumulate(np.zeros((GRID, GRID)), X0, ext, p0, gam)
    u1 = np.abs(Sz).max()
    print(f"  U1 F==0 (range term only) -> max|S| = {u1:.3e}   "
          f"rel. to urban max|S| {ref:.3e}: {u1/ref:.1e}   "
          f"{'PASS' if u1 / ref < 1e-12 else 'FAIL'}"
          "   [the shape/distance confound of attempt 2 is analytically absent]",
          flush=True)
    Sc = accumulate(f_repo("constant", cfg, keys0), X0, ext, p0, gam)[0]
    u2 = np.abs(Sc).max()
    print(f"  U2 constant field         -> max|S| = {u2:.3e}   rel: {u2/ref:.1e}   "
          f"{'PASS' if u2 / ref < 1e-12 else 'FAIL'}", flush=True)
    Fst = f_stratified(GRID, ext, lvl, tn0, PERIOD)
    xi_st = np.array([-np.sin(tn0), np.cos(tn0), 0.0])
    rng = np.random.default_rng(7)
    Xn = X0.copy()
    Xn[:, :, :2] += rng.normal(0, 0.5, Xn[:, :, :2].shape)
    r_cl = readouts(*accumulate(Fst, X0, ext, p0, gam), xi_st)[2]
    r_ns = readouts(*accumulate(Fst, Xn, ext, p0, gam), xi_st)[2]
    print(f"  U4 X_true + 0.5 m noise   -> stratified r: clean {r_cl:.2e} -> "
          f"noisy {r_ns:.2e}   {'PASS' if r_ns < TOL_NULL else 'FAIL'}"
          "   [null is independent of the pose linearization point]", flush=True)
    mu_o = spectrum(*accumulate(Fst, X0, ext, p0, gam))[0]
    mu_c = spectrum(*accumulate(Fst, X0, ext, p0, gam, base=ctr))[0]
    mu_t = spectrum(*accumulate(Fst, X0, ext, p0, gam,
                                base=tuple(X0[1, :, :2].mean(0))))[0]
    d5 = max(np.abs(mu_c / mu_o - 1).max(), np.abs(mu_t / mu_o - 1).max())
    print(f"  U5 base point origin/centre/centroid -> max rel. change in mu: "
          f"{d5:.2e}   {'PASS' if d5 < 1e-8 else 'FAIL'}"
          "   [guards the moving-centroid basis bug]", flush=True)

    # ---------------------------------------------------------- A: main table
    print("\n[A] MAIN TABLE (urban control FIRST; pooled over seeds; "
          "eps = rasterization leakage along the claimed stabilizer flow)", flush=True)
    print(f"{'kind':<18}{'role':<12}{'eps_leak':>9}{'mu1':>11}{'mu2':>11}{'mu3':>11}"
          f"{'mu3/mu1':>11}{'erank':>7}{'ang(v3)':>10}{'phi_min':>9}   verdict",
          flush=True)
    P = PERIOD
    full = lambda r, a, e, sm: ("PASS (full rank, no null)" if r > TOL_FULL
                                else "FAIL (spurious small mu3)")
    nul = lambda r, a, e, sm: ("PASS (rank 2, null on xi_pred)"
                               if (r < TOL_NULL and a < TOL_ANG)
                               else "FAIL")
    res = {}
    res["urban"] = arm("urban", "CONTROL", lambda: f_repo("urban", cfg, keys0),
                       None, None, full, Xs, cfg, "negative control (repo field)")
    res["placebo"] = arm("warped-strata", "PLACEBO",
                         lambda: f_warped(GRID, ext, lvl, tn0, P), None, None,
                         full, Xs, cfg, "matched gradient stats, symmetry destroyed")
    res["constant"] = arm(
        "constant", "UNIT TEST", lambda: f_repo("constant", cfg, keys0),
        None, None,
        lambda r, a, e, sm: ("PASS (S==0, U2)" if sm.max() / ref < 1e-12 else "FAIL"),
        Xs, cfg, "grad F == 0 exactly: NOT EVIDENCE, implementation check only")
    res["strat_repo"] = arm(
        "stratified-repo", "PRIMARY",
        lambda: f_repo("stratified", cfg, keys0), xi_st, flow_trans(xi_st), nul,
        Xs, cfg, "as shipped in world.py (n=y_hat, period 12.5 m)")
    res["strat_long"] = arm(
        "stratified-long", "PRIMARY", lambda: f_stratified(GRID, ext, lvl, tn0, P),
        xi_st, flow_trans(xi_st), nul, Xs, cfg,
        f"swept-design profile (period {P:g} m)")
    xi_rad = np.array([ctr[1], -ctr[0], 1.0])
    gate = lambda r, a, e, sm: (("PASS (rank 2, null on xi_pred)"
                                 if (r < TOL_NULL and a < TOL_ANG) else "INCONCLUSIVE")
                                if e < TOL_EPS else
                                f"NOT TESTABLE (eps_leak {e:.2f} >= {TOL_EPS})")
    res["rad_repo"] = arm(
        "radial-repo", "GATED", lambda: f_repo("radial", cfg, keys0), xi_rad,
        flow_rot(ctr), gate, Xs, cfg, "as shipped (4 rings): expected to fail the gate")
    res["rad_bowl"] = arm(
        "radial-bowl", "GATED", lambda: f_radial(GRID, ext, lvl, ctr, 2 * ext),
        xi_rad, flow_rot(ctr), gate, Xs, cfg, "single half-period bowl")

    # ---------------------------------------------------------- B: theta_n sweep
    print(f"\n[B] PRIMARY ARM -- stratified normal swept over {NANG} angles "
          f"(period {P:g} m, grid {SGRID}), geometry held byte-identical (CRN)",
          flush=True)
    print("    prediction: null xi = (-sin tn, cos tn, 0); the angle of minimum "
          "information phi_min = tn + 90 deg exactly", flush=True)
    print("    NOTE: only AXIS-ALIGNED strata are exactly null after bilinear "
          "interpolation; off-axis rows carry an O(h) rasterization floor, which is "
          "why the sweep runs at a refined grid and why [C] checks the h-scaling.",
          flush=True)
    print(f"{'tn(deg)':>9}{'eps_leak':>10}{'mu3/mu1':>11}{'ang(v3)':>10}"
          f"{'phi_min':>10}{'pred phi':>10}{'resid':>9}   pass", flush=True)
    tns = np.linspace(0.0, np.pi, NANG, endpoint=False)
    meas, preds, allok = [], [], True
    for tn in tns:
        F = f_stratified(SGRID, ext, lvl, tn, P)
        xi = np.array([-np.sin(tn), np.cos(tn), 0.0])
        S = np.zeros((3, 3)); B = np.zeros((3, 3))
        rr, aa = [], []
        for X in Xs:
            s, b = accumulate(F, X, ext, p0, gam)
            S += s; B += b
            o = readouts(s, b, xi)
            rr.append(o[2]); aa.append(o[4])
        mu, V, r, _, ang = readouts(S, B, xi)
        phi, _, _ = trans_circle(S, B)
        pp = (np.degrees(tn) + 90.0) % 180.0
        resid = (phi - pp + 90.0) % 180.0 - 90.0
        el = eps_leak(F, ext, flow_trans(xi))
        ok = (max(rr) < TOL_NULL) and (max(aa) < TOL_ANG)
        allok &= ok
        meas.append(phi); preds.append(np.degrees(tn))
        print(f"{np.degrees(tn):>9.1f}{el:>10.2e}{r:>11.2e}{ang:>10.3f}"
              f"{phi:>10.2f}{pp:>10.2f}{resid:>9.3f}   "
              f"{'PASS' if ok else 'FAIL'}  (worst-seed r={max(rr):.1e} "
              f"ang={max(aa):.2f})", flush=True)
    m = np.asarray(meas)
    m = np.degrees(np.unwrap(np.radians(2 * m)) / 2)      # unwrap with period 180
    A = np.stack([np.asarray(preds), np.ones(len(preds))], 1)
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    pred = A @ coef
    ss = 1.0 - ((m - pred) ** 2).sum() / max(((m - m.mean()) ** 2).sum(), 1e-30)
    c3 = (0.98 <= coef[0] <= 1.02) and ss > 0.999
    print(f"    C3 regression phi_min ~ tn:  slope {coef[0]:.5f} (pred 1) "
          f"intercept {coef[1] % 180.0:.3f} deg (pred 90)  R^2 {ss:.6f}   "
          f"{'PASS' if c3 else 'FAIL'}", flush=True)

    # ---------------------------------------------------------- C: radial
    print("\n[C] RADIAL ARM -- centre sweep and grid-refinement convergence "
          "(gate: eps_leak < %.2f)" % TOL_EPS, flush=True)
    print(f"{'cx':>7}{'cy':>7}{'eps_leak':>10}{'mu3/mu1':>11}{'ang(v3)':>10}   status",
          flush=True)
    print(f"    centre sweep at grid {SGRID}; predicted null xi = (c_y, -c_x, 1)",
          flush=True)
    for dx in (-8.0, 0.0, 8.0):
        for dy in (-8.0, 0.0, 8.0):
            c = (ctr[0] + dx, ctr[1] + dy)
            F = f_radial(SGRID, ext, lvl, c, 2 * ext)
            xi = np.array([c[1], -c[0], 1.0])
            S = np.zeros((3, 3)); B = np.zeros((3, 3))
            for X in Xs:
                s, b = accumulate(F, X, ext, p0, gam)
                S += s; B += b
            _, _, r, _, ang = readouts(S, B, xi)
            el = eps_leak(F, ext, flow_rot(c))
            st = ("gate PASS" if el < TOL_EPS else "gate FAIL -> NOT TESTABLE")
            print(f"{c[0]:>7.1f}{c[1]:>7.1f}{el:>10.2e}{r:>11.2e}{ang:>10.3f}   {st}",
                  flush=True)
    print("    grid refinement: discretization leakage must FALL with h, a "
          "conditioning floor is h-independent. Axis-aligned strata (r_strat0) is "
          "exactly null at every h -- the free internal control.", flush=True)
    tn60 = np.radians(60.0)
    xi60 = np.array([-np.sin(tn60), np.cos(tn60), 0.0])
    print(f"{'G':>6}{'h(m)':>8}{'eps_rad':>10}{'r_radial':>11}{'eps_st60':>10}"
          f"{'r_strat60':>11}{'r_strat0':>11}", flush=True)
    for G in REFINE:
        Fr = f_radial(G, ext, lvl, ctr, 2 * ext)
        F60 = f_stratified(G, ext, lvl, tn60, P)
        Fs = f_stratified(G, ext, lvl, tn0, P)
        rr = readouts(*accumulate(Fr, X0, ext, p0, gam), xi_rad)[2]
        r60 = readouts(*accumulate(F60, X0, ext, p0, gam), xi60)[2]
        rs = readouts(*accumulate(Fs, X0, ext, p0, gam), xi_st)[2]
        print(f"{G:>6}{ext / G:>8.3f}{eps_leak(Fr, ext, flow_rot(ctr)):>10.2e}"
              f"{rr:>11.2e}{eps_leak(F60, ext, flow_trans(xi60)):>10.2e}"
              f"{r60:>11.2e}{rs:>11.2e}", flush=True)

    # ---------------------------------------------------------- D: realized weights
    if REALIZED:
        print("\n[D] SECONDARY (OPERATIONAL, NEVER CONFIRMATORY): realized estimator "
              "weights on the realized link set", flush=True)
        print("    R^-1 is singular here (w=0 off the delivered/attempted sets), so "
              "this cannot speak to the converse.", flush=True)
        print(f"{'kind':<14}{'links/slot':>11}{'max|S|':>11}{'mu1':>11}{'mu2':>11}"
              f"{'mu3':>11}{'mu3/mu1':>11}{'ang(v3)':>10}"
              f"{'zero eig of M^c(3N)':>21}", flush=True)
        for kind, xi in (("urban", None), ("constant", None),
                         ("stratified", xi_st), ("radial", xi_rad)):
            cfg.world.kind = kind
            wld = world.make_world(keys0.world_gen, cfg)
            F = jnp.asarray(wld.F_true, jnp.float64)
            c = estimator.make_consts(cfg)
            lut = channel.nakagami_lut(getattr(cfg.channel.nakagami_m, kind))
            st = channel.init_state(keys0.fading, n)
            ch = jax.jit(lambda t, p, v, s, tx: channel.step(
                t, keys0, p, v, F, s, tx, lut, cfg.channel, ext, cfg.time.dt))
            blocks = jax.jit(lambda Xs, d: estimator._slot_channel_blocks(Xs, F, d, c))
            sched = make_sched(n, int(cfg.channel.budget_frac * n),
                               cfg.channel.conflict_radius)
            age = jnp.zeros(n * n)
            S = np.zeros((3, 3)); nz = []; nlink = []
            for t in range(1, STEPS):
                pos = jnp.asarray(X0[t, :, :2])
                vel = jnp.asarray((X0[t, :, :2] - X0[t - 1, :, :2]) / cfg.time.dt)
                tx, age = sched(pos, age)
                obs, st = ch(t, pos, vel, st, tx)
                if t % EVERY:
                    continue
                z = jnp.zeros((n, n))
                d = estimator.plain_slot(jnp.zeros((n, 3)), jnp.zeros((n, n, 3)),
                                         jnp.zeros((n, n), bool), obs.rssi_q,
                                         obs.success, obs.attempted, obs.success, z, z)
                Axx = np.asarray(blocks(jnp.asarray(X0[t]), d)[0], np.float64)
                P = X0[t, :, :2]
                L = np.zeros((3 * n, 3))
                L[0::3, 0] = 1.0; L[1::3, 1] = 1.0
                L[0::3, 2] = -P[:, 1]; L[1::3, 2] = P[:, 0]; L[2::3, 2] = 1.0
                S += L.T @ Axx @ L
                ev = np.linalg.eigvalsh(0.5 * (Axx + Axx.T))
                nz.append(int((np.abs(ev) <= 1e-12 * max(np.abs(ev).max(), 1e-300)).sum()))
                nlink.append(int(np.asarray(obs.attempted).sum()))
            S = 0.5 * (S + S.T)
            _, B = accumulate(F, X0, ext, p0, gam)
            mu, _, r, _, ang = readouts(S, B, xi)
            print(f"{kind:<14}{np.mean(nlink):>11.2f}{np.abs(S).max():>11.3e}"
                  f"{mu[0]:>11.3e}{mu[1]:>11.3e}"
                  f"{mu[2]:>11.3e}{r:>11.2e}"
                  f"{('--' if np.isnan(ang) else f'{ang:.3f}'):>10}"
                  f"{f'U3 min={min(nz)} (need>={n})':>21}   "
                  f"{'U3 PASS' if min(nz) >= n else 'U3 FAIL'}", flush=True)

    print("\n>>> CONFIRM requires: C1 stratified mu3/mu1 < %.0e every seed & angle; "
          "C2 ang(v3,xi_pred) < %.1f deg; C3 slope in [0.98,1.02] R^2>0.999; "
          "C4 placebo > %.0e; C5 urban > %.0e." % (TOL_NULL, TOL_ANG, TOL_FULL,
                                                   TOL_FULL), flush=True)
    print(">>> constant is unit test U2 only, never a data point; radial is reported "
          "only if its eps_leak gate passes.", flush=True)
    print("DONE diag_stabfisher", flush=True)


if __name__ == "__main__":
    main()
