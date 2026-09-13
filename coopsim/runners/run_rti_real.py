"""Measured-radio model check on the surveyed ten-node Utah SPAN basement data.

A. Emission model: fit RSS = P0 - 10 gamma log10(d) on the first basement walk's
   per-link median baseline, using 45 undirected links at surveyed distances;
   report gamma and the shadowing residual sigma,
   the real-radio counterpart of the simulator's path-loss + noise assumptions.
B. Real-data tomography: reconstruct the attenuation field from the basement
   person-walk RSS using AETHER's own Fresnel line-integral forward model
   (coopsim.field.line_integral) and Huber-TV field prior, with node positions
   fixed to survey. Report the distance of each reconstructed attenuation peak
   to the entire known walked polyline. This is not a synchronized person-position
   error: a peak anywhere on the walked path receives zero error.
Run: python -u -m coopsim.runners.run_rti_real
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import rti_data
from coopsim.config import load_config
from coopsim.field import line_integral, tv_residuals, huber_weights

ROOT = rti_data.ROOT
from coopsim.paths import RUNS
OUT = RUNS / "rti_real.npz"


def emission_fit(coords, baseline):
    """Fit RSS = P0 - 10 gamma log10(d) on the per-link clear-channel baseline of a
    verified-geometry dataset. baseline (n,n) is the median RSS per undirected link."""
    n = coords.shape[0]
    d, y = [], []
    for a in range(n):
        for b in range(a + 1, n):
            if np.isfinite(baseline[a, b]):
                d.append(np.linalg.norm(coords[a] - coords[b]))
                y.append(baseline[a, b])
    d, y = np.array(d), np.array(y)
    x = 10.0 * np.log10(d)
    A = np.stack([np.ones_like(x), -x], 1)
    (p0, gamma), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (p0 - gamma * x)
    return dict(p0=float(p0), gamma=float(gamma), sigma=float(resid.std()),
                n_links=len(d), r2=float(1 - resid.var() / y.var()),
                link_d=d, link_rss=y)


def build_W(coords, extent, res):
    """Fresnel line-integral weight matrix: W[link, voxel], linear in F."""
    n = coords.shape[0]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    f = lambda F, xi, xj: line_integral(xi, xj, F, extent)
    grad = jax.jit(jax.grad(f, argnums=0))
    W = []
    for a, b in pairs:
        g = grad(jnp.zeros((res, res)), jnp.asarray(coords[a]), jnp.asarray(coords[b]))
        W.append(np.asarray(g).reshape(-1))
    return np.array(W), pairs


def solve_field(W, y, res, lam_tv=1.0, lam0=0.05, delta=3.0, iters=8):
    """min ||W f - y||^2 + lam0||f||^2 + lam_tv HuberTV(f), f>=0 (IRLS)."""
    G2 = W.shape[1]
    WtW = W.T @ W
    Wty = W.T @ y
    k1, k2 = [np.asarray(x) for x in _tv_idx(res)]
    f = np.zeros(G2)
    for _ in range(iters):
        e = f[k2] - f[k1]
        w = lam_tv * np.asarray(huber_weights(jnp.asarray(e), delta))
        L = np.zeros((G2, G2))
        np.add.at(L, (k1, k1), w); np.add.at(L, (k2, k2), w)
        np.add.at(L, (k1, k2), -w); np.add.at(L, (k2, k1), -w)
        A = WtW + lam0 * np.eye(G2) + L
        f = np.linalg.solve(A, Wty)
        f = np.clip(f, 0, None)
    return f.reshape(res, res)


def _tv_idx(G):
    k = np.arange(G * G).reshape(G, G)
    v1, v2 = k[:-1, :].reshape(-1), k[1:, :].reshape(-1)
    h1, h2 = k[:, :-1].reshape(-1), k[:, 1:].reshape(-1)
    return np.concatenate([v1, h1]), np.concatenate([v2, h2])


def tomography(run=1):
    """Reconstruct peaks and measure their distance to the entire walked path."""
    base = ROOT / "basement"
    coords, rss, ts = rti_data.load_rss(base / f"basement_listenx_out_{run}.txt",
                                        base / "sensor_coords_basement_m.txt")
    path = rti_data.load_path(base / f"path_basement_{run}_f.txt",
                              base / "pivot_coords_basement_m.txt")
    n, T = coords.shape[0], rss.shape[0]
    extent = float(np.ceil(coords.max() + 0.5))
    res = int(round(extent / 0.25))
    W, pairs = build_W(coords, extent, res)
    baseline = np.nanmedian(rss, axis=0)                 # per-link clear RSS

    grid = np.stack(np.meshgrid(np.linspace(0, extent, res),
                                np.linspace(0, extent, res)), -1).reshape(-1, 2)
    # sample frames across the run (person moving); reconstruct each
    frames = np.linspace(int(0.05 * T), T - 1, 120).astype(int)
    errs, snaps, kept_frames, peaks = [], [], [], []
    ffield = np.zeros(res * res)
    for t in frames:
        y = []
        for (a, b) in pairs:
            atten = baseline[a, b] - rss[t, a, b]        # person blocks -> positive
            y.append(atten if np.isfinite(atten) else 0.0)
        y = np.clip(np.array(y), 0, None)
        if y.max() < 1.0:                                # no clear blockage this frame
            continue
        F = solve_field(W, y, res)
        ffield += F.reshape(-1)
        peak = grid[np.argmax(F.reshape(-1))]
        errs.append(rti_data.point_to_polyline(peak, path))
        kept_frames.append(t)
        peaks.append(peak)
        if len(snaps) < 3:
            snaps.append(F)
    errs = np.array(errs)
    kept_frames = np.asarray(kept_frames, dtype=int)
    return dict(coords=coords, path=path, extent=extent, res=res, baseline=baseline,
                err_median=float(np.median(errs)), err_mean=float(errs.mean()),
                errors=errs, frame_indices=kept_frames, timestamps_ms=ts[kept_frames],
                peaks=np.asarray(peaks), n_nodes=n,
                n_frames=len(errs), avg_field=(ffield / max(len(errs), 1)).reshape(res, res),
                snaps=np.array(snaps) if snaps else np.zeros((0, res, res)))


def main():
    print("=== B. real-data tomography (basement person-walk) ===", flush=True)
    print("  metric: attenuation-peak distance to the entire walked polyline; "
          "not synchronized person-position error", flush=True)
    walks = {}
    for r in (1, 2):
        t = walks[r] = tomography(r)
        print(f"  path {r}: peak-to-path median {t['err_median']:.2f} m, "
              f"mean {t['err_mean']:.2f} m over {t['n_frames']} frames", flush=True)
    tomo = walks[1]
    print(f"  surveyed nodes: {tomo['n_nodes']}", flush=True)
    print(f"  arena {tomo['extent']:.1f} m, grid {tomo['res']}x{tomo['res']} "
          f"@ {tomo['extent']/tomo['res']:.2f} m", flush=True)

    print("\n=== A. emission model fit (basement baseline, verified geometry) ===",
          flush=True)
    em = emission_fit(tomo["coords"], tomo["baseline"])
    print(f"  gamma = {em['gamma']:.2f}   P0 = {em['p0']:.1f} dBm   "
          f"shadowing sigma = {em['sigma']:.2f} dB   R2 = {em['r2']:.2f}   "
          f"({em['n_links']} links)", flush=True)
    ch = load_config().channel
    print(f"  simulation reference: gamma={ch.gamma:g}, "
          f"sigma_f={ch.sigma_f_db:g} dB")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    walk_data = {
        f"walk{r}_{key}": t[key]
        for r, t in walks.items()
        for key in ("path", "errors", "err_median", "err_mean", "n_frames",
                    "frame_indices", "timestamps_ms", "peaks")
    }
    np.savez_compressed(OUT, gamma=em["gamma"], p0=em["p0"], sigma=em["sigma"],
                        metric="peak_to_walked_polyline_distance_m",
                        n_nodes=tomo["n_nodes"], n_links=em["n_links"],
                        walk_ids=np.array([1, 2]), data_root=str(ROOT),
                        r2=em["r2"], err_median=tomo["err_median"],
                        err_mean=tomo["err_mean"], coords=tomo["coords"],
                        path=tomo["path"], avg_field=tomo["avg_field"],
                        snaps=tomo["snaps"], extent=tomo["extent"],
                        link_d=em["link_d"], link_rss=em["link_rss"], **walk_data)
    print(f"\nsaved {OUT}")
    print("DONE rti_real")


if __name__ == "__main__":
    main()
