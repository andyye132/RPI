"""Field-frame common-mode diagnostic: does the estimated field co-drift with the
team (H1, map co-drift) or stay world-anchored while poses drift anyway (H2/H3)?

For each m1 save-field npz (field_snaps K x G x G, F_true, snap_t, err_mean K x 2),
estimate the translation d_F of the estimated field pattern vs truth by FFT
cross-correlation with subpixel peak interpolation, then compare against the team's
mean pose error. Sign convention: d_F = +d when F_hat(x) ~ F_true(x - d), i.e. the
pattern is displaced by +d; under co-drift the walls are mapped at (true + pose
error), so H1 predicts d_F ~ err_mean and cm_minus = |err_mean - d_F| small.
Run (CPU): python -u coopsim/runners/diag_frame.py
"""
import glob
import os

import numpy as np

from coopsim.paths import RUNS as _RUNS


def field_shift(F_hat, F_true, voxel_m):
    """Translation (meters, 2-vector) maximizing circular cross-correlation."""
    a = F_hat - F_hat.mean()
    b = F_true - F_true.mean()
    C = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    G = F_true.shape[0]
    k = np.unravel_index(np.argmax(C), C.shape)
    d = np.zeros(2)
    for ax in (0, 1):
        km, k0, kp = (k[ax] - 1) % G, k[ax], (k[ax] + 1) % G
        idx = [list(k), list(k), list(k)]
        idx[0][ax], idx[2][ax] = km, kp
        cm_, c0, cp = C[tuple(idx[0])], C[tuple(idx[1])], C[tuple(idx[2])]
        denom = cm_ - 2 * c0 + cp
        sub = 0.5 * (cm_ - cp) / denom if abs(denom) > 1e-12 else 0.0
        s = k0 + sub
        if s > G / 2:
            s -= G
        d[ax] = s
    # array axis 0 = row = y-ish grid index, axis 1 = col; field is (G, G) over the
    # arena with the same row/col convention as err_mean's (x, y) needs checking:
    # world.py rasterizes F[ix, iy] with ix from x, so axis 0 = x, axis 1 = y.
    return d * voxel_m


def selftest():
    G = 40
    rng = np.random.default_rng(0)
    F = np.zeros((G, G))
    F[10:14, 20:26] = 10.0
    F += 0.1 * rng.standard_normal((G, G))
    shifted = np.roll(F, (2, 1), axis=(0, 1))   # pattern moved +2 voxels ax0, +1 ax1
    d = field_shift(shifted, F, 1.0)
    assert np.allclose(d, [2, 1], atol=0.15), f"selftest shift {d} != [2, 1]"
    print("SELFTEST OK (recovered +[2, 1] voxel shift)", flush=True)


def main():
    selftest()
    pat = os.environ.get(
        "DIAG_GLOB", str(_RUNS / "m1_diag_s*.npz"))
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no files match {pat}")
        return
    for p in files:
        d = np.load(p)
        if "err_mean" not in d:
            print(f"SKIP {p}: no err_mean (old-format snapshot file)")
            continue
        snaps, F_true = d["field_snaps"], d["F_true"]
        snap_t, err_mean = d["snap_t"], d["err_mean"]
        G = F_true.shape[0]
        voxel_m = 50.0 / G
        print(f"\n== {os.path.basename(p)} ({len(snap_t)} snaps, G={G}) ==")
        print(f"{'t':>6s} {'cm_world':>9s} {'|d_F|':>7s} {'cm_minus':>9s} "
              f"{'cm_plus':>8s}")
        rows = []
        for k in range(len(snap_t)):
            dF = field_shift(snaps[k], F_true, voxel_m)
            e = err_mean[k]
            row = (float(np.linalg.norm(e)), float(np.linalg.norm(dF)),
                   float(np.linalg.norm(e - dF)), float(np.linalg.norm(e + dF)))
            rows.append(row)
            print(f"{int(snap_t[k]):6d} {row[0]:9.3f} {row[1]:7.3f} "
                  f"{row[2]:9.3f} {row[3]:8.3f}")
        last = np.array(rows[len(rows) * 2 // 3:])
        print(f"last third: cm_world={last[:, 0].mean():.3f} "
              f"|d_F|={last[:, 1].mean():.3f} "
              f"cm_minus={last[:, 2].mean():.3f} cm_plus={last[:, 3].mean():.3f}")
        print("verdict hint: cm_minus << cm_world -> H1 co-drift (bounded in map "
              "frame); |d_F| ~ 0 with cm_world large -> H2/H3 weak anchoring")
    print("DONE diag_frame")


if __name__ == "__main__":
    main()
