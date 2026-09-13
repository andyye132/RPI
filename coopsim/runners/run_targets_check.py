"""B5 gate: the jitted fixed-shape GM-PHD against a plain numpy Vo-Ma reference.

Canned scenario (no comms, known world-frame detections): targets with births, deaths,
and a crossing; identical detection streams to both filters. Pass = GOSPA series of
the two filters agree within tolerance, cardinality tracks truth, and the label stream
yields sane IDF1 on the clean single-target case.
Run: python -u -m coopsim.runners.run_targets_check
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

from coopsim import targets
from coopsim.config import load_config
from coopsim.metrics.gospa import gospa
from coopsim.metrics.identity import identity_metrics

DT = 0.1


# ---- plain numpy Vo-Ma reference (dynamic lists, no shape tricks) ----
def np_reference(det_stream, cfg, sigma_a, survival):
    comps = []          # dicts w,m,P
    est_out = []
    F = np.eye(4); F[0, 2] = F[1, 3] = DT
    q = sigma_a ** 2
    Q = q * np.array([[DT**4/4, 0, DT**3/2, 0], [0, DT**4/4, 0, DT**3/2],
                      [DT**3/2, 0, DT**2, 0], [0, DT**3/2, 0, DT**2]])
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
    g = cfg.gmphd
    for z, R, p_d, kappa in det_stream:
        for c in comps:
            c["w"] *= survival ** DT
            c["m"] = F @ c["m"]
            c["P"] = F @ c["P"] @ F.T + Q
        new = [dict(w=c["w"] * (1 - p_d), m=c["m"], P=c["P"]) for c in comps]
        assoc = np.zeros(len(z))
        for d in range(len(z)):
            cands = []
            for c in comps:
                S = H @ c["P"] @ H.T + R[d]
                diff = z[d] - H @ c["m"]
                mah = diff @ np.linalg.solve(S, diff)
                qz = np.exp(-0.5 * mah) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
                K = c["P"] @ H.T @ np.linalg.inv(S)
                cands.append(dict(w=c["w"] * p_d * qz, m=c["m"] + K @ diff,
                                  P=(np.eye(4) - K @ H) @ c["P"]))
            tot = sum(x["w"] for x in cands)
            assoc[d] = tot / (kappa + tot) if tot > 0 else 0.0
            for x in cands:
                x["w"] /= (kappa + tot)
            new += cands
        for d in range(len(z)):                          # detection-driven births
            if assoc[d] < g.birth_weight:
                new.append(dict(w=g.birth_weight,
                                m=np.array([z[d][0], z[d][1], 0, 0]),
                                P=np.diag([0.25, 0.25, 1.5**2, 1.5**2])))
        new = [c for c in new if c["w"] > g.prune]
        merged = []
        new.sort(key=lambda c: -c["w"])
        used = [False] * len(new)
        for i, c in enumerate(new):
            if used[i]:
                continue
            grp = [j for j in range(len(new)) if not used[j] and
                   (new[j]["m"] - c["m"]) @ np.linalg.solve(
                       new[j]["P"] + 1e-6 * np.eye(4), new[j]["m"] - c["m"])
                   <= g.merge_mahal]
            ws = sum(new[j]["w"] for j in grp)
            mbar = sum(new[j]["w"] * new[j]["m"] for j in grp) / ws
            Pbar = sum(new[j]["w"] * (new[j]["P"] + np.outer(new[j]["m"] - mbar,
                                                             new[j]["m"] - mbar))
                       for j in grp) / ws
            for j in grp:
                used[j] = True
            merged.append(dict(w=ws, m=mbar, P=Pbar))
        comps = merged[:64]
        est_out.append(np.array([c["m"][:2] for c in comps if c["w"] > g.extract])
                       .reshape(-1, 2))
    return est_out


def make_scenario(rng, T=400, crossing=True):
    """Two targets crossing + one late birth + one death; returns truth + det stream."""
    tau = np.zeros((T, 3, 4)); alive = np.zeros((T, 3), bool)
    tau[0, 0] = [10, 25, 1.0, 0.0]; tau[0, 1] = [40, 25, -1.0, 0.0]
    alive[:, 0] = True; alive[: int(0.7 * T), 1] = True; alive[int(0.4 * T):, 2] = True
    tau[0, 2] = [25, 10, 0.0, 1.0]
    for t in range(1, T):
        tau[t, :, :2] = tau[t - 1, :, :2] + tau[t - 1, :, 2:] * DT
        tau[t, :, 2:] = tau[t - 1, :, 2:]
    dets = []
    for t in range(T):
        zs, Rs = [], []
        for k in range(3):
            if alive[t, k] and rng.random() < 0.9:
                zs.append(tau[t, k, :2] + rng.normal(0, 0.15, 2))
                Rs.append(np.eye(2) * 0.15 ** 2)
        for _ in range(rng.poisson(2.0)):
            zs.append(rng.uniform(0, 50, 2)); Rs.append(np.eye(2) * 0.15 ** 2)
        kappa = 2.0 / (50.0 * 50.0)
        dets.append((np.array(zs).reshape(-1, 2), np.array(Rs).reshape(-1, 2, 2),
                     0.9, kappa))
    return tau, alive, dets


def run_jax(det_stream, cfg, j_max=64, d_max=24):
    phd = targets.init(j_max, robot_id=0)
    g = cfg.gmphd
    step = jax.jit(lambda p, z, R, mask, pd: targets.step(
        p, z, R, mask, pd, 2.0 / 2500.0, g, DT, 0.3, 0.995, j_max))
    est_out, card = [], []
    for z, R, p_d, kappa in det_stream:
        D = len(z)
        zf = np.zeros((d_max, 2)); Rf = np.tile(np.eye(2), (d_max, 1, 1))
        mask = np.zeros(d_max, bool)
        zf[:D] = z; Rf[:D] = R; mask[:D] = True
        pd_vec = jnp.full(j_max, p_d)
        phd = step(phd, jnp.asarray(zf), jnp.asarray(Rf), jnp.asarray(mask), pd_vec)
        est, emask = targets.extract(phd, g.extract, 16)
        e = np.asarray(est)[np.asarray(emask)]
        est_out.append(e[:, :2].reshape(-1, 2))
        card.append(float(targets.cardinality(phd)))
    return est_out, card


def main():
    cfg = load_config()
    rng = np.random.default_rng(0)
    tau, alive, dets = make_scenario(rng)

    est_np = np_reference(dets, cfg, 0.3, 0.995)
    est_jx, card = run_jax(dets, cfg)

    g_np = [gospa(tau[t][alive[t]], est_np[t])["gospa"] for t in range(len(dets))]
    g_jx = [gospa(tau[t][alive[t]], est_jx[t])["gospa"] for t in range(len(dets))]
    g_np, g_jx = np.array(g_np), np.array(g_jx)
    settle = slice(30, None)
    diff = np.abs(g_np[settle] - g_jx[settle]).mean()
    scale = g_np[settle].mean()
    print(f"GOSPA mean: numpy={scale:.3f} jax={g_jx[settle].mean():.3f} "
          f"|diff|={diff:.3f}")
    ok1 = diff < 0.5 * scale + 0.3
    print(f"{'PASS' if ok1 else 'FAIL'} targets.jax_vs_numpy")

    n_alive = alive.sum(1)
    card_err = np.abs(np.array(card)[settle] - n_alive[settle]).mean()
    print(f"cardinality mean abs err = {card_err:.2f}")
    ok2 = card_err < 1.0
    print(f"{'PASS' if ok2 else 'FAIL'} targets.cardinality")

    # clean single target -> IDF1 1.0 on the label stream
    tau1 = np.zeros((300, 1, 4)); tau1[:, 0, :2] = [25, 25]; alive1 = np.ones((300, 1), bool)
    dets1 = [(np.array([[25, 25]]) + rng.normal(0, 0.1, (1, 2)),
              np.eye(2)[None] * 0.01, 0.95, 1e-6) for _ in range(300)]
    est1, _ = run_jax(dets1, cfg)
    est_stack = np.full((300, 16, 2), 1e6); est_lab = np.zeros((300, 16), int)
    est_mask = np.zeros((300, 16), bool)
    phd = targets.init(64, 0)
    g = cfg.gmphd
    step = jax.jit(lambda p, z, R, mask, pd: targets.step(
        p, z, R, mask, pd, 1e-6, g, DT, 0.3, 0.995, 64))
    for t, (z, R, p_d, kappa) in enumerate(dets1):
        zf = np.zeros((24, 2)); Rf = np.tile(np.eye(2), (24, 1, 1)); mask = np.zeros(24, bool)
        zf[:1] = z; Rf[:1] = R; mask[:1] = True
        phd = step(phd, jnp.asarray(zf), jnp.asarray(Rf), jnp.asarray(mask),
                   jnp.full(64, p_d))
        est, emask = targets.extract(phd, g.extract, 16)
        est_stack[t], est_lab[t] = np.asarray(est)[:, :2], np.asarray(est)[:, 4].astype(int)
        est_mask[t] = np.asarray(emask)
    idm = identity_metrics(tau1, alive1, est_stack, est_lab, est_mask)
    print(f"single-target idf1={idm['idf1']:.3f} idsw={idm['idsw']}")
    ok3 = idm["idf1"] > 0.9 and idm["idsw"] <= 1
    print(f"{'PASS' if ok3 else 'FAIL'} targets.labels_idf1")

    print("ALL PASS" if (ok1 and ok2 and ok3) else "FAILURES PRESENT")
    print("DONE targets_check")


if __name__ == "__main__":
    main()
