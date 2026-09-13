"""Experiment B: matched supplied-map interface ablation on moving trajectories.

This is a controlled nonlinear RSSI localization experiment, not the production
packet/blockage simulator. It uses its exact 64-sample, bilinear field emission.
Known headings, all six undirected links observed, 80 slots, four robots. A common
relative-only fit sets the initialization and freezes robust relative weights.
The scalar quality multiplier is also used by both richer arms, making their
effective relative quadratic identical. The known map is never optimized.
Run only on an allocated compute node. Outputs include every trial and covariance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse, stats
from scipy.linalg import null_space
from scipy.sparse.linalg import splu, spsolve

from coopsim import field, world
from coopsim.config import load_config, crn_keys

N, T, DT, EXT = 4, 80, .5, 50.
D = 2 * N * T
II, JJ = np.triu_indices(N, 1)
L = len(II)
SCENARIOS = {
    "correct": {"shift_m": [0., 0.], "scale": 1.},
    "registration": {"shift_m": [1., 0.], "scale": 1.},
    "attenuation": {"shift_m": [0., 0.], "scale": 1.2},
    "combined": {"shift_m": [1., 0.], "scale": 1.2},
}
METHODS = ("scalar", "message", "joint")


def emission_pair(a, b, f, shift, scale):
    # F_supplied(x) = scale F_true(x-shift), evaluated without a second resampling.
    return field.emission(a - shift, b - shift, f * scale, EXT, P0=-40., gamma=2.2)


emit_grad = jax.jit(jax.vmap(jax.value_and_grad(emission_pair, argnums=(0, 1)),
                           in_axes=(0, 0, None, None, None)))
emit_value = jax.jit(jax.vmap(emission_pair, in_axes=(0, 0, None, None, None)))


def radio(x, f, scenario, derivative=True):
    p = x.reshape(T, N, 2)
    a, b = p[:, II].reshape(-1, 2), p[:, JJ].reshape(-1, 2)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(f),
            jnp.asarray(scenario["shift_m"]), scenario["scale"])
    if not derivative:
        return np.asarray(emit_value(*args))
    val, (gi, gj) = emit_grad(*args)
    gi, gj = np.asarray(gi), np.asarray(gj)
    j = np.zeros((T, L, 2 * N))
    for k, (i, q) in enumerate(zip(II, JJ)):
        j[:, k, 2*i:2*i+2] = gi.reshape(T, L, 2)[:, k]
        j[:, k, 2*q:2*q+2] = gj.reshape(T, L, 2)[:, k]
    return np.asarray(val), j


def linear_factors(y0, odom, rel):
    rows, cols, values, target = [], [], [], []
    def factor(indices, coeffs, y, sig):
        r = len(target)
        rows.extend([r] * len(indices)); cols.extend(indices)
        values.extend([v/sig for v in coeffs]); target.append(y/sig)
    for i in range(2*N):
        factor([i], [1.], y0.ravel()[i], .5)
    for t in range(1, T):
        for i in range(2*N):
            factor([t*2*N+i, (t-1)*2*N+i], [1., -1.], odom[t-1].ravel()[i], .08)
    start = len(target)
    for t in range(T):
        for k, (i, j) in enumerate(zip(II, JJ)):
            for c in range(2):
                factor([t*2*N+2*j+c, t*2*N+2*i+c], [1., -1.], rel[t, k, c], .1)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(target), D)), np.array(target), start


def common_reference(b, y, start):
    weights = np.ones(len(y))
    for _ in range(10):
        a = b.T @ b.multiply(weights[:, None])
        x = spsolve(a.tocsc(), b.T @ (weights*y))
        r = (b @ x-y)[start:].reshape(T, L, 2)
        weights[start:] = np.repeat(np.minimum(1., 2.5 / np.maximum(np.linalg.norm(r, axis=-1), 1e-12)), 2)
    # Final reference and weights are shared by all arms, including the scalar arm.
    x = spsolve((b.T @ b.multiply(weights[:, None])).tocsc(), b.T @ (weights*y))
    return x, weights


def channel_normal(x, f, sc, obs, method):
    pred, j = radio(x, f, sc)
    r = (pred-obs)/3.
    j /= 3.
    if method == "joint":
        # Direct stacked observation likelihood.
        full = sparse.block_diag(list(j), format="csr")
        return (full.T @ full).tocsc(), np.asarray(full.T @ r), r
    # Independently construct one pose information matrix/vector per actual slot.
    # The map is fixed, so there is no uncertain-field temporal coupling to discard.
    h = np.einsum("tli,tlj->tij", j, j)
    g = np.einsum("tli,tl->ti", j, r.reshape(T, L))
    return sparse.block_diag(list(h), format="csc"), g.ravel(), r


def solve_arm(b, y, weights, initial, f, sc, obs, method):
    a0 = (b.T @ b.multiply(weights[:, None])).tocsc()
    eta = b.T @ (weights*y)
    if method == "scalar":
        x = spsolve(a0, eta)
        info, iterations, success = a0, 1, True
        grad = a0 @ x-eta
    else:
        x = initial.copy()
        damping = 1e-3
        success = False
        for iterations in range(1, 26):
            hc, gc, r = channel_normal(x, f, sc, obs, method)
            info, grad = a0+hc, a0 @ x-eta+gc
            residual = b @ x-y
            cost = .5*(np.dot(weights*residual, residual)+np.dot(r, r))
            if np.max(np.abs(grad)) < 1e-5:
                success = True
                break
            accepted = False
            for _ in range(9):
                step = spsolve(info+sparse.eye(D, format="csc")*damping, -grad)
                candidate = x+step
                rc = (radio(candidate, f, sc, False)-obs)/3.
                rr = b @ candidate-y
                trial = .5*(np.dot(weights*rr, rr)+np.dot(rc, rc))
                if np.isfinite(trial) and trial <= cost:
                    x = candidate
                    damping = max(damping/3., 1e-9)
                    accepted = True
                    if np.max(np.abs(step)) < 1e-5:
                        success = True
                    break
                damping *= 10.
            if success or not accepted:
                break
        hc, gc, _ = channel_normal(x, f, sc, obs, method)
        info, grad = a0+hc, a0 @ x-eta+gc
    # Undamped local Gauss-Newton covariance, conditional on frozen IRLS weights.
    rhs = np.zeros((D, 2*N)); rhs[-2*N:] = np.eye(2*N)
    cov = splu(info).solve(rhs)[-2*N:]
    return x.reshape(T, N, 2), .5*(cov+cov.T), {
        "iterations": iterations, "converged": bool(np.max(np.abs(grad)) < 1e-5),
        "small_step_or_gradient_stop": bool(success),
        "gradient_inf": float(np.max(np.abs(grad))),
    }


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def trial(seed, out):
    rng = np.random.default_rng(seed)
    cfg = load_config(overrides=[f"shapes.n_robots={N}", f"time.steps={T}", f"time.dt={DT}"])
    keys = crn_keys("interface_matched", "B", seed)
    f = np.array(world.make_world(keys.world_gen, cfg).F_true, dtype=float)
    truth = np.array(world.make_trajectories(keys.target_gen, cfg)[0][..., :2], dtype=float)
    y0 = truth[0]+rng.normal(0., .5, (N, 2))
    odom = np.diff(truth, axis=0)+rng.normal(0., .08, (T-1, N, 2))
    rel = truth[:, JJ]-truth[:, II]
    contaminated = rng.random((T, L)) < .1
    rel += rng.normal(0., .1, (T, L, 2))*np.where(contaminated[..., None], 8., 1.)
    rssi = radio(truth.ravel(), f, SCENARIOS["correct"], False)+rng.normal(0., 3., T*L)
    b, y, start = linear_factors(y0, odom, rel)
    reference, robust = common_reference(b, y, start)
    archive = {"truth": truth, "map": f, "initial_observation": y0, "odometry": odom,
               "relative_observations": rel, "rssi": rssi, "shared_initialization": reference,
               "robust_weights": robust, "outlier_mask": contaminated}
    output = {"seed": seed, "input_hashes": {k: sha(v) for k, v in archive.items()}, "scenarios": {}}
    cmean = np.tile(np.eye(2)/N, (1, N))
    relbasis = null_space(cmean)
    for scenario, sc in SCENARIOS.items():
        quality = .5+.5*stats.norm.cdf((radio(reference, f, sc, False)+88.)/3.)
        weights = robust.copy()
        weights[start:] *= np.repeat(quality, 2)
        archive[f"{scenario}_effective_weights"] = weights
        rows = {}
        for method in METHODS:
            x, cov, opt = solve_arm(b, y, weights, reference, f, sc, rssi, method)
            error = x-truth
            common = error.mean(axis=1)
            relative = error-common[:, None]
            pc = cmean @ cov @ cmean.T
            er = relbasis.T @ error[-1].ravel()
            pr = relbasis.T @ cov @ relbasis
            nees = float(common[-1] @ np.linalg.solve(pc, common[-1]))
            nees_rel = float(er @ np.linalg.solve(pr, er))
            rows[method] = {
                "common_rmse_final_third": float(np.sqrt(np.mean(np.sum(common[2*T//3:]**2, axis=-1)))),
                "relative_rmse_final_third": float(np.sqrt(np.mean(np.sum(relative[2*T//3:]**2, axis=-1)))),
                "terminal_common_error": float(np.linalg.norm(common[-1])),
                "terminal_common_rms_uncertainty": float(np.sqrt(np.trace(pc))),
                "terminal_relative_rms_uncertainty": float(np.sqrt(np.trace(pr)/N)),
                "common_nees": nees, "relative_nees": nees_rel,
                "common_coverage95": bool(nees <= stats.chi2.ppf(.95, 2)),
                "relative_coverage95": bool(nees_rel <= stats.chi2.ppf(.95, 2*N-2)),
                "effective_weights_hash": sha(weights), "optimizer": opt,
            }
            archive[f"{scenario}_{method}_estimate"] = x
            archive[f"{scenario}_{method}_terminal_covariance"] = cov
        xm, xj = archive[f"{scenario}_message_estimate"], archive[f"{scenario}_joint_estimate"]
        cm, cj = archive[f"{scenario}_message_terminal_covariance"], archive[f"{scenario}_joint_terminal_covariance"]
        output["scenarios"][scenario] = {"methods": rows,
            "message_joint_max_pose_difference": float(np.max(np.abs(xm-xj))),
            "message_joint_relative_covariance_difference": float(np.linalg.norm(cm-cj)/np.linalg.norm(cj))}
    np.savez_compressed(out/f"seed_{seed}.npz", **archive)
    (out/f"seed_{seed}.json").write_text(json.dumps(output, indent=2)+"\n")
    return output


def protocol():
    return {"experiment": "B", "seeds": list(range(200, 232)), "robots": N, "slots": T, "dt": DT,
        "scenarios": SCENARIOS, "methods": METHODS, "map": "40x40 generated urban; fixed and supplied to all arms",
        "measurement": "all six undirected RSSI and relative-position links every slot; no censoring, scheduling, or latent blockage",
        "noise": {"initial_m": .5, "odom_m": .08, "relative_m": .1, "relative_outlier_probability": .1,
                  "outlier_scale": 8., "rssi_db": 3.},
        "robust": "10 common relative-only vector Huber IRLS steps, threshold 2.5; freeze weights; q=.5+.5*Phi((predicted_RSSI+88)/3); use identical q*Huber weights in every arm",
        "solver": "full retained 640-dimensional known-heading trajectory; common initialization; <=25 LM iterations; terminal undamped GN covariance",
        "scope": "finite-horizon nonlinear controlled RSSI experiment; covariance is local conditional IRLS approximation; no exact nonlinear calibration or asymptotic boundedness claim",
        "predeclared_metrics": ["final-third common and relative RMSE", "terminal common and relative NEES", "95% coverage", "terminal RMS uncertainty", "message-joint mean/covariance agreement"],
        "inference": "32 paired seeds; mean/seed SD and Student-t 95% CI; chi-square aggregate NEES intervals descriptive only for data-dependent robust Hessians"}


def aggregate(out):
    records = [json.loads((out/f"seed_{s}.json").read_text()) for s in range(200, 232)]
    result = {"protocol": protocol(), "scenarios": {}}
    for scenario in SCENARIOS:
        rr = [r["scenarios"][scenario] for r in records]
        cell = {"methods": {}, "paired_message_minus_scalar": {}}
        for method in METHODS:
            rows = [r["methods"][method] for r in rr]
            metrics = {}
            for key in rows[0]:
                if key in ("effective_weights_hash", "optimizer"):
                    continue
                z = np.array([r[key] for r in rows], dtype=float)
                se = z.std(ddof=1)/np.sqrt(len(z))
                if key.endswith("coverage95"):
                    interval = stats.binomtest(int(z.sum()), len(z)).proportion_ci(method="wilson")
                    ci = [float(interval.low), float(interval.high)]
                else:
                    ci = [float(z.mean()-stats.t.ppf(.975, 31)*se), float(z.mean()+stats.t.ppf(.975, 31)*se)]
                metrics[key] = {"mean": float(z.mean()), "sd": float(z.std(ddof=1)), "ci95": ci}
            # The original execution's `converged` also included a small damped
            # step. Use actual final gradients so no stalled fit is mislabeled.
            metrics["first_order_converged_count"] = sum(r["optimizer"]["gradient_inf"] < 1e-5 for r in rows)
            metrics["median_gradient_inf"] = float(np.median([r["optimizer"]["gradient_inf"] for r in rows]))
            metrics["max_gradient_inf"] = max(r["optimizer"]["gradient_inf"] for r in rows)
            cell["methods"][method] = metrics
        for key in ("common_rmse_final_third", "relative_rmse_final_third"):
            z = np.array([r["methods"]["message"][key]-r["methods"]["scalar"][key] for r in rr])
            se = z.std(ddof=1)/np.sqrt(len(z))
            cell["paired_message_minus_scalar"][key] = {"mean": float(z.mean()), "sd": float(z.std(ddof=1)),
                "ci95": [float(z.mean()-stats.t.ppf(.975,31)*se), float(z.mean()+stats.t.ppf(.975,31)*se)]}
        cell["message_joint_max_pose_difference"] = max(r["message_joint_max_pose_difference"] for r in rr)
        cell["message_joint_max_relative_covariance_difference"] = max(r["message_joint_relative_covariance_difference"] for r in rr)
        if any(len({v["effective_weights_hash"] for v in r["methods"].values()}) != 1 for r in rr):
            raise RuntimeError("Relative weights differ across arms")
        result["scenarios"][scenario] = cell
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    labels = ["Correct map", "+1 m shift", "+20% attenuation", "Both errors"]
    colors = ["#6b7280", "#0072b2", "#d55e00"]
    keys = ["common_rmse_final_third", "relative_rmse_final_third", "terminal_common_rms_uncertainty"]
    titles = ["Common-mode error (m)", "Relative error (m)", "Common uncertainty (m)"]
    for ax, key, title in zip(axes, keys, titles):
        for mi, (method, color) in enumerate(zip(METHODS, colors)):
            vals = [result["scenarios"][s]["methods"][method][key] for s in SCENARIOS]
            means = np.array([v["mean"] for v in vals])
            errs = np.array([v["ci95"][1]-v["mean"] for v in vals])
            ax.errorbar(np.arange(4)+(mi-1)*.16, means, yerr=errs, fmt=["s", "o", "x"][mi],
                        color=color, capsize=2, label=method.capitalize())
        ax.set_xticks(range(4), labels, rotation=22, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out/f"interface_errors.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for mi, (method, color) in enumerate(zip(METHODS, colors)):
        vals = [result["scenarios"][s]["methods"][method] for s in SCENARIOS]
        axes[0].plot(range(4), [v["common_nees"]["mean"]/2 for v in vals], ["s-", "o-", "x--"][mi], color=color, label=method.capitalize())
        axes[1].plot(range(4), [100*v["common_coverage95"]["mean"] for v in vals], ["s-", "o-", "x--"][mi], color=color)
    axes[0].axhline(1, color="black", ls=":", lw=1)
    axes[1].axhline(95, color="black", ls=":", lw=1)
    axes[0].set_title("Terminal common NEES / 2"); axes[0].set_yscale("log")
    axes[1].set_title("Common 95% coverage (%)")
    for ax in axes:
        ax.set_xticks(range(4), labels, rotation=22, ha="right"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out/f"interface_consistency.{ext}", dpi=180)
    plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--write-protocol", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.write_protocol:
        (args.output/"protocol.json").write_text(json.dumps(protocol(), indent=2)+"\n")
    elif args.aggregate:
        aggregate(args.output)
    elif args.seed is not None:
        print(json.dumps(trial(args.seed, args.output)), flush=True)
    else:
        p.error("choose --seed, --aggregate, or --write-protocol")


if __name__ == "__main__":
    main()
