"""Fixed-data solver-budget sensitivity for Experiment B; use a compute node.

Reads original archived NPZ data/weights. Never overwrites original B results.
See results/acc2027_strengthening/solver_sensitivity/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse, stats
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.sparse.linalg import splu, spsolve
from coopsim.runners import run_interface_matched as base


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Objective:
    def __init__(self, data, scenario):
        self.data = data
        self.sc = base.SCENARIOS[scenario]
        self.f = data["map"]
        self.yc = data["rssi"]
        self.b, self.y, _ = base.linear_factors(data["initial_observation"], data["odometry"], data["relative_observations"])
        self.w = data[f"{scenario}_effective_weights"]
        self.a = (self.b.T @ self.b.multiply(self.w[:, None])).tocsc()
        self.eta = self.b.T @ (self.w*self.y)
        self.initial = data["shared_initialization"].copy()

    def value(self, x):
        r = self.b @ x-self.y
        rc = (base.radio(x, self.f, self.sc, False)-self.yc)/3.
        return float(.5*(np.dot(self.w*r, r)+np.dot(rc, rc)))

    def normal(self, x, method="message"):
        hc, gc, rc = base.channel_normal(x, self.f, self.sc, self.yc, method)
        r = self.b @ x-self.y
        value = float(.5*(np.dot(self.w*r, r)+np.dot(rc, rc)))
        return value, self.a+hc, np.asarray(self.a @ x-self.eta+gc)


def tight_lm(obj):
    x = obj.initial.copy()
    damping = 1e-3
    trace, states = [], {}
    reason, stop = "budget", False
    for iteration in range(1, 301):
        cost, h, grad = obj.normal(x)
        ginf = float(np.max(np.abs(grad)))
        step_inf, accepted, attempts = 0., False, 0
        if ginf <= 1e-6:
            reason, stop = "gradient", True
        else:
            for attempts in range(1, 10):
                step = spsolve(h+sparse.eye(base.D, format="csc")*damping, -grad)
                candidate = x+step
                trial = obj.value(candidate)
                step_inf = float(np.max(np.abs(step)))
                if np.isfinite(trial) and trial <= cost:
                    x = candidate
                    damping = max(damping/3., 1e-9)
                    accepted = True
                    if step_inf < 1e-12:
                        reason, stop = "step_stagnation", True
                    break
                damping *= 10.
            if not accepted:
                reason, stop = "rejected_stagnation", True
        trace.append({"iteration": iteration, "cost_before": cost,
                      "cost_after": obj.value(x), "gradient_before_inf": ginf,
                      "damping_after": damping, "step_inf": step_inf,
                      "accepted": accepted, "attempts": attempts})
        if iteration in (25, 100, 300):
            states[iteration] = (x.copy(), {"iterations": iteration, "stop_reason": reason if stop else "budget"})
        if stop:
            for budget in (25, 100, 300):
                if budget not in states:
                    states[budget] = (x.copy(), {"iterations": iteration, "stop_reason": reason})
            break
    return states, trace


def independent_lbfgs(obj):
    scale = 1/np.sqrt(np.maximum(obj.a.diagonal(), 1e-12))
    calls, trace = 0, []
    def fun(u):
        nonlocal calls
        calls += 1
        cost, _, grad = obj.normal(obj.initial+scale*u)
        return cost, scale*grad
    def callback(u):
        x = obj.initial+scale*u
        cost, _, grad = obj.normal(x)
        trace.append({"iteration": len(trace)+1, "cost": cost, "physical_gradient_inf": float(np.max(np.abs(grad)))})
    result = minimize(fun, np.zeros(base.D), method="L-BFGS-B", jac=True, callback=callback,
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 50, "maxfun": 20000})
    return obj.initial+scale*result.x, {"iterations": result.nit, "evaluations": calls,
        "scipy_success": bool(result.success), "message": str(result.message), "status": int(result.status)}, trace


def metrics(obj, x, baseline_x, baseline_cov):
    value, h, grad = obj.normal(x)
    valuej, hj, gj = obj.normal(x, "joint")
    hd = h-hj
    denom = max(np.linalg.norm(h.data), 1.)
    equality = {"relative_information_difference": float(np.linalg.norm(hd.data)/denom),
                "max_gradient_difference": float(np.max(np.abs(grad-gj))), "objective_difference": abs(value-valuej)}
    if max(equality.values()) > 1e-9:
        raise RuntimeError("Message/joint normal equations differ")
    rhs = np.zeros((base.D, 2*base.N)); rhs[-2*base.N:] = np.eye(2*base.N)
    cov = splu(h).solve(rhs)[-2*base.N:]
    cov = .5*(cov+cov.T)
    if not np.isfinite(x).all() or np.linalg.eigvalsh(cov).min() <= 0:
        raise RuntimeError("Invalid solver output")
    error = x.reshape(base.T, base.N, 2)-obj.data["truth"]
    common = error.mean(axis=1)
    relative = error-common[:, None]
    c = np.tile(np.eye(2)/base.N, (1, base.N))
    basis = null_space(c)
    pc, pr = c @ cov @ c.T, basis.T @ cov @ basis
    er = basis.T @ error[-1].ravel()
    nees = float(common[-1] @ np.linalg.solve(pc, common[-1]))
    nr = float(er @ np.linalg.solve(pr, er))
    bcost = obj.value(baseline_x)
    hstep = spsolve(h, -grad)
    result = {"objective": value, "objective_relative_improvement": (bcost-value)/max(bcost, 1.),
        "gradient_inf": float(np.max(np.abs(grad))), "undamped_gn_step_inf_m": float(np.max(np.abs(hstep))),
        "first_order_converged": bool(np.max(np.abs(grad)) <= 1e-6),
        "common_rmse_final_third": float(np.sqrt(np.mean(np.sum(common[2*base.T//3:]**2, axis=-1)))),
        "relative_rmse_final_third": float(np.sqrt(np.mean(np.sum(relative[2*base.T//3:]**2, axis=-1)))),
        "terminal_common_error": float(np.linalg.norm(common[-1])),
        "terminal_common_rms_uncertainty": float(np.sqrt(np.trace(pc))),
        "terminal_relative_rms_uncertainty": float(np.sqrt(np.trace(pr)/base.N)),
        "common_nees": nees, "relative_nees": nr,
        "common_coverage95": bool(nees <= stats.chi2.ppf(.95, 2)),
        "relative_coverage95": bool(nr <= stats.chi2.ppf(.95, 2*base.N-2)),
        "trajectory_rms_change_m": float(np.sqrt(np.mean((x-baseline_x)**2))),
        "max_coordinate_change_m": float(np.max(np.abs(x-baseline_x))),
        "relative_terminal_covariance_change": float(np.linalg.norm(cov-baseline_cov)/np.linalg.norm(baseline_cov)),
        "message_joint_equality": equality}
    return result, cov, grad


def probes(obj, x, grad, seed):
    directions = []
    for axis in (0, 1):
        d = np.zeros((base.T, base.N, 2)); d[..., axis] = 1.
        directions.append(d.ravel())
    directions.append(grad/max(np.max(np.abs(grad)), 1e-15))
    d = np.zeros(base.D); d[np.argmax(np.abs(grad))] = 1.
    directions.append(d)
    rng = np.random.default_rng(seed+100000)
    for _ in range(8):
        d = rng.standard_normal(base.D); directions.append(d/np.max(np.abs(d)))
    value = obj.value(x)
    radii = np.logspace(-2, -7, 6)
    changes = np.array([[[obj.value(x+sign*r*d)-value for sign in (-1., 1.)]
                          for d in directions] for r in radii])
    return {"radii_m": radii.tolist(), "directions": ["translation_x", "translation_y", "gradient", "largest_gradient_coordinate"]+[f"random_{i}" for i in range(8)],
            "objective_changes": changes.tolist(), "min_objective_change_by_radius": changes.min(axis=(1,2)).tolist(),
            "smallest_objective_change": float(changes.min()),
            "note": "Finite directional probes diagnose local descent only; not a local/global optimality certificate."}


def run(seed, source, output):
    start_time = time.monotonic()
    with np.load(source/f"seed_{seed}.npz") as z:
        data = {k: z[k] for k in z.files}
    original = json.loads((source/f"seed_{seed}.json").read_text())
    for key, checksum in original["input_hashes"].items():
        assert sha(data[key]) == checksum, key
    result = {"seed": seed, "input_hashes": original["input_hashes"], "scenarios": {}}
    arrays = {}
    for scenario in base.SCENARIOS:
        obj = Objective(data, scenario)
        oldx = data[f"{scenario}_message_estimate"].ravel()
        oldcov = data[f"{scenario}_message_terminal_covariance"]
        wh = sha(obj.w)
        assert all(v["effective_weights_hash"] == wh for v in original["scenarios"][scenario]["methods"].values())
        replayx, replaycov, _ = base.solve_arm(obj.b, obj.y, obj.w, obj.initial, obj.f, obj.sc, obj.yc, "message")
        replay = {"max_pose_difference": float(np.max(np.abs(replayx.ravel()-oldx))),
                  "max_covariance_difference": float(np.max(np.abs(replaycov-oldcov)))}
        if max(replay.values()) > 1e-8:
            raise RuntimeError(f"Original replay mismatch {seed} {scenario}: {replay}")
        states, lm_trace = tight_lm(obj)
        lbfgs_x, lbfgs_meta, lbfgs_trace = independent_lbfgs(obj)
        variants = {"original25": (oldx, original["scenarios"][scenario]["methods"]["message"]["optimizer"])}
        variants.update({f"tight{budget}": state for budget, state in states.items()})
        variants["lbfgs1000"] = (lbfgs_x, lbfgs_meta)
        rows, probe_results = {}, {}
        for name, (x, meta) in variants.items():
            row, cov, grad = metrics(obj, x, oldx, oldcov)
            row["optimizer"] = meta
            rows[name] = row
            arrays[f"{scenario}_{name}_estimate"] = x.reshape(base.T, base.N, 2)
            arrays[f"{scenario}_{name}_terminal_covariance"] = cov
            if name in ("original25", "tight300", "lbfgs1000"):
                probe_results[name] = probes(obj, x, grad, seed)
        result["scenarios"][scenario] = {"methods": rows, "original_replay": replay,
            "effective_weights_hash": wh, "scalar": original["scenarios"][scenario]["methods"]["scalar"],
            "tight_lm_trace": lm_trace, "lbfgs_trace": lbfgs_trace, "directional_probes": probe_results}
        print(f"seed={seed} scenario={scenario} completed in {time.monotonic()-start_time:.1f}s", flush=True)
    result["elapsed_seconds"] = time.monotonic()-start_time
    np.savez_compressed(output/f"seed_{seed}.npz", **arrays)
    (output/f"seed_{seed}.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--source", type=Path, default=Path("results/acc2027_revision/experiment_b"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    run(a.seed, a.source, a.output)


if __name__ == "__main__":
    main()
