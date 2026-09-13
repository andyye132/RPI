"""Experiment A: moving geometry, uncertain shared field, Gaussian messages.

This is a calibrated *fixed-linearization* experiment, not a claim about a
nonlinear optimizer. All retained trajectory positions are distinct variables.
The radio Jacobians come from physical inter-robot line integrals through a
spatial attenuation basis. Run on a Slurm compute node, never a login node.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import chi2, t as student_t


METHODS = ("joint", "corrected", "naive_slotwise")
LABELS = ("Joint", "Corrected trajectory", "Naive slotwise")
COLORS = ("#1f77b4", "#2ca02c", "#d95f02")


def sym(a):
    return 0.5 * (a + a.T)


def inv(a):
    return sym(np.linalg.solve(a, np.eye(len(a))))


def ci(values):
    values = np.asarray(values)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1)
                 * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean - half, mean + half],
            "sd": float(values.std(ddof=1))}


def default_config():
    return {
        "experiment": "A: shared-map message construction",
        "scope": "Linear Gaussian model frozen at a moving physical geometry; not nonlinear A-HY equivalence",
        "seed": 202609061, "replicates": 1000, "robots": 3, "steps": 12,
        "dt_s": 1.0, "known_heading": True, "field_basis": "3x3 Gaussian RBF attenuation per metre",
        "field_centers_axis_m": [0.0, 3.5, 7.0], "field_width_m": 1.7,
        "field_mean_db_per_m": [0.25, 0.85, 0.35, 1.0, 0.4, 0.75, 0.3, 0.95, 0.2],
        "field_std_db_per_m": 0.25, "field_correlation_length_m": 3.0,
        "radio_sigma_db": 0.12, "pathloss_gamma": 2.2,
        "relative_position_sigma_m": 0.06,
        "initial_common_position_sigma_m": 0.65,
        "initial_individual_position_sigma_m": 0.10,
        "process_individual_sigma_m": 0.045,
        "quadrature_nodes": 32, "finite_difference_m": 1e-5,
        "retained_window_steps": 4,
        "ci_policy": "Student t interval across independent replicates; NEES null interval from chi-square with N*d degrees of freedom",
        "primary_endpoints": ["posterior mean difference from joint", "relative Frobenius covariance difference from joint", "full trajectory normalized NEES", "terminal common-mode normalized NEES", "coordinatewise 95% coverage"],
        "negative_control": "Naive slotwise integration assigns a fresh independent field with the same prior to each slot; within-slot correlations are preserved",
        "positive_controls": ["cumulative trajectory Schur message", "recursive four-slot marginalization retaining pose-field correlations", "zero map-uncertainty limit"],
        "sampling": "Independent prior pose perturbations, one shared field perturbation per replicate, independent radio and relative observation noises; same realization for all methods",
    }


def geometry(config):
    steps, robots = config["steps"], config["robots"]
    times = np.arange(steps, dtype=float)
    positions = np.empty((steps, robots, 2))
    for k in range(steps):
        center = np.array([2.2 + 0.23 * k, 3.1 + 0.32 * np.sin(0.4 * k)])
        for r in range(robots):
            angle = 2 * np.pi * r / robots + 0.065 * k
            positions[k, r] = center + 1.3 * np.array([np.cos(angle), np.sin(angle)])
    axis = np.asarray(config["field_centers_axis_m"])
    centers = np.array([(x, y) for x in axis for y in axis])
    nodes, weights = np.polynomial.legendre.leggauss(config["quadrature_nodes"])
    nodes, weights = 0.5 * (nodes + 1), 0.5 * weights
    mean_field = np.asarray(config["field_mean_db_per_m"])

    def line_basis(start, finish):
        points = start[None, :] + nodes[:, None] * (finish - start)[None, :]
        sq = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        basis = np.exp(-sq / (2 * config["field_width_m"] ** 2))
        return np.linalg.norm(finish - start) * (weights @ basis)

    def radio(start, finish):
        distance = np.linalg.norm(finish - start)
        return (-10 * config["pathloss_gamma"] * np.log10(distance)
                - line_basis(start, finish) @ mean_field)

    edges = [(0, 1), (0, 2), (1, 2)]
    p = 2 * robots
    a = np.zeros((steps * len(edges), steps * p))
    b = np.zeros((steps * len(edges), len(centers)))
    ell = np.zeros((steps * len(edges) * 2, steps * p))
    expected_radio = []
    eps = config["finite_difference_m"]
    for k in range(steps):
        for e, (r, s) in enumerate(edges):
            row = len(edges) * k + e
            start, finish = positions[k, r], positions[k, s]
            b[row] = -line_basis(start, finish)
            expected_radio.append(radio(start, finish))
            for d in range(2):
                shift = np.eye(2)[d] * eps
                a[row, k * p + 2 * r + d] = (radio(start + shift, finish) - radio(start - shift, finish)) / (2 * eps)
                a[row, k * p + 2 * s + d] = (radio(start, finish + shift) - radio(start, finish - shift)) / (2 * eps)
                ell[2 * row + d, k * p + 2 * r + d] = -1
                ell[2 * row + d, k * p + 2 * s + d] = 1
    p0 = (config["initial_common_position_sigma_m"] ** 2 * np.kron(np.ones((robots, robots)), np.eye(2))
          + config["initial_individual_position_sigma_m"] ** 2 * np.eye(p))
    process = config["process_individual_sigma_m"] ** 2 * np.eye(p)
    prior_pose = np.kron(np.ones((steps, steps)), p0) + np.kron(np.minimum.outer(times, times), process)
    d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    prior_field = config["field_std_db_per_m"] ** 2 * (
        0.8 * np.exp(-d2 / (2 * config["field_correlation_length_m"] ** 2)) + 0.2 * np.eye(len(centers)))
    return positions, centers, a, b, ell, prior_pose, prior_field, np.asarray(expected_radio)


def schur(h, info, keep):
    keep = np.asarray(keep, dtype=int)
    discard = np.setdiff1d(np.arange(len(h)), keep)
    if not len(discard):
        return h.copy(), info.copy()
    hkd = h[np.ix_(keep, discard)]
    hdd = h[np.ix_(discard, discard)]
    return (sym(h[np.ix_(keep, keep)] - hkd @ np.linalg.solve(hdd, hkd.T)),
            info[keep] - hkd @ np.linalg.solve(hdd, info[discard]))


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    positions, centers, a, b, ell, pp, pf, radio_mean = geometry(config)
    n, steps, robots = config["replicates"], config["steps"], config["robots"]
    dim, fdim, p = a.shape[1], b.shape[1], 2 * robots
    rr, rl = config["radio_sigma_db"] ** 2, config["relative_position_sigma_m"] ** 2
    rng = np.random.default_rng(config["seed"])
    xtruth = np.linalg.cholesky(pp) @ rng.standard_normal((dim, n))
    ftruth = np.linalg.cholesky(pf) @ rng.standard_normal((fdim, n))
    radio_noise = np.sqrt(rr) * rng.standard_normal((len(a), n))
    relative_noise = np.sqrt(rl) * rng.standard_normal((len(ell), n))
    yr = a @ xtruth + b @ ftruth + radio_noise
    yl = ell @ xtruth + relative_noise

    task_h = inv(pp) + ell.T @ ell / rl
    task_b = ell.T @ yl / rl
    field_h = inv(pf) + b.T @ b / rr
    cross_h = a.T @ b / rr
    full_h = np.block([[task_h + a.T @ a / rr, cross_h], [cross_h.T, field_h]])
    full_b = np.concatenate([task_b + a.T @ yr / rr, b.T @ yr / rr])
    joint_cov_all = inv(full_h)
    joint_mean_all = np.linalg.solve(full_h, full_b)
    covariances = {"joint": joint_cov_all[:dim, :dim]}
    means = {"joint": joint_mean_all[:dim]}

    # Correct canonical radio likelihood: one field prior, all distinct poses.
    message_h = sym(a.T @ a / rr - cross_h @ np.linalg.solve(field_h, cross_h.T))
    message_b = a.T @ yr / rr - cross_h @ np.linalg.solve(field_h, b.T @ yr / rr)
    corrected_h, corrected_b = task_h + message_h, task_b + message_b
    covariances["corrected"] = inv(corrected_h)
    means["corrected"] = np.linalg.solve(corrected_h, corrected_b)

    # Naive sends an independently marginalized field prior in each slot.
    naive_h, naive_b = task_h.copy(), task_b.copy()
    slot_cov = []
    for k in range(steps):
        rows = slice(k * 3, (k + 1) * 3)
        ak, bk, yk = a[rows], b[rows], yr[rows]
        vk = rr * np.eye(3) + bk @ pf @ bk.T
        slot_cov.append(vk)
        naive_h += ak.T @ np.linalg.solve(vk, ak)
        naive_b += ak.T @ np.linalg.solve(vk, yk)
    covariances["naive_slotwise"] = inv(naive_h)
    means["naive_slotwise"] = np.linalg.solve(naive_h, naive_b)

    # Verify cumulative replacement/increments at every prefix. Previous
    # message is embedded, then replaced; no observation is counted twice.
    prefix_checks = []
    old_h = np.zeros((0, 0))
    old_b = np.zeros((0, n))
    for k in range(1, steps + 1):
        kd = k * p
        ak, bk, yk = a[:3 * k, :kd], b[:3 * k], yr[:3 * k]
        lk, lyk = ell[:6 * k, :kd], yl[:6 * k]
        base_h = inv(pp[:kd, :kd]) + lk.T @ lk / rl
        base_b = lk.T @ lyk / rl
        khf = inv(pf) + bk.T @ bk / rr
        khx = ak.T @ bk / rr
        kh = sym(ak.T @ ak / rr - khx @ np.linalg.solve(khf, khx.T))
        kb = ak.T @ yk / rr - khx @ np.linalg.solve(khf, bk.T @ yk / rr)
        embedded_h = np.zeros_like(kh)
        embedded_b = np.zeros_like(kb)
        embedded_h[:len(old_h), :len(old_h)] = old_h
        embedded_b[:len(old_b)] = old_b
        delta_h, delta_b = kh - embedded_h, kb - embedded_b
        out_h, out_b = base_h + embedded_h + delta_h, base_b + embedded_b + delta_b
        kh_joint = np.block([[base_h + ak.T @ ak / rr, khx], [khx.T, khf]])
        kb_joint = np.concatenate([base_b + ak.T @ yk / rr, bk.T @ yk / rr])
        refcov, refmean = inv(kh_joint)[:kd, :kd], np.linalg.solve(kh_joint, kb_joint)[:kd]
        prefix_checks.append({"slots": k, "pose_dimension": kd,
            "max_mean_difference_m": float(np.max(np.abs(np.linalg.solve(out_h, out_b) - refmean))),
            "relative_covariance_difference": float(np.linalg.norm(inv(out_h) - refcov) / np.linalg.norm(refcov)),
            "increment_min_eigenvalue": float(np.linalg.eigvalsh(sym(delta_h))[0])})
        old_h, old_b = kh, kb

    # True recursive fixed-window update: retain field correlations while
    # eliminating expired pose variables. Full posterior then a Schur message
    # is a replacement, not an increment added to an already absorbed prior.
    wh = block_diag(inv(pp[:p, :p]), inv(pf))
    wb = np.zeros((p + fdim, n))
    retained = [0]
    window_checks = []
    for k in range(steps):
        if k:
            pose_count = len(retained) * p
            ids = np.r_[np.arange(pose_count), np.arange(pose_count + p, pose_count + p + fdim)]
            nh, nb = np.zeros((len(wh) + p, len(wh) + p)), np.zeros((len(wb) + p, n))
            nh[np.ix_(ids, ids)], nb[ids] = wh, wb
            wh, wb = nh, nb
            proc = np.concatenate([-np.eye(p), np.eye(p)], axis=1) / config["process_individual_sigma_m"]
            ids = np.arange(pose_count - p, pose_count + p)
            wh[np.ix_(ids, ids)] += proc.T @ proc
            retained.append(k)
        wd = len(retained) * p
        ids = np.r_[np.arange(wd - p, wd), np.arange(wd, wd + fdim)]
        jac = np.concatenate([a[3 * k:3 * (k + 1), k * p:(k + 1) * p], b[3 * k:3 * (k + 1)]], axis=1)
        wh[np.ix_(ids, ids)] += jac.T @ jac / rr
        wb[ids] += jac.T @ yr[3 * k:3 * (k + 1)] / rr
        lk = ell[6 * k:6 * (k + 1), k * p:(k + 1) * p]
        wh[wd - p:wd, wd - p:wd] += lk.T @ lk / rl
        wb[wd - p:wd] += lk.T @ yl[6 * k:6 * (k + 1)] / rl
        if len(retained) > config["retained_window_steps"]:
            wh, wb = schur(wh, wb, np.arange(p, len(wh)))
            retained.pop(0)
            wd -= p
        replacement_h, replacement_b = schur(wh, wb, np.arange(wd))
        kd = (k + 1) * p
        ak, bk, lk = a[:3 * (k + 1), :kd], b[:3 * (k + 1)], ell[:6 * (k + 1), :kd]
        khx = ak.T @ bk / rr
        hbatch = np.block([[inv(pp[:kd, :kd]) + lk.T @ lk / rl + ak.T @ ak / rr, khx],
                          [khx.T, inv(pf) + bk.T @ bk / rr]])
        bbatch = np.concatenate([lk.T @ yl[:len(lk)] / rl + ak.T @ yr[:len(ak)] / rr,
                                bk.T @ yr[:len(ak)] / rr])
        keeps = np.concatenate([np.arange(s * p, (s + 1) * p) for s in retained])
        cref, mref = inv(hbatch)[np.ix_(keeps, keeps)], np.linalg.solve(hbatch, bbatch)[keeps]
        window_checks.append({"slot": k, "retained_slots": retained.copy(),
            "max_mean_difference_m": float(np.max(np.abs(np.linalg.solve(replacement_h, replacement_b) - mref))),
            "relative_covariance_difference": float(np.linalg.norm(inv(replacement_h) - cref) / np.linalg.norm(cref)),
            "retained_pose_field_cross_norm": float(np.linalg.norm(wh[:wd, wd:]))})

    common = np.zeros((2 * steps, dim))
    for k in range(steps):
        common[2 * k:2 * k + 2, k * p:(k + 1) * p] = np.tile(np.eye(2), (1, robots)) / robots
    terminal = common[-2:]
    records, stats = [], {}
    common_curves, common_covs = {}, {}
    for name in METHODS:
        covariance, mean = covariances[name], means[name]
        errors = mean - xtruth
        common_errors = common @ errors
        common_curves[name] = np.linalg.norm(common_errors.reshape(steps, 2, n), axis=1)
        cc = common @ covariance @ common.T
        common_covs[name] = cc
        nees = np.sum(errors * np.linalg.solve(covariance, errors), axis=0) / dim
        ce = terminal @ errors
        terminal_nees = np.sum(ce * np.linalg.solve(terminal @ covariance @ terminal.T, ce), axis=0) / 2
        coverage = np.mean(np.abs(errors) <= 1.95996398454 * np.sqrt(np.diag(covariance))[:, None], axis=0)
        rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        mean_discrepancy = mean - means["joint"]
        stats[name] = {
            "position_coordinate_rmse_m": ci(rmse),
            "terminal_common_error_m": ci(common_curves[name][-1]),
            "normalized_full_trajectory_nees": ci(nees),
            "normalized_terminal_common_nees": ci(terminal_nees),
            "coordinatewise_95pct_coverage": ci(coverage),
            "posterior_mean_rms_difference_from_joint_m": float(np.sqrt(np.mean(mean_discrepancy ** 2))),
            "posterior_mean_max_abs_difference_from_joint_m": float(np.max(np.abs(mean_discrepancy))),
            "relative_covariance_frobenius_difference_from_joint": float(np.linalg.norm(covariance - covariances["joint"]) / np.linalg.norm(covariances["joint"])),
            "reported_terminal_common_rms_uncertainty_m": float(np.sqrt(np.trace(terminal @ covariance @ terminal.T))),
            "reported_position_coordinate_rms_uncertainty_m": float(np.sqrt(np.trace(covariance) / dim)),
        }
        for j in range(n):
            records.append({"replicate": j, "method": name, "position_coordinate_rmse_m": float(rmse[j]),
                "terminal_common_error_m": float(common_curves[name][-1, j]), "normalized_nees": float(nees[j]),
                "terminal_common_normalized_nees": float(terminal_nees[j]), "coverage95": float(coverage[j]),
                "posterior_mean_rms_difference_from_joint_m": float(np.sqrt(np.mean(mean_discrepancy[:, j] ** 2)))})

    vtrue, vnaive = rr * np.eye(len(a)) + b @ pf @ b.T, block_diag(*slot_cov)
    # In the supplied (zero-uncertainty) field limit the slotwise and shared
    # construction both reduce to independent radio noise.
    hz = task_h + a.T @ a / rr
    hzslot = task_h.copy()
    for k in range(steps):
        hzslot += a[k * 3:(k + 1) * 3].T @ a[k * 3:(k + 1) * 3] / rr
    summary = {"config": config, "dimensions": {"trajectory": dim, "field": fdim, "radio": len(a)},
        "methods": stats, "full_nees_null_ci95": (chi2.ppf([0.025, 0.975], n * dim) / (n * dim)).tolist(),
        "terminal_common_nees_null_ci95": (chi2.ppf([0.025, 0.975], n * 2) / (n * 2)).tolist(),
        "cumulative_prefix_checks": prefix_checks, "recursive_window_checks": window_checks,
        "zero_map_uncertainty_precision_relative_difference": float(np.linalg.norm(hz - hzslot) / np.linalg.norm(hz)),
        "discarded_radio_cross_slot_covariance_relative_norm": float(np.linalg.norm(vtrue - vnaive) / np.linalg.norm(vtrue)),
        "condition_numbers": {"joint_precision": float(np.linalg.cond(full_h)), "pose_covariance": float(np.linalg.cond(covariances["joint"]))},
        "calibration_note": "The full trajectory NEES has d=72 degrees of freedom per independent replicate; terminal centroid NEES has d=2. Coordinates/time points within a replicate are not treated as independent samples.",
    }
    exact_rows = prefix_checks + window_checks
    summary["exactness_checks_passed"] = bool(
        all(r["max_mean_difference_m"] < 1e-8 and r["relative_covariance_difference"] < 1e-8 for r in exact_rows)
        and summary["zero_map_uncertainty_precision_relative_difference"] < 1e-12)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "replicates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "raw_draws_and_posteriors.npz", nominal_positions=positions,
        field_centers=centers, A=a, B=b, relative_jacobian=ell, prior_pose_cov=pp, prior_field_cov=pf,
        nominal_radio=radio_mean, true_pose_perturbations=xtruth, true_field_perturbations=ftruth,
        radio_residuals=yr, relative_residuals=yl, radio_noise=radio_noise, relative_noise=relative_noise,
        true_marginal_radio_cov=vtrue, naive_marginal_radio_cov=vnaive,
        **{f"{name}_posterior_mean": means[name] for name in METHODS},
        **{f"{name}_posterior_covariance": covariances[name] for name in METHODS})
    with (output / "posterior_means_replicate0000.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["slot", "robot", "axis", "nominal_m", "truth_m", "joint_mean_m", "corrected_mean_m", "naive_mean_m", "joint_std_m", "naive_std_m"])
        for k in range(steps):
            for robot in range(robots):
                for axis in range(2):
                    idx = k * p + 2 * robot + axis
                    nom = positions[k, robot, axis]
                    writer.writerow([k, robot, "xy"[axis], nom, nom + xtruth[idx, 0],
                        *[nom + means[m][idx, 0] for m in METHODS], np.sqrt(covariances["joint"][idx, idx]),
                        np.sqrt(covariances["naive_slotwise"][idx, idx])])
    make_figures(config, output, positions, centers, common, xtruth, means, covariances,
                 common_curves, common_covs, records, summary, vtrue, vnaive)
    runner = Path(__file__)
    source_copy = output / "run_message_validation.py"
    source_copy.write_bytes(runner.read_bytes())
    manifest = {"python": platform.python_version(), "numpy": np.__version__,
        "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in sorted(output.iterdir())
                  if file.is_file() and file.name != "manifest.json" and file.suffix != ".log"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"exactness_checks_passed": summary["exactness_checks_passed"], "methods": stats}, indent=2), flush=True)
    if not summary["exactness_checks_passed"]:
        raise SystemExit("Exactness checks failed")


def make_figures(config, output, positions, centers, common, truth, means, covs,
                 common_curves, common_covs, records, summary, vtrue, vnaive):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    gx, gy = np.meshgrid(np.linspace(0, 7.5, 100), np.linspace(0, 7, 100))
    points = np.stack([gx, gy], axis=-1)
    field = np.exp(-np.sum((points[:, :, None] - centers[None, None]) ** 2, axis=-1)
                   / (2 * config["field_width_m"] ** 2)) @ np.asarray(config["field_mean_db_per_m"])
    ax.contourf(gx, gy, field, levels=12, cmap="Greys", alpha=0.35)
    for r in range(config["robots"]):
        ax.plot(positions[:, r, 0], positions[:, r, 1], ".-", label=f"Robot {r + 1}")
        ax.scatter(*positions[0, r], s=32, marker="s")
    for k in (0, config["steps"] - 1):
        for r, s in ((0, 1), (0, 2), (1, 2)):
            ax.plot(positions[k, [r, s], 0], positions[k, [r, s], 1], color="0.4", lw=0.65, alpha=0.6)
    ax.set(xlabel="x (m)", ylabel="y (m)", title="(a) Moving geometry and mean field")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[0, 1]
    tt = np.arange(config["steps"])
    for name, label, color in zip(METHODS, LABELS, COLORS):
        curve = common_curves[name]
        avg = np.mean(curve, axis=1)
        half = student_t.ppf(.975, config["replicates"] - 1) * np.std(curve, axis=1, ddof=1) / np.sqrt(config["replicates"])
        ax.plot(tt, avg, color=color, label=label, linestyle="--" if name == "corrected" else "-", linewidth=1.7)
        if name != "corrected":
            ax.fill_between(tt, avg-half, avg+half, color=color, alpha=.16)
    ax.set(xlabel="Time (s)", ylabel="Centroid error (m)", title="(b) Posterior error; 95% mean CI")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for i, name in enumerate(METHODS):
        for offset, key, marker, label in ((-.10, "normalized_full_trajectory_nees", "o", "All 72 coordinates"),
                                         (.10, "normalized_terminal_common_nees", "s", "Terminal centroid (2D)")):
            stat = summary["methods"][name][key]
            ax.errorbar(i + offset, stat["mean"], yerr=[[stat["mean"] - stat["ci95"][0]], [stat["ci95"][1] - stat["mean"]]],
                        color=COLORS[i], marker=marker, capsize=3, linestyle="none", label=label if i == 0 else None)
    ax.axhline(1, color="0.3", lw=.8)
    ax.set(xticks=np.arange(3), xticklabels=["Joint", "Corrected", "Naive"], ylabel="NEES / degrees of freedom", title="(c) Consistency; 95% mean CI")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    vals = [summary["methods"][m]["relative_covariance_frobenius_difference_from_joint"] for m in METHODS[1:]]
    ax.bar([0, 1], vals, color=COLORS[1:], width=.55)
    ax.set_yscale("log")
    ax.set(xticks=[0, 1], xticklabels=["Corrected", "Naive"], ylabel="Relative covariance difference", title="(d) Difference from joint posterior")
    ax.set_ylim(1e-14, 2)
    for i, val in enumerate(vals):
        ax.text(i, val * 2.5, f"{val:.2e}", ha="center", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"experiment_a_message_validation.{ext}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    truth_centroid = (common @ truth[:, 0]).reshape(config["steps"], 2)
    for d, ax in enumerate(axes):
        ax.plot(tt, truth_centroid[:, d], color="black", marker=".", label="True perturbation")
        for name, label, color in zip(METHODS, LABELS, COLORS):
            cm = (common @ means[name][:, 0]).reshape(config["steps"], 2)[:, d]
            std = np.sqrt(np.diag(common_covs[name])).reshape(config["steps"], 2)[:, d]
            ax.plot(tt, cm, color=color, linestyle="--" if name == "corrected" else "-", label=label)
            if name != "corrected":
                ax.fill_between(tt, cm - 1.959964 * std, cm + 1.959964 * std, color=color, alpha=.14)
        ax.set(xlabel="Time (s)", ylabel=f"Centroid {('x', 'y')[d]} perturbation (m)", title="Prespecified replicate 0; 95% intervals")
    axes[0].legend(fontsize=6.8, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"experiment_a_posterior_example.{ext}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3), constrained_layout=True)
    for ax, covariance, title in zip(axes, (vtrue, vnaive), ("Shared field: correct radio correlation", "Naive: cross-slot correlation discarded")):
        std = np.sqrt(np.diag(covariance))
        im = ax.imshow(covariance / np.outer(std, std), vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
        ax.set(xlabel="Radio observation index", ylabel="Radio observation index", title=title)
    fig.colorbar(im, ax=axes, shrink=.8, label="Correlation")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"experiment_a_temporal_correlations.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-config-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else default_config()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preregistered_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not args.write_config_only:
        run(config, args.output)


if __name__ == "__main__":
    main()
