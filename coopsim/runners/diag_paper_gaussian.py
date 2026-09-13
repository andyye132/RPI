"""Deterministic checks of the updated paper's specified Gaussian constructions.

This deliberately small float64 experiment retains distinct variables at distinct
times, correlated pose-field priors, and both canonical parameters. It validates
the algebra independently of the nonlinear current-pose A-HY/A-HYM estimator; it
does not establish moving-window equivalence for that implementation.

Run on a compute node (no JAX/GPU required)::

    python -m coopsim.runners.diag_paper_gaussian --output results/gaussian.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sym(matrix):
    return (matrix + matrix.T) * 0.5


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))


def marginal(precision, information, keep):
    """Eliminate all other variables from 1/2 x.T H x - b.T x."""
    keep = np.asarray(keep, dtype=int)
    discard = np.setdiff1d(np.arange(len(information)), keep)
    hkk = precision[np.ix_(keep, keep)]
    bk = information[keep]
    if not len(discard):
        return hkk.copy(), bk.copy()
    hkd = precision[np.ix_(keep, discard)]
    hdd = precision[np.ix_(discard, discard)]
    return (sym(hkk - hkd @ np.linalg.solve(hdd, hkd.T)),
            bk - hkd @ np.linalg.solve(hdd, information[discard]))


def embed(precision, information, indices, size):
    result_h, result_b = np.zeros((size, size)), np.zeros(size)
    result_h[np.ix_(indices, indices)] = precision
    result_b[indices] = information
    return result_h, result_b


def check_posterior(precision, information, full_h, full_b, keep, tolerance):
    """Reference uses a full covariance solve, independent of Schur elimination."""
    full_cov = np.linalg.solve(full_h, np.eye(len(full_b)))
    full_mean = np.linalg.solve(full_h, full_b)
    expected_cov = full_cov[np.ix_(keep, keep)]
    expected_mean = full_mean[keep]
    expected_h = np.linalg.solve(expected_cov, np.eye(len(keep)))
    expected_b = expected_h @ expected_mean
    covariance = np.linalg.solve(precision, np.eye(len(information)))
    mean = np.linalg.solve(precision, information)
    errors = {
        "precision_relative_error": relative_error(precision, expected_h),
        "information_vector_relative_error": relative_error(information, expected_b),
        "mean_relative_error": relative_error(mean, expected_mean),
        "covariance_relative_error": relative_error(covariance, expected_cov),
    }
    errors["passed"] = max(errors.values()) < tolerance
    return errors


def factors(rng, pose_dim, field_dim):
    # Two-robot position differences: these task observations annihilate the two
    # common translation generators. Process factors preserve distinct times.
    shape = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]])
    process = np.concatenate([-np.eye(pose_dim), np.eye(pose_dim)], axis=1) / 0.6
    channel = rng.normal(size=(5, pose_dim + field_dim))
    return (shape, rng.normal(size=2), process, rng.normal(size=pose_dim),
            channel, rng.normal(size=5))


def add_factor(precision, information, jacobian, observation, indices):
    precision[np.ix_(indices, indices)] += jacobian.T @ jacobian
    information[indices] += jacobian.T @ observation


def run(seed=20260905, steps=8, window=3, tolerance=1e-10):
    if steps < 3 or not 1 < window < steps:
        raise ValueError("Require steps >= 3 and 1 < window < steps")
    rng = np.random.default_rng(seed)
    p, f = 4, 3
    prior_root = rng.normal(size=(p + f, p + f))
    prior_h = prior_root.T @ prior_root + 2 * np.eye(p + f)
    prior_b = prior_h @ rng.normal(size=p + f)
    initial_cross_norm = float(np.linalg.norm(prior_h[:p, p:]))

    # Growing trajectory: outer owns pose-only factors; inner owns the complete
    # correlated prior and every channel factor. A field prior is assigned once.
    channel_h, channel_b = prior_h.copy(), prior_b.copy()
    task_h, task_b = np.zeros_like(prior_h), np.zeros_like(prior_b)
    old_message_h, old_message_b = np.zeros((0, 0)), np.zeros(0)
    window_h, window_b = prior_h.copy(), prior_b.copy()
    retained_ids = [0]
    retained_checks, moving_checks, cross_contrasts = [], [], []
    first_batch = None
    for t in range(steps):
        zdim = (t + 1) * p
        dim = zdim + f
        if t:
            old_dim = t * p
            indices = np.r_[np.arange(old_dim), np.arange(zdim, dim)]
            channel_h, channel_b = embed(channel_h, channel_b, indices, dim)
            task_h, task_b = embed(task_h, task_b, indices, dim)
            nw = len(retained_ids) * p
            wi = np.r_[np.arange(nw), np.arange(nw + p, nw + p + f)]
            window_h, window_b = embed(window_h, window_b, wi, nw + p + f)
            retained_ids.append(t)

        shape, shape_y, process, process_y, radio, radio_y = factors(rng, p, f)
        current = np.arange(t * p, zdim)
        channel_indices = np.r_[current, np.arange(zdim, dim)]
        add_factor(task_h, task_b, shape, shape_y, current)
        add_factor(channel_h, channel_b, radio, radio_y, channel_indices)
        nw = len(retained_ids) * p
        wc = np.arange(nw - p, nw)
        add_factor(window_h, window_b, shape, shape_y, wc)
        add_factor(window_h, window_b, radio, radio_y, np.r_[wc, np.arange(nw, nw + f)])
        if t:
            add_factor(task_h, task_b, process, process_y, np.arange((t - 1) * p, zdim))
            add_factor(window_h, window_b, process, process_y, np.arange(nw - 2 * p, nw))

        full_h, full_b = channel_h + task_h, channel_b + task_b
        message_h, message_b = marginal(channel_h, channel_b, np.arange(zdim))
        previous_h, previous_b = embed(old_message_h, old_message_b,
                                      np.arange(t * p), zdim)
        delta_h, delta_b = message_h - previous_h, message_b - previous_b
        outer_h = task_h[:zdim, :zdim] + previous_h + delta_h
        outer_b = task_b[:zdim] + previous_b + delta_b
        check = check_posterior(outer_h, outer_b, full_h, full_b,
                                np.arange(zdim), tolerance)
        check.update(slot=t, retained_pose_dimension=zdim,
                     increment_min_eigenvalue=float(np.linalg.eigvalsh(sym(delta_h))[0]))
        retained_checks.append(check)
        old_message_h, old_message_b = message_h.copy(), message_b.copy()
        if t == 0:
            first_batch = check.copy()

        if len(retained_ids) > window:
            # Absorb all old contributions into one correlated retained prior.
            # Replace the old outer message after discarding coordinates.
            window_h, window_b = marginal(window_h, window_b,
                                           np.arange(p, len(window_b)))
            retained_ids.pop(0)
            nw -= p
            damaged_h = window_h.copy()
            damaged_h[:nw, nw:] = 0.0
            damaged_h[nw:, :nw] = 0.0
            dh, db = marginal(damaged_h, window_b, np.arange(nw))
            kept_pose = np.concatenate([np.arange(k * p, (k + 1) * p)
                                       for k in retained_ids])
            bad = check_posterior(dh, db, full_h, full_b, kept_pose, tolerance)
            bad.update(slot=t, discarded_cross_block_norm=float(
                np.linalg.norm(window_h[:nw, nw:])))
            cross_contrasts.append(bad)

        # Full replacement also preserves canonical information vectors. There
        # is no PSD clipping or recycling of old observations as new evidence.
        replacement_h, replacement_b = marginal(window_h, window_b, np.arange(nw))
        kept_pose = np.concatenate([np.arange(k * p, (k + 1) * p) for k in retained_ids])
        check = check_posterior(replacement_h, replacement_b, full_h, full_b,
                                kept_pose, tolerance)
        check.update(slot=t, retained_slots=retained_ids.copy(),
                     retained_pose_dimension=nw,
                     pose_field_cross_block_norm=float(np.linalg.norm(window_h[:nw, nw:])))
        moving_checks.append(check)

    # Paper's exact two-observation counterexample: y_s = z + f + noise,
    # one unit field prior. Include a nonzero vector to test posterior means too.
    h = np.array([[3., 2.], [2., 3.]])  # unit pose prior plus the two factors
    b = np.array([1., 1.])              # y = [2, -1], zero-mean priors
    correct_h, correct_b = marginal(h, b, [0])
    bad_h, bad_b = np.array([[2.]]), np.array([0.5])
    independent = check_posterior(bad_h, bad_b, h, b, [0], tolerance)
    independent.update(joint_channel_information=float(correct_h[0, 0] - 1),
                       independent_channel_information=1.0,
                       information_ratio=float(1.0 / (correct_h[0, 0] - 1)),
                       joint_channel_vector=float(correct_b[0]),
                       independent_channel_vector=float(bad_b[0]))

    # A current-pose block obtained by identifying x0 == x1 describes a
    # different model, even though both coordinates have translation generator 1.
    trajectory_h = np.array([[3., -1., 1.], [-1., 3., 1.], [1., 1., 3.]])
    trajectory_b = np.array([2., -1., 1.])
    current_h, current_b = marginal(trajectory_h, trajectory_b, [1])
    identify = np.array([[1., 0.], [1., 0.], [0., 1.]])
    tied_h, tied_b = marginal(identify.T @ trajectory_h @ identify,
                              identify.T @ trajectory_b, [0])
    tied = {
        "trajectory_current_precision": float(current_h[0, 0]),
        "identified_pose_precision": float(tied_h[0, 0]),
        "trajectory_current_mean": float(current_b[0] / current_h[0, 0]),
        "identified_pose_mean": float(tied_b[0] / tied_h[0, 0]),
        "precision_difference": float(abs(tied_h[0, 0] - current_h[0, 0])),
    }

    # Replacing a factor with a weaker one gives an indefinite/negative
    # correction. Clipping it changes both the covariance and posterior mean.
    base_h, base_b = np.diag([2., 3.]), np.array([0.5, -0.2])
    direction = np.array([1., 0.7])
    old_h, new_h = 4 * np.outer(direction, direction), np.outer(direction, direction)
    old_b, new_b = 4 * direction, -direction
    correction_h, correction_b = new_h - old_h, new_b - old_b
    eig, vec = np.linalg.eigh(correction_h)
    clipped = (vec * np.maximum(eig, 0)) @ vec.T
    replacement = check_posterior(base_h + old_h + correction_h,
                                  base_b + old_b + correction_b,
                                  base_h + new_h, base_b + new_b, [0, 1], tolerance)
    clip_check = check_posterior(base_h + old_h + clipped,
                                 base_b + old_b + correction_b,
                                 base_h + new_h, base_b + new_b, [0, 1], tolerance)
    replacement.update(correction_min_eigenvalue=float(eig[0]),
                       psd_clipped_contrast=clip_check)

    # A prior mean reset to the linearization point silently discards b. This
    # illustrates estimator.marginalize's current approximation away from a
    # stationary point; no change to historical estimator behavior is made here.
    dropped_vector = check_posterior(correct_h, np.zeros_like(correct_b),
                                     h, b, [0], tolerance)
    positives = all(r["passed"] for r in retained_checks + moving_checks) and replacement["passed"]
    negatives = (all(not r["passed"] for r in cross_contrasts)
                 and not independent["passed"] and not clip_check["passed"]
                 and not dropped_vector["passed"] and tied["precision_difference"] > 1e-3)
    return {
        "scope": "Specified linear Gaussian systems; not nonlinear A-HY/A-HYM equivalence",
        "dtype": "float64", "seed": seed, "steps": steps, "window": window,
        "pose_dimension_per_slot": p, "field_dimension": f, "tolerance": tolerance,
        "initial_pose_field_cross_block_norm": initial_cross_norm,
        "batch_correlated_prior": first_batch,
        "retained_trajectory": retained_checks,
        "moving_window_replacement": moving_checks,
        "discarded_cross_blocks_contrast": cross_contrasts,
        "independent_slot_elimination_contrast": independent,
        "identifying_distinct_pose_times_contrast": tied,
        "factor_replacement": replacement,
        "discarded_information_vector_contrast": dropped_vector,
        "passed": bool(positives and negatives),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()
    result = run(seed=args.seed, steps=args.steps, window=args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    worst = max(max(row[key] for key in row if key.endswith("relative_error"))
                for group in ("retained_trajectory", "moving_window_replacement")
                for row in result[group])
    print(f"Gaussian checks passed={result['passed']} worst_relative_error={worst:.3e}")
    print(f"Saved {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
