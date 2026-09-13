"""Small implementation checks for gauge normalization and finite-grid identity.

Uses the repository's factors, bilinear interpolation, 64-point line quadrature,
and generated fields. The finite-grid sensitivity uses numpy.gradient with
centered interior differences and first-order one-sided boundary differences.
It is intentionally compared with the actual automatic derivative of emission;
the two derivative constructions need not agree. No continuum or persistent-
excitation conclusion is inferred from a small numerical residual.

Run on a compute node with JAX_PLATFORMS=cpu and --output <json path>.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import factors, field, world
from coopsim.config import crn_keys, load_config


def generators(poses, dtype=np.float64):
    rows = np.asarray(factors.gauge_generators(
        jnp.asarray(poses, dtype=dtype)[None], jnp.ones(1, bool)), dtype=dtype)
    g = rows.T
    norms = np.linalg.norm(g, axis=0)
    return g / norms, norms


def relative_checks(poses, rng):
    n = len(poses)
    weights = {
        "unit": np.ones(3 * n * n),
        "uniform_1e-3": np.full(3 * n * n, 1e-3),
        "uniform_1e3": np.full(3 * n * n, 1e3),
        "six_decades": 10 ** rng.uniform(-3, 3, 3 * n * n),
        "one_directed_link": np.zeros(3 * n * n),
    }
    weights["one_directed_link"][3:6] = 1e6
    self_rows = np.repeat(np.eye(n, dtype=bool).reshape(-1), 3)
    results = {}
    for dtype in (np.float32, np.float64):
        x = jnp.asarray(poses, dtype=dtype)
        z = jnp.zeros((1, n, n, 3), dtype=dtype)
        _, ji, jj = factors.relpose_factors(x[None], z)
        ji, jj = np.asarray(ji, dtype=dtype), np.asarray(jj, dtype=dtype)
        jacobian = np.zeros((3 * n * n, 3 * n), dtype=dtype)
        for k in range(n * n):
            i, j = divmod(k, n)
            rows = slice(3 * k, 3 * k + 3)
            jacobian[rows, 3 * i:3 * i + 3] += ji[k]
            jacobian[rows, 3 * j:3 * j + 3] += jj[k]
        gn, _ = generators(poses, dtype)
        residual = jacobian @ gn
        eps = float(np.finfo(dtype).eps)
        orth = float(np.linalg.norm(gn.T @ gn - np.eye(3)))
        rows = []
        for name, values in weights.items():
            w = values.astype(dtype).copy()
            w[self_rows] = 0
            h = jacobian.T @ (w[:, None] * jacobian)
            projected = gn.T @ h @ gn
            weighted_residual = np.sqrt(w)[:, None] * residual
            factor_gram = weighted_residual.T @ weighted_residual
            den = max(float(np.linalg.norm(h)), np.finfo(dtype).tiny)
            rows.append({
                "weights": name,
                "information_norm": float(np.linalg.norm(h)),
                "direct_projected_information_relative_norm": float(np.linalg.norm(projected) / den),
                "factor_residual_information_relative_norm": float(np.linalg.norm(factor_gram) / den),
                "direct_projected_min_eigenvalue": float(np.linalg.eigvalsh(
                    (projected + projected.T) * 0.5)[0]),
            })
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(jacobian), 1.0))
        results[np.dtype(dtype).name] = {
            "epsilon": eps,
            "orthonormality_error": orth,
            "relative_jacobian_null_residual": relative,
            "weight_sweep": rows,
            "passed": bool(orth < 100 * eps and relative < 100 * eps
                           and max(row["direct_projected_information_relative_norm"]
                                   for row in rows) < 100 * eps),
        }
    return results


def run(seed=20260905, grid=40):
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX_ENABLE_X64=1 is required for the finite-grid instrument")
    rng = np.random.default_rng(seed)
    cfg = load_config(overrides=[f"shapes.grid={grid}"])
    n, ext = cfg.shapes.n_robots, cfg.world.extent
    poses = np.column_stack([rng.uniform(0.16 * ext, 0.84 * ext, (n, 2)),
                             rng.uniform(-0.5, 0.5, n)])
    gn, norms = generators(poses)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    grad_emit = jax.jit(jax.vmap(jax.value_and_grad(
        lambda a, b, f: field.emission(a, b, f, ext, P0=cfg.channel.p0_dbm,
                                       gamma=cfg.channel.gamma), argnums=(0, 1)),
        in_axes=(0, 0, None)))
    grad_field = jax.jit(jax.vmap(jax.grad(
        lambda f, a, b: field.line_integral(a, b, f, ext), argnums=0),
        in_axes=(None, 0, 0)))

    def instrument(f, x):
        values, (gi, gj) = grad_emit(jnp.asarray(x[iu, :2]),
                                    jnp.asarray(x[ju, :2]), jnp.asarray(f))
        jac = np.zeros((m, 3 * n))
        for k, (i, j) in enumerate(zip(iu, ju)):
            jac[k, 3 * i:3 * i + 2] += np.asarray(gi[k])
            jac[k, 3 * j:3 * j + 2] += np.asarray(gj[k])
        basis, _ = generators(x)
        return np.asarray(values), jac, jac @ basis

    fields = {}
    keys = crn_keys("paper_alignment", "gauge", seed)
    for kind in ("urban", "stratified", "radial", "constant"):
        cfg.world.kind = kind
        fields[kind] = np.asarray(world.make_world(keys.world_gen, cfg).F_true,
                                  dtype=np.float64)
    shifted = poses.copy()
    shifted[:, :2] += [0.3, 0.4]  # exact 0.5 m rigid translation, all chords interior
    constants = {}
    for label, f in (("zero", np.zeros((grid, grid))), ("positive", fields["constant"])):
        value0, jac0, k0 = instrument(f, poses)
        value1, jac1, k1 = instrument(f, shifted)
        err0 = float(np.linalg.norm(k0) / max(np.linalg.norm(jac0), 1.0))
        err1 = float(np.linalg.norm(k1) / max(np.linalg.norm(jac1), 1.0))
        difference = float(np.max(np.abs(value1 - value0)))
        constants[label] = {
            "field_level_db_per_m": float(f[0, 0]),
            "original_relative_gauge_jacobian": err0,
            "shifted_relative_gauge_jacobian": err1,
            "max_received_power_change_db": difference,
            "passed": bool(max(err0, err1, difference) < 1e-10),
        }

    h = ext / (grid - 1)
    y, x = np.meshgrid(np.linspace(0, ext, grid), np.linspace(0, ext, grid), indexing="ij")
    center = poses[:, :2].mean(axis=0)
    finite = {}
    for kind in ("urban", "stratified", "radial"):
        f = fields[kind]
        _, _, actual_k = instrument(f, poses)
        fy, fx = np.gradient(f, h, h, edge_order=1)
        phi = np.stack([fx, fy, (x - center[0]) * fy - (y - center[1]) * fx], -1)
        phi_n = phi.reshape(-1, 3) / norms
        chord = np.asarray(grad_field(jnp.asarray(f), jnp.asarray(poses[iu, :2]),
                                      jnp.asarray(poses[ju, :2]))).reshape(m, -1)
        predicted_b = chord @ phi_n
        error = actual_k + predicted_b
        bnorm, enorm = np.linalg.norm(predicted_b, 2), np.linalg.norm(error, 2)
        actual_information = actual_k.T @ actual_k
        predicted_information = predicted_b.T @ predicted_b
        gap = float(np.linalg.norm(actual_information - predicted_information, 2))
        bound = float(2 * bnorm * enorm + enorm ** 2)
        finite[kind] = {
            "actual_gauge_jacobian_spectral_norm": float(np.linalg.norm(actual_k, 2)),
            "sampled_prediction_spectral_norm": float(bnorm),
            "Eh_spectral_norm": float(enorm),
            "Eh_relative_to_actual_jacobian": float(enorm / max(np.linalg.norm(actual_k, 2), 1e-300)),
            "gram_discrepancy_spectral_norm": gap,
            "gram_discrepancy_bound": bound,
            "bound_holds": bool(gap <= bound + 1e-10 * max(bound, 1.0)),
            "actual_gauge_information_eigenvalues": np.linalg.eigvalsh(actual_information).tolist(),
            "sampled_gauge_information_eigenvalues": np.linalg.eigvalsh(predicted_information).tolist(),
        }

    relative = relative_checks(poses, rng)
    return {
        "scope": "Fixed interior geometry, implementation diagnostics; no continuum or all-time PE certificate",
        "seed": seed, "grid": grid, "extent_m": ext, "robots": n, "unordered_links": m,
        "gauge_center": center.tolist(), "column_norms": norms.tolist(),
        "orthonormality_error_float64": float(np.linalg.norm(gn.T @ gn - np.eye(3))),
        "rigid_translation_m": [0.3, 0.4],
        "minimum_endpoint_boundary_margin_m": float(min(poses[:, :2].min(), ext - shifted[:, :2].max())),
        "derivative_operator": "numpy.gradient: centered interior, first-order one-sided boundary; spacing extent/(grid-1)",
        "implemented_interpolation": "jax.scipy.ndimage.map_coordinates(order=1, mode=constant, cval=0); automatic endpoint derivative; no custom JVP in coopsim.field",
        "quadrature_samples_per_link": field.K,
        "noise_precision": "Omega = identity on every unordered link",
        "constant_fields": constants,
        "weighted_relative_factors": relative,
        "finite_grid": finite,
        "passed": bool(all(row["passed"] for row in constants.values())
                       and all(row["passed"] for row in relative.values())
                       and all(row["bound_holds"] for row in finite.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--grid", type=int, default=40)
    args = parser.parse_args()
    result = run(seed=args.seed, grid=args.grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Gauge diagnostics passed={result['passed']}")
    for kind, row in result["finite_grid"].items():
        print(f"{kind}: relative_Eh={row['Eh_relative_to_actual_jacobian']:.3e} "
              f"bound_holds={row['bound_holds']}")
    print(f"Saved {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
