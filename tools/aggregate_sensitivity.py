"""Aggregate and validate the fixed-data solver-sensitivity study on the Experiment B inputs.

Reconstruction of the archived results/acc2027_strengthening/solver_sensitivity/aggregate_sensitivity.py,
which is not part of the public tree. README semantics: OUT is this file's own directory (it holds the
per-seed outputs of coopsim.runners.run_interface_solver_sensitivity); ROOT is the repository root three
levels up (results/<group>/solver_sensitivity); SOURCE is the Experiment B directory that the sensitivity
runner was pointed at, defaulting to the archived B directory under ROOT. Override SOURCE with --source
(or AETHER_B_SOURCE) when the study was run against a freshly generated B directory.

Writes validation.json, summary.json, report.md and the solver_error_sensitivity /
solver_numerical_sensitivity PDF/PNG pairs into OUT. Exits non-zero if any validation check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent


def _find_root(start):
    """Repository root: the nearest ancestor containing coopsim/ and configs/ (README layout puts this
    script at results/<group>/solver_sensitivity/, i.e. three levels down; the tools/ copy is one level down)."""
    for cand in (start, *start.parents):
        if (cand / "coopsim").is_dir() and (cand / "configs").is_dir():
            return cand
    return start


ROOT = _find_root(OUT)
DEFAULT_SOURCE = ROOT / "results" / "acc2027_revision" / "experiment_b"
SEEDS = list(range(200, 232))
SCENARIOS = ("correct", "registration", "attenuation", "combined")
VARIANTS = ("original25", "tight25", "tight100", "tight300", "lbfgs1000")
NUMERIC = ("objective", "objective_relative_improvement", "gradient_inf", "undamped_gn_step_inf_m",
           "common_rmse_final_third", "relative_rmse_final_third", "terminal_common_error",
           "terminal_common_rms_uncertainty", "terminal_relative_rms_uncertainty", "common_nees",
           "relative_nees", "trajectory_rms_change_m", "max_coordinate_change_m",
           "relative_terminal_covariance_change")
BOOLEAN = ("first_order_converged", "common_coverage95", "relative_coverage95")
PAIRED = ("common_rmse_final_third", "relative_rmse_final_third", "terminal_common_error",
          "terminal_common_rms_uncertainty", "common_nees")
RECON_TOL = 1e-9


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half],
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def proportion(flags):
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = stats.binomtest(k, n).proportion_ci(method="wilson")
    return {"count": k, "n": n, "fraction": k / n, "ci95": [float(lo), float(hi)]}


def load(source, seeds):
    rows, missing = {}, []
    for seed in seeds:
        j, z = OUT / f"seed_{seed}.json", OUT / f"seed_{seed}.npz"
        sj, sz = source / f"seed_{seed}.json", source / f"seed_{seed}.npz"
        if not (j.exists() and z.exists() and sj.exists() and sz.exists()):
            missing.append(seed)
            continue
        rows[seed] = {"json": json.loads(j.read_text()), "npz": z, "src_json": json.loads(sj.read_text()), "src_npz": sz}
    return rows, missing


def validate(rows, source, reconstruct=True):
    sys.path.insert(0, str(ROOT))
    from coopsim.runners import run_interface_solver_sensitivity as sens  # noqa: E402
    checks = {"seeds_expected": len(SEEDS), "seeds_found": len(rows), "problems": 0, "checkpoints": 0,
              "input_hash_mismatches": [], "weight_hash_mismatches": [], "replay_failures": [],
              "message_joint_failures": [], "original_estimate_mismatches": [], "trace_length_mismatches": [],
              "reconstruction_max_metric_difference": 0.0, "reconstruction_max_covariance_difference": 0.0,
              "reconstruction_failures": [], "lbfgs_scipy_failures": [], "source": str(source)}
    for seed, r in rows.items():
        js = r["json"]
        with np.load(r["src_npz"]) as z:
            src = {k: z[k] for k in z.files}
        with np.load(r["npz"]) as z:
            out = {k: z[k] for k in z.files}
        src_hashes = r["src_json"]["input_hashes"]
        for key, checksum in js["input_hashes"].items():
            if checksum != src_hashes.get(key) or sha(src[key]) != checksum:
                checks["input_hash_mismatches"].append([seed, key])
        for sc in SCENARIOS:
            checks["problems"] += 1
            s = js["scenarios"][sc]
            wh = sha(src[f"{sc}_effective_weights"])
            src_methods = r["src_json"]["scenarios"][sc]["methods"]
            if s["effective_weights_hash"] != wh or any(m["effective_weights_hash"] != wh for m in src_methods.values()):
                checks["weight_hash_mismatches"].append([seed, sc])
            if max(s["original_replay"].values()) > 1e-8:
                checks["replay_failures"].append([seed, sc, s["original_replay"]])
            if not np.array_equal(out[f"{sc}_original25_estimate"], src[f"{sc}_message_estimate"]):
                checks["original_estimate_mismatches"].append([seed, sc])
            if len(s["tight_lm_trace"]) != s["methods"]["tight300"]["optimizer"]["iterations"]:
                checks["trace_length_mismatches"].append([seed, sc])
            if not s["methods"]["lbfgs1000"]["optimizer"].get("scipy_success", False):
                checks["lbfgs_scipy_failures"].append([seed, sc, s["methods"]["lbfgs1000"]["optimizer"].get("message")])
            obj = sens.Objective(src, sc) if reconstruct else None
            oldx = src[f"{sc}_message_estimate"].ravel()
            oldcov = src[f"{sc}_message_terminal_covariance"]
            for v in VARIANTS:
                checks["checkpoints"] += 1
                row = s["methods"][v]
                if max(row["message_joint_equality"].values()) > 1e-9:
                    checks["message_joint_failures"].append([seed, sc, v])
                if not reconstruct:
                    continue
                x = out[f"{sc}_{v}_estimate"].ravel()
                try:
                    rec, cov, _ = sens.metrics(obj, x, oldx, oldcov)
                except RuntimeError as exc:
                    checks["reconstruction_failures"].append([seed, sc, v, str(exc)])
                    continue
                for key in NUMERIC:
                    d = abs(rec[key] - row[key]) / max(1.0, abs(row[key]))
                    checks["reconstruction_max_metric_difference"] = max(checks["reconstruction_max_metric_difference"], float(d))
                    if d > RECON_TOL:
                        checks["reconstruction_failures"].append([seed, sc, v, key, rec[key], row[key]])
                for key in BOOLEAN:
                    if bool(rec[key]) != bool(row[key]):
                        checks["reconstruction_failures"].append([seed, sc, v, key, rec[key], row[key]])
                dc = float(np.max(np.abs(cov - out[f"{sc}_{v}_terminal_covariance"])) / max(1.0, np.max(np.abs(cov))))
                checks["reconstruction_max_covariance_difference"] = max(checks["reconstruction_max_covariance_difference"], dc)
                if dc > RECON_TOL:
                    checks["reconstruction_failures"].append([seed, sc, v, "terminal_covariance", dc])
    checks["passed"] = bool(
        checks["seeds_found"] == checks["seeds_expected"] and checks["checkpoints"] == 5 * 4 * checks["seeds_expected"]
        and not checks["input_hash_mismatches"] and not checks["weight_hash_mismatches"] and not checks["replay_failures"]
        and not checks["message_joint_failures"] and not checks["original_estimate_mismatches"]
        and not checks["trace_length_mismatches"] and not checks["reconstruction_failures"])
    return checks


def summarize(rows):
    seeds = sorted(rows)
    summary = {"seeds": seeds, "n_seeds": len(seeds), "scenarios": {}, "headline": {}}
    max_change_mm, converged_total, grad_all, cov_change_all = 0.0, 0, [], []
    for sc in SCENARIOS:
        block = {"methods": {}, "paired_vs_original25": {}, "scalar": {}, "probes": {}}
        per = {v: [rows[s]["json"]["scenarios"][sc]["methods"][v] for s in seeds] for v in VARIANTS}
        for v in VARIANTS:
            m = {k: ci([row[k] for row in per[v]]) for k in NUMERIC}
            m.update({k: proportion([bool(row[k]) for row in per[v]]) for k in BOOLEAN})
            opt = [row["optimizer"] for row in per[v]]
            m["optimizer"] = {"iterations": ci([o["iterations"] for o in opt])}
            if v.startswith("tight"):
                m["optimizer"]["stop_reasons"] = dict(Counter(o["stop_reason"] for o in opt))
            if v == "lbfgs1000":
                m["optimizer"]["scipy_success_count"] = int(sum(bool(o.get("scipy_success")) for o in opt))
                m["optimizer"]["evaluations"] = ci([o["evaluations"] for o in opt])
            block["methods"][v] = m
            converged_total += m["first_order_converged"]["count"]
            grad_all += [row["gradient_inf"] for row in per[v]]
            if v != "original25":
                cov_change_all += [row["relative_terminal_covariance_change"] for row in per[v]]
        for v in VARIANTS[1:]:
            d = {}
            for k in PAIRED:
                diff = [b[k] - a[k] for a, b in zip(per["original25"], per[v])]
                d[k] = ci(diff)
                if k.endswith("_m") or "rmse" in k or "error" in k or "uncertainty" in k:
                    d[k]["mean_mm"] = d[k]["mean"] * 1e3
            block["paired_vs_original25"][v] = d
            max_change_mm = max(max_change_mm, abs(d["common_rmse_final_third"]["mean"]) * 1e3)
        scal = [rows[s]["json"]["scenarios"][sc]["scalar"] for s in seeds]
        block["scalar"] = {k: ci([row[k] for row in scal]) for k in ("common_rmse_final_third", "relative_rmse_final_third", "terminal_common_error")}
        for v in ("original25", "tight300", "lbfgs1000"):
            pr = [rows[s]["json"]["scenarios"][sc]["directional_probes"][v] for s in seeds]
            smallest = [p["smallest_objective_change"] for p in pr]
            block["probes"][v] = {"smallest_objective_change": ci(smallest),
                                  "descent_direction_found_count": int(sum(x < 0 for x in smallest)),
                                  "min_objective_change_by_radius_mean": np.mean([p["min_objective_change_by_radius"] for p in pr], axis=0).tolist(),
                                  "radii_m": pr[0]["radii_m"]}
        block["mean_common_rmse_final_third"] = {v: block["methods"][v]["common_rmse_final_third"]["mean"] for v in VARIANTS}
        block["scenario_mean_change_mm_vs_original25"] = {v: block["paired_vs_original25"][v]["common_rmse_final_third"]["mean"] * 1e3 for v in VARIANTS[1:]}
        summary["scenarios"][sc] = block
    summary["headline"] = {
        "checkpoints": 5 * 4 * len(seeds), "problems": 4 * len(seeds),
        "correct_map_mean_common_rmse_m": {v: summary["scenarios"]["correct"]["mean_common_rmse_final_third"][v] for v in ("original25", "tight300", "lbfgs1000")},
        "max_abs_scenario_mean_change_mm": max_change_mm,
        "first_order_converged_checkpoints": converged_total,
        "gradient_inf": {"median": float(np.median(grad_all)), "max": float(np.max(grad_all)), "threshold": 1e-6},
        "relative_terminal_covariance_change_refined": {"mean": float(np.mean(cov_change_all)), "median": float(np.median(cov_change_all)), "max": float(np.max(cov_change_all))},
    }
    return summary


def report(summary, checks):
    h = summary["headline"]
    lines = ["# Solver sensitivity on the Experiment B inputs", "",
             f"Seeds: {summary['n_seeds']} ({summary['seeds'][0]}-{summary['seeds'][-1]}); problems: {h['problems']}; checkpoints: {h['checkpoints']}; validation passed: {checks['passed']}.", "",
             "## Mean common-mode RMSE, final third (m)", "",
             "| scenario | original25 | tight25 | tight100 | tight300 | lbfgs1000 | max |change| vs original (mm) |", "|---|---|---|---|---|---|---|"]
    for sc in SCENARIOS:
        b = summary["scenarios"][sc]
        m = b["mean_common_rmse_final_third"]
        ch = max(abs(x) for x in b["scenario_mean_change_mm_vs_original25"].values())
        lines.append(f"| {sc} | " + " | ".join(f"{m[v]:.5f}" for v in VARIANTS) + f" | {ch:.3f} |")
    lines += ["", "## Paired change vs original25, common RMSE final third (mm, mean [95% CI])", "",
              "| scenario | " + " | ".join(VARIANTS[1:]) + " |", "|---|---|---|---|---|"]
    for sc in SCENARIOS:
        p = summary["scenarios"][sc]["paired_vs_original25"]
        cells = []
        for v in VARIANTS[1:]:
            d = p[v]["common_rmse_final_third"]
            cells.append(f"{d['mean']*1e3:+.3f} [{d['ci95'][0]*1e3:+.3f}, {d['ci95'][1]*1e3:+.3f}]")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    lines += ["", "## Numerical status", "",
              f"- Max |scenario-mean change| across scenarios and variants: {h['max_abs_scenario_mean_change_mm']:.3f} mm.",
              f"- First-order converged checkpoints (gradient inf-norm <= 1e-6): {h['first_order_converged_checkpoints']} of {h['checkpoints']}.",
              f"- Gradient inf-norm over all checkpoints: median {h['gradient_inf']['median']:.3g}, max {h['gradient_inf']['max']:.3g}.",
              f"- Relative terminal-covariance change of refined fits vs original: mean {h['relative_terminal_covariance_change_refined']['mean']:.3g}, max {h['relative_terminal_covariance_change_refined']['max']:.3g}.",
              f"- Independent reconstruction of metrics/covariances from saved estimates: max metric difference {checks['reconstruction_max_metric_difference']:.2e}, max covariance difference {checks['reconstruction_max_covariance_difference']:.2e}.",
              f"- Original replay mismatches: {len(checks['replay_failures'])}; input-hash mismatches: {len(checks['input_hash_mismatches'])}; weight-hash mismatches: {len(checks['weight_hash_mismatches'])}; message/joint failures: {len(checks['message_joint_failures'])}; L-BFGS-B scipy non-success: {len(checks['lbfgs_scipy_failures'])}.", ""]
    for sc in SCENARIOS:
        t = summary["scenarios"][sc]["methods"]["tight300"]["optimizer"]["stop_reasons"]
        lines.append(f"- {sc}: tight LM stop reasons {t}; probes with a descent direction at original25 / tight300 / lbfgs1000: "
                     + " / ".join(str(summary["scenarios"][sc]["probes"][v]["descent_direction_found_count"]) for v in ("original25", "tight300", "lbfgs1000")) + f" of {summary['n_seeds']}.")
    return "\n".join(lines) + "\n"


def figures(summary, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
    colors = {"original25": "#2a78d6", "tight25": "#eb6834", "tight100": "#1baf7a", "tight300": "#eda100", "lbfgs1000": "#e87ba4"}
    labels = {"original25": "original LM (25)", "tight25": "strict LM (25)", "tight100": "strict LM (100)",
              "tight300": "strict LM (300)", "lbfgs1000": "L-BFGS-B (1000)"}
    seeds = summary["seeds"]
    x = np.arange(len(SCENARIOS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    show = ("original25", "tight300", "lbfgs1000")
    for i, v in enumerate(show):
        m = [summary["scenarios"][sc]["methods"][v]["common_rmse_final_third"] for sc in SCENARIOS]
        mean = np.array([r["mean"] for r in m]); lo = np.array([r["ci95"][0] for r in m]); hi = np.array([r["ci95"][1] for r in m])
        off = (i - 1) * 0.22
        ax.errorbar(x + off, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=5, lw=1.2, capsize=3, color=colors[v], label=labels[v])
    sm = [summary["scenarios"][sc]["scalar"]["common_rmse_final_third"]["mean"] for sc in SCENARIOS]
    ax.plot(x, sm, ls="--", lw=1.2, color="#898781", marker="_", ms=14, label="scalar weights (reference)")
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS); ax.set_ylabel("common-mode RMSE, final third (m)")
    ax.set_title("Message arm by solver budget (mean, 95% CI over seeds)"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for i, v in enumerate(VARIANTS[1:]):
        d = [summary["scenarios"][sc]["paired_vs_original25"][v]["common_rmse_final_third"] for sc in SCENARIOS]
        mean = np.array([r["mean"] for r in d]) * 1e3; lo = np.array([r["ci95"][0] for r in d]) * 1e3; hi = np.array([r["ci95"][1] for r in d]) * 1e3
        off = (i - 1.5) * 0.18
        ax.errorbar(x + off, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=5, lw=1.2, capsize=3, color=colors[v], label=labels[v])
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS); ax.set_ylabel("paired change vs original LM (mm)")
    ax.set_title("Paired change vs original LM (mm)"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"solver_error_sensitivity.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    def collect(key, variants):
        return [[rows[s]["json"]["scenarios"][sc]["methods"][v][key] for s in seeds for sc in SCENARIOS] for v in variants]
    panels = [("gradient_inf", VARIANTS, "gradient inf-norm at checkpoint", 1e-6, "first-order threshold 1e-6"),
              ("relative_terminal_covariance_change", VARIANTS[1:], "relative terminal-covariance change vs original", None, None),
              ("undamped_gn_step_inf_m", VARIANTS, "undamped Gauss-Newton step, inf-norm (m)", None, None)]
    for ax, (key, variants, ylabel, thr, thr_label) in zip(axes, panels):
        data = collect(key, variants)
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "ms": 3, "alpha": 0.5}, medianprops={"color": "#0b0b0b", "lw": 1.2})
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(colors[v]); patch.set_alpha(0.35); patch.set_edgecolor(colors[v])
        ax.set_xticks(range(1, len(variants) + 1)); ax.set_xticklabels([labels[v] for v in variants], rotation=20, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel(ylabel); ax.grid(axis="y"); ax.set_axisbelow(True)
        if thr is not None:
            ax.axhline(thr, color="#d03b3b", lw=1, ls="--"); ax.text(0.55, thr * 1.4, thr_label, color="#d03b3b", fontsize=8)
    axes[0].set_title(f"{summary['headline']['problems']} problems per variant")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"solver_numerical_sensitivity.{ext}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=Path(os.environ.get("AETHER_B_SOURCE", DEFAULT_SOURCE)))
    p.add_argument("--seeds", type=str, default="200-231", help="inclusive range, e.g. 200-231")
    p.add_argument("--no-reconstruct", action="store_true", help="skip the independent metric/covariance reconstruction")
    a = p.parse_args()
    lo, hi = (int(t) for t in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    rows, missing = load(a.source, seeds)
    if missing:
        raise SystemExit(f"Missing per-seed outputs (OUT={OUT}, SOURCE={a.source}) for seeds {missing}; all seeds are required.")
    checks = validate(rows, a.source, reconstruct=not a.no_reconstruct)
    checks["seeds_expected"] = len(seeds)
    checks["passed"] = bool(checks["passed"] and checks["checkpoints"] == 5 * 4 * len(seeds))
    (OUT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    summary = summarize(rows)
    summary["validation_passed"] = checks["passed"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = report(summary, checks)
    (OUT / "report.md").write_text(text)
    figures(summary, rows)
    print(text)
    print(f"Wrote validation.json, summary.json, report.md and figures to {OUT}")
    if not checks["passed"]:
        raise SystemExit("Solver-sensitivity validation failed; see validation.json")


if __name__ == "__main__":
    main()
