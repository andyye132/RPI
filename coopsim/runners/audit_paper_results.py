"""Audit paper endpoints from explicitly selected, hashed episode files.

Run on a compute node, e.g. python -m coopsim.runners.audit_paper_results
--output results/paper_alignment_results.json. No JAX import or simulation.
The built-in cells preserve the historical seed cohort AND separately apply
the attachment's exclusion of tuning seeds 0--5. Missing data are findings,
never silently replaced by a smaller sample or a different experiment store.

An optional --manifest JSON replaces the built-in cells. Each cell supplies
id, runs_a, runs_b, arm_a (test), arm_b (control), seeds (explicit integer list),
expected_n, expected_steps, expected_robots, dt, and optional suffix_a/suffix_b.
File names follow urban_R1_{arm}_s{seed}{suffix}.npz. Relative paths in the
manifest are resolved relative to its parent, not the caller's directory.
Statistics are paired within seed; all metrics use float64 and seed SD (ddof=1).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t as student_t


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coopsim.metrics.gauge import gauge_decompose


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_hash(config):
    """Mirror coopsim.config.config_hash without importing JAX on an audit."""
    def flatten(mapping, prefix=""):
        items = []
        for key, value in sorted(mapping.items()):
            if isinstance(value, dict):
                items.extend(flatten(value, f"{prefix}{key}."))
            else:
                items.append(f"{prefix}{key}={value}")
        return items
    return hashlib.sha256("|".join(flatten(config)).encode()).hexdigest()[:12]


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None,
                "df": None, "ci95": None, "t": None, "p_two_sided": None}
    mean = float(values.mean())
    result = dict(n=n, mean=mean, sd=None, se=None, df=n - 1,
                  ci95=None, t=None, p_two_sided=None)
    if n < 2:
        return result
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    result.update(sd=sd, se=float(se), t_critical95=critical,
                  ci95=[mean - critical * se, mean + critical * se])
    if se > 0:
        tvalue = mean / se
        result.update(t=float(tvalue),
                      p_two_sided=float(2 * student_t.sf(abs(tvalue), n - 1)))
    else:
        result["p_two_sided"] = 1.0 if mean == 0 else 0.0
    return result


def default_cells(scratch):
    v4 = str(scratch / "netcomm_v4" / "runs")
    hyg = str(scratch / "netcomm_hygrepro" / "runs")
    common = dict(runs_a=v4, runs_b=v4, arm_a="A_J", arm_b="A0",
                  expected_steps=3000, expected_robots=8, dt=0.1)
    cells = [
        dict(common, id="supplied_drift_historical", seeds=list(range(46)),
             expected_n=46, allow_tuning_seeds=True,
             provenance="Historical supplied-map n46; contains excluded tuning seeds."),
        dict(common, id="supplied_drift_paper_exclusion", seeds=list(range(6, 46)),
             expected_n=46,
             provenance="Same surviving data, excludes 0--5; paper n46 cannot hold."),
        dict(id="supplied_hy_historical", runs_a=hyg, runs_b=hyg,
             arm_a="A_HY", arm_b="A0", seeds=list(range(8, 20)), expected_n=12,
             expected_steps=3000, expected_robots=8, dt=0.1,
             provenance="Job4662546, pre-CRN-fix reproduction; table -0.813 is terminal."),
    ]
    local = str(REPO / "scratch" / "netcomm_v3" / "runs")
    for name, arm, control, stop in (("unsaturated_scalar", "A_BL", "A0", 32),
                                     ("unsaturated_supplied_joint", "A_J", "A0", 16),
                                     ("supplied_hy_joint", "A_HY", "A_J", 32)):
        for lo in (0, 6):
            cells.append(dict(id=f"{name}_{'historical' if lo == 0 else 'heldout'}",
                              runs_a=local, runs_b=local, arm_a=arm, arm_b=control,
                              seeds=list(range(lo, stop)), expected_n=stop - lo,
                              expected_steps=3000, expected_robots=8, dt=0.1,
                              provenance="Repo-local surviving store; mixed settings must be checked using config hashes."))
    return cells


def archived_m1(files, label, seed_lo, seed_hi, expected_steps):
    """Recompute archived M1 results from its per-seed 0.001 m telemetry.

    M1 wrote a point each 100 slots plus the final slot, NOT every pose. Growth
    and final-third summaries here are explicitly of the logged sample grid.
    Duplicate (arm,seed,t) observations must agree; no result-based selection.
    """
    rows, sources, metadata = {}, [], []
    for path in files:
        sources.append(dict(file=str(path.resolve()), sha256=sha256(path)))
        arm = seed = None
        for line in path.read_text().splitlines():
            if line.startswith(("CFG_SET", "M1 v3 gate")):
                metadata.append(line)
            match = re.match(r"\[seed (\d+)\] (\w+)", line)
            if match:
                seed, arm = int(match[1]), match[2]
            match = re.match(r"\s+t=\s*(\d+) cm=\s*([\d.eE+-]+) rel=\s*([\d.eE+-]+)", line)
            if match and arm is not None:
                slot, cm, rel = int(match[1]), float(match[2]), float(match[3])
                row = rows.setdefault((arm, seed), {})
                if slot in row and row[slot] != (cm, rel):
                    raise ValueError(f"conflicting archived values: {path}, {arm}, seed{seed}, slot{slot}")
                row[slot] = (cm, rel)
    seeds = list(range(seed_lo, seed_hi))
    findings = []
    metrics = {a: {} for a in ("A0", "estF")}
    for arm in metrics:
        for seed in seeds:
            row = rows.get((arm, seed), {})
            if expected_steps - 1 not in row:
                findings.append(f"{arm} seed{seed} missing terminal slot")
                continue
            slots = np.array(sorted(row))
            values = np.array([row[t][0] for t in slots])
            fit = slots > 0.25 * (expected_steps - 1)
            design = np.column_stack([np.ones(fit.sum()), np.sqrt(slots[fit] * 0.1)])
            full_design = np.column_stack([np.ones(len(slots)), np.sqrt(slots * 0.1)])
            metrics[arm][seed] = dict(common_terminal=row[expected_steps - 1][0],
                                     relative_terminal=row[expected_steps - 1][1],
                                     common_logged_final_third=float(values[slots >= 2 * expected_steps // 3].mean()),
                                     common_logged_growth=float(np.linalg.lstsq(design, values[fit], rcond=None)[0][1]),
                                     common_logged_growth_full_horizon=float(np.linalg.lstsq(full_design, values, rcond=None)[0][1]))
    paired = sorted(metrics["A0"].keys() & metrics["estF"].keys())
    if paired != seeds:
        findings.append(f"paired seeds differ from declared cohort: expected {seeds}, got {paired}")
    summaries = {}
    for name in ("common_terminal", "relative_terminal", "common_logged_final_third", "common_logged_growth",
                 "common_logged_growth_full_horizon"):
        a = np.array([metrics["estF"][s][name] for s in paired])
        b = np.array([metrics["A0"][s][name] for s in paired])
        summaries[name] = dict(a=stats(a), b=stats(b), paired_a_minus_b=stats(a - b))
    return dict(id=label, evidence="archived per-seed logs rounded to 0.001m; original full pose histories unavailable",
                sources=sources, configuration_lines=sorted(set(metadata)),
                paired_seeds=paired, expected_n=len(seeds), findings=findings,
                summaries=summaries, per_seed={a: list(dict(seed=s, **r) for s, r in values.items())
                                              for a, values in metrics.items()})


def default_archived():
    logs = REPO / "results" / "logs"
    n8 = [logs / name for name in ("v3_tv2_4162545.out", "v3_tv2confirm_4166154.out",
                                   "v3_tv2_x40_4188699.out", "v3_tv2_x80_4211290.out")]
    n16 = [logs / name for name in ("v3_n16conf_o8_4267180.out", "v3_n16conf_o24_4267181.out",
                                    "v3_n16conf_o40_4267182.out")]
    extension = [logs / "v3_n16long_4275522.out"]
    return [archived_m1(n8, "online_n8_historical", 0, 80, 3000),
            archived_m1(n8, "online_n8_heldout", 6, 80, 3000),
            archived_m1(n16, "online_n16_confirmation", 8, 56, 3000),
            archived_m1(extension, "online_n16_extension", 8, 16, 10000)]


def load_episode(path, cell, arm, seed):
    # These are user-owned local experiment artifacts, not downloaded pickle data.
    with np.load(path, allow_pickle=True) as data:
        true = np.asarray(data["X_true"], dtype=np.float64)
        est = np.asarray(data["x_est"], dtype=np.float64)
        expected = (cell["expected_steps"], cell["expected_robots"], 3)
        if true.shape != expected or est.shape != expected:
            raise ValueError(f"shape mismatch: true={true.shape}, est={est.shape}, expected={expected}")
        if not np.isfinite(true).all() or not np.isfinite(est).all():
            raise ValueError("nonfinite poses")
        metadata = {}
        for key in ("arch", "seed", "world", "regime", "cfg_hash", "wall_per_slot"):
            if key in data:
                metadata[key] = np.asarray(data[key]).item()
        if "arch" in metadata and metadata["arch"] != arm:
            raise ValueError(f"arch mismatch: {metadata['arch']} != {arm}")
        if "seed" in metadata and int(metadata["seed"]) != seed:
            raise ValueError(f"seed mismatch: {metadata['seed']} != {seed}")
        series = gauge_decompose(true[:, :, :2], est[:, :, :2])
        T = len(true)
        ts = np.arange(T, dtype=np.float64) * cell["dt"]
        fit = ts > 0.25 * ts[-1]
        design = np.column_stack([np.ones(fit.sum()), np.sqrt(ts[fit])])
        metrics = {}
        for key in ("common", "relative", "ate"):
            values = series[key]
            metrics[key + "_final_third"] = float(values[2 * T // 3:].mean())
            metrics[key + "_terminal"] = float(values[-1])
            metrics[key + "_growth"] = float(np.linalg.lstsq(design, values[fit], rcond=None)[0][1])
        # Byte equality checks the paired truth; does not assert historical CRN
        # correctness for odometry/radio streams absent from old output files.
        truth_hash = hashlib.sha256(true.tobytes()).hexdigest()
        metrics["wall_per_slot"] = metadata.get("wall_per_slot")
        full_config = None
        config_source = None
        if "config_json" in data:
            full_config = json.loads(str(np.asarray(data["config_json"]).item()))
            config_source = "embedded config_json"
        sidecar = path.with_suffix(".json")
        sidecar_source = None
        if sidecar.is_file():
            provenance = json.loads(sidecar.read_text())
            for name, expected_value in (("arch", arm), ("seed", seed),
                                         ("steps", cell["expected_steps"])):
                if provenance.get(name) != expected_value:
                    raise ValueError(f"sidecar {name} mismatch: {provenance.get(name)} != {expected_value}")
            for name in ("world", "regime", "cfg_hash"):
                if name in metadata and provenance.get(name) != metadata[name]:
                    raise ValueError(f"sidecar {name} differs from NPZ metadata")
            sidecar_config = provenance.get("config")
            if not isinstance(sidecar_config, dict):
                raise ValueError("sidecar is missing full config object")
            if full_config is not None and full_config != sidecar_config:
                raise ValueError("embedded and sidecar configurations disagree")
            full_config = sidecar_config
            config_source = "sidecar config"
            sidecar_source = dict(file=str(sidecar.resolve()), sha256=sha256(sidecar),
                                  slurm_job_id=provenance.get("slurm_job_id"),
                                  jax_platforms=provenance.get("jax_platforms"))
        if full_config is not None:
            computed_hash = configuration_hash(full_config)
            if metadata.get("cfg_hash") != computed_hash:
                raise ValueError(f"full config hash {computed_hash} differs from NPZ cfg_hash {metadata.get('cfg_hash')}")
            if full_config.get("shapes", {}).get("n_robots") != cell["expected_robots"]:
                raise ValueError("full config robot count differs from declared experiment")
            if full_config.get("time", {}).get("dt") != cell["dt"]:
                raise ValueError("full config dt differs from declared endpoint time scale")
        return dict(file=str(path.resolve()), sha256=sha256(path), seed=seed,
                    metadata=metadata, config=full_config, config_source=config_source,
                    sidecar=sidecar_source, truth_sha256=truth_hash,
                    metrics=metrics)


def audit_cell(cell, cache):
    findings = []
    rejected_episodes = []
    requested = cell["seeds"]
    if len(requested) != len(set(requested)):
        raise ValueError(f"{cell['id']}: duplicate declared seeds")
    if len(requested) != cell["expected_n"]:
        findings.append(f"declared cohort has {len(requested)} seeds, paper expects {cell['expected_n']}")
    tuning = sorted(s for s in requested if s < 6)
    if tuning:
        findings.append(f"includes tuning seeds prohibited by attached paper: {tuning}")
    per_arm = {}
    for side in ("a", "b"):
        arm = cell["arm_" + side]
        rows = {}
        root = Path(cell["runs_" + side])
        prefix = cell.get("prefix", "urban_R1")
        suffix = cell.get("suffix_" + side, "")
        for seed in requested:
            path = root / f"{prefix}_{arm}_s{seed}{suffix}.npz"
            if not path.is_file():
                continue
            try:
                key = (str(path.resolve()), str(path.with_suffix(".json").absolute()), arm, seed,
                       cell["expected_steps"], cell["expected_robots"], cell["dt"])
                if key not in cache:
                    cache[key] = load_episode(path, cell, arm, seed)
                rows[seed] = cache[key]
            except Exception as error:
                findings.append(f"invalid {path}: {error}")
                rejected_episodes.append(dict(file=str(path.resolve()), sha256=sha256(path),
                                              arm=arm, seed=seed, reason=str(error)))
        missing = sorted(set(requested) - rows.keys())
        if missing:
            findings.append(f"{arm} missing declared seeds: {missing}")
        per_arm[side] = rows
    paired = sorted(per_arm["a"].keys() & per_arm["b"].keys())
    invalid_truth = [s for s in paired if per_arm["a"][s]["truth_sha256"] != per_arm["b"][s]["truth_sha256"]]
    if invalid_truth:
        findings.append(f"paired truths differ for seeds {invalid_truth}; excluded from paired statistics")
        paired = [s for s in paired if s not in invalid_truth]
    if len(paired) != cell["expected_n"]:
        findings.append(f"paired n={len(paired)}, expected n={cell['expected_n']}")
    summaries = {}
    metrics = ("common_growth", "common_final_third", "common_terminal",
               "relative_final_third", "relative_terminal", "ate_final_third")
    for metric in metrics:
        a = np.array([per_arm["a"][s]["metrics"][metric] for s in paired])
        b = np.array([per_arm["b"][s]["metrics"][metric] for s in paired])
        summaries[metric] = {"a": stats(a), "b": stats(b), "paired_a_minus_b": stats(a - b)}
        if len(a) >= 2:
            # This reproduces diag_drift's old mislabeled ± convention explicitly.
            for side in ("a", "b"):
                summaries[metric][side]["legacy_2_093_se_halfwidth"] = 2.093 * summaries[metric][side]["se"]
    hashes = {side: sorted({str(row["metadata"].get("cfg_hash", "MISSING")) for row in rows.values()})
              for side, rows in per_arm.items()}
    for side, vals in hashes.items():
        if len(vals) > 1:
            findings.append(f"arm {side} contains multiple config hashes: {vals}")
    if any(row["config"] is None for rows in per_arm.values() for row in rows.values()):
        findings.append("legacy files store only cfg_hash, not full configuration; hash alone cannot reconstruct settings")
    return dict(spec=cell, paired_seeds=paired, paired_n=len(paired),
                count_matches=(len(paired) == cell["expected_n"]),
                tuning_seeds=tuning, findings=findings, summaries=summaries,
                statistics_scope="all surviving pairs with identical truth, including configuration/provenance findings; invalid episodes and truth-mismatched pairs excluded",
                rejected_episodes=rejected_episodes, truth_mismatched_seeds=invalid_truth,
                config_hashes=hashes,
                episodes={side: list(rows.values()) for side, rows in per_arm.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path.home() / "scratch")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "results" / "paper_alignment_results.json")
    parser.add_argument("--strict", action="store_true", help="nonzero status on any reported finding")
    parser.add_argument("--skip-archived", action="store_true", help="skip built-in historical M1 log reconstruction")
    args = parser.parse_args()
    cells = default_cells(args.scratch)
    if args.manifest:
        loaded = json.loads(args.manifest.read_text())
        cells = loaded["cells"] if isinstance(loaded, dict) else loaded
        for cell in cells:
            for key in ("runs_a", "runs_b"):
                path = Path(cell[key])
                if not path.is_absolute():
                    cell[key] = str((args.manifest.parent / path).resolve())
    cache = {}
    results = [audit_cell(cell, cache) for cell in cells]
    source_paths = [Path(__file__), REPO / "coopsim" / "metrics" / "gauge.py",
                    REPO / "coopsim" / "harness.py", REPO / "configs" / "config.yaml"]
    output = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  statistics="seed SD ddof=1; SE=SD/sqrt(n); two-sided t intervals df=n-1; paired test-minus-control",
                  endpoints="growth: intercept OLS on sqrt(t), t>0.25*t_last; final-third: indices 2*T//3 onward; terminal: final slot",
                  sources=[dict(file=str(p.resolve()), sha256=sha256(p)) for p in source_paths],
                  cells=results,
                  archived=[] if args.skip_archived else default_archived())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for result in results:
        print(f"{result['spec']['id']}: n={result['paired_n']}, expected={result['spec']['expected_n']}")
        for finding in result["findings"]:
            print(f"  FLAG: {finding}")
        for metric in ("common_growth", "common_final_third", "common_terminal"):
            values = result["summaries"][metric]
            print(f"  {metric}: " + json.dumps(values))
    print(f"WROTE {args.output}")
    if args.strict and any(result["findings"] for result in results + output["archived"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
