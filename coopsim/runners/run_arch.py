"""B7/B9 runner: one full episode per (world, regime, arch, seed), with offline
metrics. Args via env: ARCH, WORLD, REGIME, SEED, STEPS, ORACLE_FIELD, or
SLURM_ARRAY_TASK_ID decoding (seed = task id).
Run: ARCH=A_J python -u -m coopsim.runners.run_arch
"""
import os
import json
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from coopsim import harness
from coopsim.config import load_config, config_dict, config_hash
from coopsim.metrics.gauge import gauge_decompose
from coopsim.metrics.gospa import gospa_series
from coopsim.metrics.identity import identity_metrics


def main():
    arch = os.environ.get("ARCH", "A_J")
    world = os.environ.get("WORLD", "urban")
    regime = os.environ.get("REGIME", "R1")
    seed = int(os.environ.get("SEED", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    steps = int(os.environ.get("STEPS", "0")) or None
    overrides = [s for s in os.environ.get("CFG_SET", "").split(",") if s]
    cfg = load_config(overrides=overrides)
    cfg.world.kind = world
    cfg.targets.regime = regime
    cfg.estimator.oracle_field = os.environ.get("ORACLE_FIELD", "0") == "1"

    from coopsim.paths import SCRATCH
    out_dir = Path(cfg.logging.out_dir or SCRATCH) / "runs"
    print(f"run_arch: {world} {regime} {arch} seed={seed} steps={steps or cfg.time.steps}"
          f" oracleF={cfg.estimator.oracle_field}", flush=True)
    # Save settings before execution so a failed job is also diagnosable. This
    # sidecar complements legacy NPZ cfg_hash values without changing numerics.
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(arch=arch, world=world, regime=regime, seed=seed,
                      steps=steps or cfg.time.steps, cfg_hash=config_hash(cfg),
                      config=config_dict(cfg),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      jax_platforms=os.environ.get("JAX_PLATFORMS"))
    tag = ""
    if arch.startswith("A_BL") and cfg.bilevel.inner != "ge":
        tag = f"_{cfg.bilevel.inner}"
    manifest = out_dir / f"{world}_{regime}_{arch}_s{seed}{tag}.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    path, out = harness.run_episode(cfg, arch, seed, steps=steps, out_dir=out_dir)
    print(f"saved {path}  wall/slot={out['wall_per_slot']*1000:.0f} ms", flush=True)

    T = out["x_est"].shape[0]
    g = gauge_decompose(out["X_true"][:, :, :2], out["x_est"][:, :, :2])
    q = [T // 4, T // 2, 3 * T // 4, T - 1]
    print("common-mode: " + "  ".join(f"t={t}: {g['common'][t]:.3f}" for t in q))
    print("relative:    " + "  ".join(f"t={t}: {g['relative'][t]:.3f}" for t in q))

    # mean-over-robots GOSPA on the fused pictures
    n = out["tracks"].shape[1]
    gos = np.zeros((n, T))
    idf = []
    for r in range(n):
        s = gospa_series(out["tau"], out["alive"], out["tracks"][:, r, :, :2],
                         out["tracks_mask"][:, r])
        gos[r] = s["gospa"]
        idm = identity_metrics(out["tau"], out["alive"],
                               out["tracks"][:, r, :, :2],
                               out["tracks"][:, r, :, 4].astype(int),
                               out["tracks_mask"][:, r])
        idf.append(idm["idf1"])
    gm = gos.mean(0)
    print("GOSPA(mean over robots): "
          + "  ".join(f"t={t}: {gm[t]:.2f}" for t in q))
    print(f"IDF1 mean={np.mean(idf):.3f}")
    print(f"delivery={out['succ'].mean():.3f} collisions_proxy="
          f"{(out['sched'].sum(axis=(1,2)) > 0).mean():.2f}")
    print("DONE run_arch")


if __name__ == "__main__":
    main()
