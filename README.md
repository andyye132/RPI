# AETHER: the coordination variable decides

Gauge-aware cooperative localization and multi-target tracking for robot teams
on degraded radio links. The packet stream is treated as a sensor of the
environment rather than as an erasure channel. This repository holds the
simulator (`coopsim`) and the experiment runners behind the ACC 2027 paper
*Gauge Information at the Radio-to-Pose Interface in Cooperative Localization*.
Every experiment in the paper runs from this tree alone; the archived raw
evidence is distributed separately (see [Archived evidence](#archived-evidence)).

A team using only invariant relative factors cannot obtain direct information
about its global frame from those factors. Radio observables are different:
received power depends on what the signal passed through, so it is covariant
rather than invariant under a rigid motion of the team and can carry gauge
information. Whether it actually does is decided by what crosses the interface
between the channel estimator and the task estimator:

| Architecture | Interface variable | Gauge information |
|---|---|---|
| `A0` | none (channel-blind) | zero |
| `A_BL`, `A_BLS`, `A_BLF` | per-link scalars (weight / schedule / gate) | **exactly zero** |
| `A_HY` | one current-pose Gaussian information message | retains radio pose sensitivity |
| `A_HYG` | the same message restricted to the three gauge generators (9 numbers) | gauge block only |
| `A_J` | none, single joint state over poses and field | full |

Fixed scalar weights, ranks and gates applied only to invariant task factors add
zero direct gauge information. Exact Gaussian message equivalence requires
matching linearizations, distinct retained trajectory variables, and correct
prior and pose-field correlations. The supplied map is an external spatial
reference.

## Layout

```
coopsim/
  config.py, paths.py   committed defaults, CRN key derivation, filesystem roots
  world.py              urban / terrain / forest / open worlds, degenerate fields, trajectories, targets
  channel.py            Nakagami fading with AR(1) coherence, blockage, SINR, delivery
  detmodel.py           range-bearing detections, LoS gating, Poisson clutter
  field.py, factors.py  attenuation-field line integrals; odometry / relative / RSSI / decode factors
  estimator.py          sliding-window joint pose-field Gauss-Newton, Schur marginalization, A-HY message
  interfaces.py         the only architecture-conditional code
  regimes.py, scheduler.py, ladder.py, power.py, guidance.py, dualctrl.py
                        link belief bank, MWIS scheduling, channel-belief inner estimators, control seams
  targets.py, targets_lmb.py, fusion.py, sota.py
                        GM-PHD (JAX), LMB-lite, detection / track fusion, pose baselines
  rti_data.py           Utah SPAN radio-tomography loader
  harness.py            episode driver; writes one npz + JSON sidecar per episode
  metrics/              GOSPA, OSPA(2), IDF1, NEES, gauge decomposition, field and channel metrics
  runners/              experiment drivers and diagnostics
configs/config.yaml     simulator defaults (Experiments A and B carry their own protocols)
tools/aggregate_sensitivity.py
                        aggregate and validate the solver-sensitivity study
requirements.lock.txt   exact environment of the reproduction below
```

## Setup

Python 3.10 or newer (3.11 was used). From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # exact versions; JAX 0.10.2 with CUDA 12 wheels
pip install -e . --no-deps
```

Without an NVIDIA driver, replace the `jax*` and `nvidia-*` lines of the lock
file by `jax[cpu]==0.10.2`. `pip install -e .` alone resolves the loose bounds
in `pyproject.toml` and installs a CPU-only JAX.

Most simulator runners default to `JAX_PLATFORMS=cuda`; on a CPU-only install
export `JAX_PLATFORMS=cpu` first, otherwise the CUDA plugin fails at import.
Experiments A and B and the paper diagnostics default to CPU and float64 by
themselves. The simulator runners (`run_arch`, `run_m1_plateau` and the other
episode drivers) are float32 and must run without `JAX_ENABLE_X64`: with it
set, `run_arch` silently switches to float64 and `run_m1_plateau` fails with a
`lax.scan` dtype error. The reproduction commands below use `env -u
JAX_ENABLE_X64` where it matters. If the Python installation has user-site packages, export
`PYTHONNOUSERSITE=1` so they cannot shadow the pinned versions. Run commands
from the repository root; on a cluster, run them on compute nodes.

Filesystem roots are environment-configurable:

| Variable | Meaning | Default |
|---|---|---|
| `AETHER_SCRATCH` | episode output (`runs/` is created inside it) | `$SCRATCH/netcomm_v3`, else `./scratch/netcomm_v3` |
| `AETHER_REPO` | repository root for logs and figures | the repository |
| `AETHER_LOGS` | runner logs read by the figure scripts | `$AETHER_REPO/results/logs` |
| `AETHER_FIGS` | figure output | `$AETHER_REPO/netcomm_figs/v3` |
| `AETHER_RTI` | Utah SPAN dataset root | `$AETHER_SCRATCH/../rti_span` |

## Build gates

About one minute on one GPU (or on CPU with `JAX_PLATFORMS=cpu`):

```bash
python -u -m coopsim.runners.run_smoke --stage all      # world, channel, detection: 18 checks
python -u -m coopsim.runners.run_targets_check          # GM-PHD against a numpy reference
python -u -m coopsim.metrics._selftest                  # metric self-tests
python -u -m coopsim.runners.run_lmb_check              # optional: LMB-lite back end
```

## Running the simulator

One episode of one architecture:

```bash
ARCH=A_J WORLD=urban REGIME=R1 SEED=0 STEPS=3000 python -u -m coopsim.runners.run_arch
```

`ARCH` is one of `A0, A_BL, A_BLS, A_BLF, A_HY, A_HYG, A_J` (`coopsim.interfaces.ARCHS`);
`WORLD` is `urban, terrain, forest, open`, plus `constant, stratified, radial`
for the stabilizer experiments; `REGIME` is `R1` to `R4` (single/multi,
static/moving targets). `ORACLE_FIELD=1` supplies the true field instead of
estimating it. `CFG_SET` overrides config keys, comma separated, e.g.
`CFG_SET='field.tv_delta_db=0.5,shapes.n_robots=16'`; values cannot contain
commas and a misspelled leaf key is accepted silently, so check the JSON
sidecar of the first episode. Episodes are written to `$AETHER_SCRATCH/runs/`
as `{world}_{regime}_{arch}_s{seed}.npz` plus a sidecar with the full effective
configuration. The file name carries no configuration hash, so use a fresh
`AETHER_SCRATCH` per configuration.

Other runners, each `python -u -m coopsim.runners.<name>`:

| Runner | Purpose | Options |
|---|---|---|
| `run_m1_plateau` | historical drift comparison: `A0`, supplied field, self-estimated field, partial-gauge ladder | `M1_STEPS M1_SEEDS M1_SEED_OFFSET M1_GE M1_ARMS M1_WARMUP M1_BOOT M1_FREEZE M1_BURST M1_SAVE_FIELD CFG_SET` |
| `run_gauge_fisher` | gauge Fisher information per architecture | |
| `run_degenerate` | per-gauge-mode drift on the degenerate fields | `DEG_SEEDS DEG_STEPS` |
| `run_rti_real` | Utah SPAN emission fit and tomography | needs `AETHER_RTI` |
| `run_sota_pose` | invariant-EKF and GBP pose baselines | `SOTA_STEPS SOTA_SEEDS SOTA_ARMS` |
| `analyze_axis`, `paired_cm`, `diag_validate` | aggregate saved episodes; paired common-mode / relative tests; per-architecture cells | |
| `diag_blcheck`, `diag_relcount`, `diag_frame` | scalar-interface non-vacuity; usable relative measurements; estimated-field displacement | |
| `diag_chordrank`, `diag_stabfisher` | chord-to-voxel operator rank; stabilizer null of the gauge-restricted Fisher | `CR_N CR_STEPS CR_EVERY`; `SF_SEEDS SF_STEPS SF_ANGLES` |
| `diag_drift`, `diag_dichotomy` | growth-coefficient analysis of saved cohorts | `DRIFT_RUNS DRIFT_ARMS DRIFT_SEEDS DRIFT_EXPECTED_N`; `DICH_GLOB` |
| `audit_paper_results` | statistics over archived cells and logs | needs the archived manifest |
| `make_figures` | historical simulator figures | |

## Reproducing the paper

Set up once per shell:

```bash
source .venv/bin/activate
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/reproduction        # results/ is ignored by Git
```

### A: shared-map message construction

Do corrected messages preserve the joint Gaussian posterior over moving poses?
1,000 paired trials, 3 robots, 12 pose times, one uncertain shared map; naive
slotwise elimination is the negative control. Seconds on CPU.

```bash
python -u -m coopsim.runners.run_message_validation --output results/reproduction/experiment_a
```

The built-in configuration is the preregistered protocol and is written to
`preregistered_config.json` beside the outputs (`--config` accepts an archived
copy). Check `exactness_checks_passed` in `summary.json` and the
corrected-versus-joint discrepancies. Expected: corrected messages match the
joint posterior to about 1e-12, naive discrepancies 0.422 m and 0.163,
terminal-centroid NEES/2 1.012 versus 1.890, coverage 94.72% versus 84.88%.

### B: matched interfaces and map errors

Does consuming radio pose information help when map access, data,
initialization and effective robust weights match? 32 seeds, four map
conditions, three arms. About one second per seed.

```bash
python -m coopsim.runners.run_interface_matched --write-protocol --output results/reproduction/experiment_b
for s in $(seq 200 231); do
  python -u -m coopsim.runners.run_interface_matched --seed "$s" --output results/reproduction/experiment_b || break
done
python -m coopsim.runners.run_interface_matched --aggregate --output results/reproduction/experiment_b
```

Aggregation needs the complete cohort. Outputs: `seed_*.npz/.json`,
`summary.json`, `interface_errors` and `interface_consistency` PDF/PNG.
Expected: correct-map common RMSE 0.140 m (messages) versus 0.542 m (scalar
weights), paired −0.402 [−0.494, −0.311]; a 1 m map shift reverses the gain
(0.981 m); coverage 31/32, 0/32, 4/32, 1/32 across the four conditions.

### Solver sensitivity on the B inputs

Does B depend on the 25-iteration LM budget? Replays every fit, then strict LM
at 25/100/300 iterations and L-BFGS-B up to 1,000. A few seconds per seed.

```bash
mkdir -p results/reproduction/solver_sensitivity
for s in $(seq 200 231); do
  python -u -m coopsim.runners.run_interface_solver_sensitivity --seed "$s" \
    --source results/reproduction/experiment_b --output results/reproduction/solver_sensitivity || break
done
cp tools/aggregate_sensitivity.py results/reproduction/solver_sensitivity/
python results/reproduction/solver_sensitivity/aggregate_sensitivity.py --source results/reproduction/experiment_b
```

The runner verifies input hashes and replays the original fit to 1e-8 before
refining; the aggregate script validates all 640 checkpoints, reconstructs
metrics and covariances independently, and writes `validation.json`,
`summary.json`, `report.md` and two figure pairs. Expected: correct-map means
0.139794 / 0.139794 / 0.138429 m for original LM / strict LM-300 / L-BFGS-B,
scenario-mean changes below 2.1 mm, no checkpoint at the 1e-6 gradient
threshold.

### Gaussian and numerical-gauge checks

```bash
python -m coopsim.runners.diag_paper_gaussian --output results/reproduction/gaussian.json
python -m coopsim.runners.diag_paper_gauge --output results/reproduction/gauge.json
```

Both exit non-zero if their controls fail. The grid-Jacobian discrepancy of the
second is nonzero by design (radial field about 6.5%, above the 5% gate), so
no exact rotational null is validated on the numerical grid.

### Longer-horizon supplied map

Does a supplied map reduce finite-horizon growth of the common-mode error with
unknown headings and packet-level effects? 46 paired seeds, 3,000 slots,
supplied-map `A_J` against radio-blind `A0`. One episode takes 5 to 10 minutes
on a GPU (several can share one; each needs under 1 GB) or 20 to 30 minutes on
one CPU core.

```bash
export AETHER_SCRATCH="$PWD/results/reproduction/long_horizon"   # fresh and empty
for seed in $(seq 6 51); do
  for arm in A_J A0; do
    env -u JAX_ENABLE_X64 JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
      ARCH="$arm" SEED="$seed" WORLD=urban REGIME=R1 STEPS=3000 ORACLE_FIELD=1 \
      CFG_SET=power.enable=false python -u -m coopsim.runners.run_arch
  done
done
DRIFT_SEEDS=6 DRIFT_EXPECTED_N=46 python -u -m coopsim.runners.diag_drift
```

On a cluster, wrap the inner command in a job array over the 92 (seed, arm)
pairs. `diag_drift` requires exactly 46 episodes per arm with identical seed
sets and prints the per-arm growth coefficients and the paired difference.
Expected: `A0` grows at 0.088 [0.062, 0.114] m s^-1/2, `A_J` shows no growth,
paired t(45) about −8.

`A_J` and `A0` at the same seed share bit-identical truth, targets, field and
noise (common random numbers); realized schedules differ because scheduling
uses the current estimates. Episodes are float32. On GPUs two runs of the same
episode are not bit-identical (scatter-add accumulation is not
order-deterministic and the scheduler amplifies last-bit differences), and CPU
runs differ across machines; only the paired cohort statistic is expected to
reproduce, and it does to within a fraction of its confidence interval.

### Online-map comparison

Does estimating the map online instead of supplying it give a mean gain? The
paper reports none. Runs `run_m1_plateau` with the field estimated online
(`estF`) against `A0` at N = 8, one seed per process, keeping the logs:

```bash
mkdir -p results/reproduction/online_map/logs
for seed in $(seq 6 37); do
  env -u JAX_ENABLE_X64 M1_STEPS=3000 M1_SEEDS=1 M1_SEED_OFFSET="$seed" M1_GE=1 M1_ARMS=A0,estF \
    python -u -m coopsim.runners.run_m1_plateau > "results/reproduction/online_map/logs/default_s${seed}.out"
done
```

Add `CFG_SET='field.tv_delta_db=0.5,field.tv_lambda=0.3,field.prior_sigma_db=15'`
for the edge-preserving prior used by the headline estimated-field curves.
Aggregate the log files with `archived_m1(files, label, seed_lo, seed_hi,
expected_steps)` from `coopsim.runners.audit_paper_results`, which fits the
logged common error against square-root time and reports paired `estF` minus
`A0` statistics. Expected: the paired growth coefficient is indistinguishable
from zero under either prior. This runner has its own random-number namespace
and is not interchangeable with the `run_arch` cohort.

## Reproduction status

Independent reproduction from this tree with `requirements.lock.txt` on a Linux
cluster node (Intel Xeon Gold 6230R, NVIDIA A40, JAX 0.10.2 CUDA 12 wheels),
cross-checked on an Apple-silicon laptop:

| Evidence | Paper | Reproduced |
|---|---|---|
| Build gates | pass | smoke 18/18, targets check, metric self-tests 13/13 |
| A: exactness, NEES/2, coverage | pass; 1.012 vs 1.890; 94.72 / 84.88% | pass; 1.012 vs 1.890; 94.72 / 84.88% |
| B: common RMSE and Table I | 0.140 vs 0.542 m; shift 0.981 m | 0.13979 vs 0.54213 m; 0.98133 m; Table I identical |
| Solver sensitivity | 0.139794 / 0.139794 / 0.138429 m; < 2.1 mm | 0.139794 / 0.139794 / 0.138431 m; 1.979 mm |
| Gaussian diagnostic | pass | pass |
| Gauge diagnostic | leakage above the 5% gate | 6.5%, above the gate |
| Long-horizon cohort | A_J −0.008, A0 0.088 m s^-1/2; t(45) −7.93 | A_J −0.022 / −0.018 (two GPU replicates), A0 0.088; t(45) −7.89 / −6.98 |
| Online-map comparison | no mean gain | no mean gain (p = 0.67 / 0.17 under two priors) |

A, B, the solver study and the diagnostics agree between the two machines to
1e-11; every random draw is bit-identical across numpy versions.

## Notes on the protocol

Architecture comparisons use common random numbers to pair truth and simulated
random streams. Map access, robust-kernel settings and endpoints differ across
experiment groups, so the runner table is not one controlled ranking; statistics
are paired per seed. `A_J`, `A_HY` and `A_HYG` are configured with the `logdet`
link policy, but `run_episode` passes a uniform pose covariance to it, so it
reduces to the same age-ordered schedule `A0` uses; that keeps those arms
schedule-matched to the baseline. `A_BL/s` is the one arm whose treatment is
scheduling. `diag_drift` excludes tuning seeds 0 to 5 by default.

## Result provenance

| Paper result | Produced by |
|---|---|
| Shared-map message exactness (Experiment A) | `run_message_validation`, `diag_paper_gaussian` |
| Matched-interface map-error study (Experiment B) and solver sensitivity | `run_interface_matched`, `run_interface_solver_sensitivity`, `tools/aggregate_sensitivity.py` |
| Numerical gauge and grid checks | `diag_paper_gauge` |
| Reduced finite-horizon growth with a supplied field | `run_arch` (`A_J` / `A0`, `ORACLE_FIELD=1`) then `diag_drift` |
| Online map: no mean gain at N=8 | `run_m1_plateau` (`M1_ARMS=A0,estF`), `audit_paper_results.archived_m1` |
| Gauge Fisher information per architecture | `run_gauge_fisher` |
| Scalar interfaces: equal common-mode error, non-vacuous | `run_arch` (`A_BL, A_BLS, A_BLF`) then `diag_validate`; `diag_blcheck`, `diag_relcount` |
| Message / joint comparison with a supplied field; gauge-projected interface | `run_arch` (`A_HY`, `A_HYG`, `ORACLE_FIELD=1`) then `diag_validate` |
| Chord-to-voxel rank; stabilizer null; degenerate fields; partial-gauge ladder | `diag_chordrank`; `diag_stabfisher`; `run_degenerate`; `run_m1_plateau` (`M1_ARMS=A0,A_M,A_A1,A_A2`) |
| Real-radio emission fit and tomography | `run_rti_real` (needs `AETHER_RTI`) |

## Archived evidence

The archived `results/` tree (raw Experiment A/B outputs, the solver study,
the paper-alignment audit with its manifest, the 46-seed supplied-map cohort,
figures and manuscript sources) is not part of this repository. It is
distributed as the delivery archive `netcomm_acc2027.zip`; contact the authors
for access. `audit_paper_results` and the `--source` option of the
solver-sensitivity runner consume that archive directly.
