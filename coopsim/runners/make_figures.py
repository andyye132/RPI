"""Publication figures for AETHER v3 (matplotlib + seaborn, no jax).

Reads the experiment logs and saved npz episodes and renders the figures whose data
exists: common-mode drift (kill gate), interface gauge-Fisher, metric decomposition,
field reconstruction, and world families. PDF + PNG to netcomm_figs/v3/.

Fixed entity-to-color assignment (seaborn colorblind palette, CVD-safe), identical
across every figure; oracle rows additionally dashed so identity never rides on
color alone. Run: python -u -m coopsim.runners.make_figures
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from coopsim.metrics.gauge import gauge_decompose
from coopsim.metrics.gospa import gospa_series
from coopsim.metrics.identity import identity_metrics
from coopsim.metrics.field_metrics import field_iou

from coopsim.paths import LOGS, RUNS, FIGS as OUT

COLOR = {"A0": "#949494", "A_BL": "#DE8F05", "A_HY": "#029E73",
         "A_J": "#0173B2", "oracleF": "#CC78BC", "estF": "#0173B2",
         "relative": "#949494"}
DASH = {"oracleF": (4, 2)}
LABEL = {"A0": "A0 channel-blind", "A_BL": "A-BL scalar bilevel",
         "A_HY": "A-HY message", "A_J": "A-J joint",
         "oracleF": "A-J oracle field", "estF": "A-J estimated field"}

sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.7, "font.size": 8.5,
    "legend.frameon": False, "pdf.fonttype": 42,
    "figure.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def parse_m1(job, pattern=None):
    """{arm: {seed: (t_array, cm_array)}} from kill-gate or sweep logs."""
    out = {}
    for f in sorted(LOGS.glob(pattern or f"v3_m1full_{job}_*.out")):
        seed, arm = None, None
        for line in f.read_text().splitlines():
            m = re.match(r"\[seed (\d+)\] (\w+)", line)
            if m:
                seed, arm = int(m.group(1)), m.group(2)
                out.setdefault(arm, {})[seed] = ([], [])
                continue
            m = re.search(r"t=\s*(\d+) cm=\s*([\d.]+)", line)
            if m and arm is not None:
                out[arm][seed][0].append(int(m.group(1)))
                out[arm][seed][1].append(float(m.group(2)))
    return out


def fig_common_mode(job="3930269", tuned_jobs=("4162545", "4166154")):
    """A0 / supplied-field / self-estimated-field drift. The estF curve uses the
    edge-preserving field prior (tv_delta 0.5, tv_lambda 0.3, prior_sigma 15),
    which is what bounds the self-contained common mode; the default prior
    over-smooths the wall edges that carry the gauge information."""
    data = parse_m1(job)
    tuned = {}
    for j in tuned_jobs:                       # pool the tuned-prior seeds
        for arm, seeds in parse_m1(None, f"v3_tv*_{j}.out").items():
            tuned.setdefault(arm, {}).update(
                {f"{j}_{s}": v for s, v in seeds.items()})
    # use BOTH arms from the tuned runs so A0 and estF stay CRN-paired on the
    # same 24 seeds; oracleF remains the supplied-field reference from the gate
    if tuned.get("estF"):
        data["estF"] = tuned["estF"]
    if tuned.get("A0"):
        data["A0"] = tuned["A0"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for arm in ("A0", "oracleF", "estF"):
        seeds = data.get(arm, {})
        if not seeds:
            continue
        ts = np.array(min(seeds.values(), key=lambda v: len(v[0]))[0])
        cm = np.stack([np.array(v[1][:len(ts)]) for v in seeds.values()])
        mu, sd = cm.mean(0), cm.std(0)
        ax.plot(ts * 0.1, mu, color=COLOR[arm], lw=1.6,
                dashes=DASH.get(arm, ()), label=LABEL[arm])
        ax.fill_between(ts * 0.1, mu - sd, mu + sd, color=COLOR[arm],
                        alpha=0.15, lw=0)
        ax.annotate(LABEL[arm], xy=(ts[-1] * 0.1, mu[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color="#3a3a3a", fontsize=7, va="center")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("scene common-mode error (m)")
    ax.set_xlim(0, 340)
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig_common_mode")


def fig_interface(log="v3_gf2_4348932.out"):
    """Gauge Fisher information per interface, with the joint estimator's own gauge
    marginal (field eliminated once against the total) overlaid on the per-slot
    message sum. The shaded gap between them is the Remark-3 double counting: the
    per-slot curve is overconfidence, not information."""
    txt = (LOGS / log).read_text()
    rows = re.findall(r"t=\s*(\d+)\s+rel\[tr=\s*(-?[\d.]+).*?"
                      r"oracle\[tr=\s*(-?[\d.]+).*?est\[tr=\s*(-?[\d.]+).*?"
                      r"joint\[tr=\s*(-?[\d.]+)", txt)
    t = np.array([int(r[0]) for r in rows]) * 0.1
    rel, orc, est, jt = (np.array([float(r[k]) for r in rows]) for k in (1, 2, 3, 4))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.fill_between(t, jt, est, color=COLOR["A_HY"], alpha=0.16, lw=0)
    ax.plot(t, orc, color=COLOR["oracleF"], lw=1.6, dashes=(4, 2),
            label="channel factors, known field")
    ax.plot(t, est, color=COLOR["A_HY"], lw=1.6,
            label="per-slot message sum (Rem. 4)")
    ax.plot(t, jt, color=COLOR["A_J"], lw=1.6,
            label="joint marginal, eliminated once")
    ax.plot(t, rel, color=COLOR["A0"], lw=1.8,
            label="relative / any scalar interface")
    ax.annotate(f"double counting\n({est[-1] / jt[-1]:.1f}$\\times$)",
                xy=(t[-1], 0.5 * (est[-1] + jt[-1])), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center",
                color="#3a3a3a", fontsize=7)
    # sits BELOW the flat grey curve at mid-span: at the right edge the joint
    # marginal has climbed through where this label used to be
    ax.annotate("exactly zero (Prop. 1)", xy=(t[len(t) // 2], 0), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top",
                color="#3a3a3a", fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gauge Fisher information")
    ax.legend(loc="upper left", fontsize=6.6)
    save(fig, "fig_interface")


def _episode_stats(path):
    d = np.load(path, allow_pickle=True)
    T = d["x_est"].shape[0]
    f3 = slice(2 * T // 3, T)
    g = gauge_decompose(d["X_true"][:, :, :2], d["x_est"][:, :, :2])
    n = d["tracks"].shape[1]
    rel = g["relative"][f3].mean()
    false_c, idf = [], []
    for r in range(n):
        s = gospa_series(d["tau"], d["alive"], d["tracks"][:, r, :, :2],
                         d["tracks_mask"][:, r])
        false_c.append(s["false"][f3].mean())
        idm = identity_metrics(d["tau"], d["alive"], d["tracks"][:, r, :, :2],
                               d["tracks"][:, r, :, 4].astype(int),
                               d["tracks_mask"][:, r])
        idf.append(idm["idf1"])
    return rel, np.mean(false_c), np.mean(idf)


def fig_decomp(regime="R4"):
    archs = ["A0", "A_BL", "A_HY", "A_J"]
    stats = {a: [] for a in archs}
    for a in archs:
        for p in sorted(RUNS.glob(f"urban_{regime}_{a}_s?.npz")):
            stats[a].append(_episode_stats(p))
    panels = [("relative pose error (m)", 0), ("GOSPA false component", 1),
              ("IDF1", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (title, k) in zip(axes, panels):
        vals = [np.array([s[k] for s in stats[a]]) for a in archs]
        x = np.arange(len(archs))
        ax.bar(x, [v.mean() for v in vals],
               yerr=[v.std() for v in vals], width=0.62,
               color=[COLOR[a] for a in archs], error_kw=dict(lw=0.8),
               capsize=2)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v.mean():.2f}", xy=(xi, v.mean() + v.std()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#3a3a3a")
        ax.set_xticks(x, ["A0", "A-BL", "A-HY", "A-J"], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x", visible=False)
    save(fig, f"fig_decomp_{regime}")


def fig_field():
    # prefer the MWIS-scheduled kill-gate estF snapshots; fall back to a harness run
    m1 = RUNS / "m1_field_s0.npz"
    src = m1 if m1.exists() else RUNS / "urban_R1_A_J_s0.npz"
    d = np.load(src, allow_pickle=True)
    snaps, F_true = d["field_snaps"], d["F_true"]
    if snaps.shape[0] == 0:
        print("no field snapshots, skipping fig_field")
        return
    vmax = max(F_true.max(), snaps[-1].max())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, (F, title) in zip(axes[:2], [(F_true, "true field"),
                                         (snaps[-1], "estimate (end)")]):
        im = ax.imshow(F, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                       extent=[0, 50, 0, 50])
        ax.set_title(title, fontsize=8)
        ax.set_xticks([0, 25, 50]); ax.set_yticks([0, 25, 50])
        ax.grid(visible=False)
    fig.colorbar(im, ax=axes[:2], shrink=0.82, pad=0.02, label="dB/m")
    mae = [np.abs(s - F_true).mean() for s in snaps]
    t = (np.arange(1, len(mae) + 1) * 100) * 0.1
    axes[2].plot(t, mae, color=COLOR["A_J"], lw=1.6)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("attenuation MAE (dB/m)")
    axes[2].annotate(f"final IoU {field_iou(snaps[-1], F_true):.2f}",
                     xy=(t[-1], mae[-1]), xytext=(-6, 10),
                     textcoords="offset points", ha="right",
                     color="#3a3a3a", fontsize=7)
    print(f"fig_field source: {src.name}", flush=True)
    save(fig, "fig_field")


def fig_worlds():
    rows = []
    for f in sorted(LOGS.glob("v3_worlds_3934248_*.out")):
        txt = f.read_text()
        m = re.search(r"run_arch: (\w+) R1 (\w+) seed=(\d+)", txt)
        c = re.findall(r"^common-mode:.*t=\d+: ([\d.]+)$", txt, re.M)
        if m and c:
            rows.append((m.group(1), m.group(2), float(c[0])))
    for f in sorted(LOGS.glob("v3_terrsm_*.out")):
        txt = f.read_text()
        c = re.findall(r"^common-mode:.*t=\d+: ([\d.]+)$", txt, re.M)
        if c:
            rows.append(("terrain (smooth prior)", "A_J", float(c[0])))
    worlds = ["forest", "open", "terrain", "terrain (smooth prior)"]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    width = 0.36
    for k, arch in enumerate(("A0", "A_J")):
        mus, sds, xs = [], [], []
        for i, w in enumerate(worlds):
            vals = np.array([r[2] for r in rows if r[0] == w and r[1] == arch])
            if len(vals) == 0:
                continue
            xs.append(i + (k - 0.5) * width)
            mus.append(vals.mean()); sds.append(vals.std())
        ax.bar(xs, mus, yerr=sds, width=width, color=COLOR[arch],
               label=LABEL[arch], error_kw=dict(lw=0.8), capsize=2)
    ax.set_xticks(range(len(worlds)),
                  ["forest", "open", "terrain\n(TV prior)",
                   "terrain\n(smooth prior)"], fontsize=7)
    ax.set_ylabel("final common-mode (m)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_worlds")


def fig_rti():
    p = RUNS / "rti_real.npz"
    if not p.exists():
        print("no rti_real.npz, skipping fig_rti")
        return
    d = np.load(p, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained")
    # emission fit
    x = 10.0 * np.log10(d["link_d"])
    axes[0].scatter(x, d["link_rss"], s=14, color=COLOR["A_J"], alpha=0.7,
                    edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 50)
    axes[0].plot(xs, float(d["p0"]) - float(d["gamma"]) * xs, color="#3a3a3a",
                 lw=1.5)
    axes[0].set_xlabel(r"$10\log_{10} d$")
    axes[0].set_ylabel("mean RSS (dBm)")
    axes[0].set_title(f"emission fit: $\\gamma$={float(d['gamma']):.2f}, "
                      f"$\\sigma$={float(d['sigma']):.1f} dB, "
                      f"$R^2$={float(d['r2']):.2f}", fontsize=8)
    # tomography: time-averaged field + walked path + nodes
    ext = float(d["extent"])
    im = axes[1].imshow(d["avg_field"], origin="lower", cmap="Blues",
                        extent=[0, ext, 0, ext])
    path = d["path"]
    axes[1].plot(path[:, 0], path[:, 1], color="#DE8F05", lw=1.4,
                 label="walked path")
    axes[1].scatter(d["coords"][:, 0], d["coords"][:, 1], s=22, marker="^",
                    color="#3a3a3a", label="nodes", zorder=3)
    axes[1].set_title(f"real-RSS tomography: peak {float(d['err_median']):.2f} m "
                      f"from path", fontsize=8)
    axes[1].set_xlim(0, ext); axes[1].set_ylim(0, ext)
    # opaque frame: node markers sat directly behind the legend text
    axes[1].legend(fontsize=6.5, loc="upper right", framealpha=0.95,
                   facecolor="white", edgecolor="0.7").set_zorder(5)
    axes[1].grid(visible=False)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="attenuation")
    save(fig, "fig_rti")


def _field_underlay(ax, F, ext=50.0, alpha=0.85):
    """Light single-hue attenuation underlay so colored marks stay dominant."""
    vmax = np.percentile(F[F > 0], 95) if (F > 0).any() else 1.0
    ax.imshow(F.T, origin="lower", extent=[0, ext, 0, ext], cmap="Greys",
              vmin=0, vmax=1.6 * vmax, alpha=alpha, zorder=0)
    ax.grid(visible=False)
    ax.set_aspect("equal")
    ax.set_xlim(0, ext); ax.set_ylim(0, ext)


def fig_topology(stem="urban_R1_A_J_s0", t=2000, win=30, trail=120):
    """Scenario snapshot: field, robot team + trails, scheduled link topology
    (delivered vs blocked over the last `win` slots), targets, sensing range."""
    p = RUNS / f"{stem}.npz"
    if not p.exists():
        print(f"skip topology ({stem} missing)")
        return
    d = np.load(p, allow_pickle=True)
    X, tau, alive = d["X_true"], d["tau"], d["alive"]
    F, sched, succ = d["F_true"], d["sched"], d["succ"]
    n = X.shape[1]
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    _field_underlay(ax, F)
    att = (sched[t - win:t] > 0).any(0)
    suc = succ[t - win:t].any(0)
    pos = X[t, :, :2]
    for i in range(n):
        for j in range(i + 1, n):
            if att[i, j] or att[j, i]:
                ok = suc[i, j] or suc[j, i]
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="#029E73" if ok else "#D55E00",
                        ls="-" if ok else (0, (3, 2)),
                        lw=1.1 if ok else 0.9, alpha=0.9 if ok else 0.65, zorder=2)
    for i in range(n):
        ax.plot(X[t - trail:t, i, 0], X[t - trail:t, i, 1],
                color="#0173B2", lw=0.7, alpha=0.35, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=30, color="#0173B2",
               edgecolor="white", linewidth=0.8, zorder=4)
    a = alive[t].astype(bool)
    if a.any():
        ax.scatter(tau[t, a, 0], tau[t, a, 1], s=34, marker="D",
                   color="#DE8F05", edgecolor="white", linewidth=0.8, zorder=4)
    ax.add_patch(plt.Circle(pos[0], 12.0, fill=False, ec="#949494",
                            ls=":", lw=0.8, zorder=1))
    handles = [
        plt.Line2D([], [], color="#0173B2", marker="o", ls="", mec="white",
                   label="robot"),
        plt.Line2D([], [], color="#DE8F05", marker="D", ls="", mec="white",
                   label="target"),
        plt.Line2D([], [], color="#029E73", lw=1.2, label="link delivered"),
        plt.Line2D([], [], color="#D55E00", lw=1.0, ls=(0, (3, 2)),
                   label="link blocked"),
        plt.Line2D([], [], color="#949494", lw=0.8, ls=":",
                   label="sensing range"),
    ]
    ax.legend(handles=handles, fontsize=6.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), handlelength=1.6,
              columnspacing=1.0)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    save(fig, "fig_topology")


def fig_gauge(regime="R1", t=2900, amp=1.5):
    """Gauge anatomy: per-robot error arrows (amplified) + the shared mean-error
    arrow (common mode), A0 vs A_J on the representative (median common-mode)
    seed. Shows visually what the decomposition separates."""
    seeds = []
    for p in sorted(RUNS.glob(f"urban_{regime}_A0_s*.npz")):
        s = p.stem.rsplit("_s", 1)[1]
        if (RUNS / f"urban_{regime}_A_J_s{s}.npz").exists():
            seeds.append(s)
    if not seeds:
        print("skip gauge (no paired episodes)")
        return
    cms = []
    for s in seeds:
        d = np.load(RUNS / f"urban_{regime}_A0_s{s}.npz")
        e = d["x_est"][t, :, :2] - d["X_true"][t, :, :2]
        cms.append(np.linalg.norm(e.mean(0)))
    s = seeds[int(np.argsort(cms)[len(cms) // 2])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5), sharey=True)
    for ax, arch in zip(axes, ("A0", "A_J")):
        d = np.load(RUNS / f"urban_{regime}_{arch}_s{s}.npz")
        pos = d["X_true"][t, :, :2]
        err = d["x_est"][t, :, :2] - pos
        cm_v = err.mean(0)
        rel = float(np.sqrt((np.linalg.norm(err - cm_v, axis=1) ** 2).mean()))
        _field_underlay(ax, d["F_true"], alpha=0.35)
        ax.quiver(pos[:, 0], pos[:, 1], amp * err[:, 0], amp * err[:, 1],
                  angles="xy", scale_units="xy", scale=1, color="#3a3a3a",
                  width=0.0045, headwidth=4.5, headlength=5, alpha=0.9,
                  zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=24, color=COLOR[arch],
                   edgecolor="white", linewidth=0.7, zorder=4)
        c = pos.mean(0)
        ax.scatter([c[0]], [c[1]], s=42, marker="x", color=COLOR[arch],
                   linewidth=1.6, zorder=5)
        ax.quiver(c[0], c[1], amp * cm_v[0], amp * cm_v[1], angles="xy",
                  scale_units="xy", scale=1, color=COLOR[arch], width=0.014,
                  headwidth=3.5, headlength=4, edgecolor="white",
                  linewidth=1.0, zorder=5)
        ax.annotate(f"common mode {np.linalg.norm(cm_v):.2f} m\n"
                    f"relative {rel:.2f} m\n(arrows {amp:g}x)",
                    (0.03, 0.97), xycoords="axes fraction", va="top",
                    fontsize=7.5, zorder=6,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2.5))
        ax.set_title(LABEL[arch], fontsize=9)
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    save(fig, "fig_gauge")


def fig_decomp_time(regime="R1", step=25):
    """Relative (left) and common-mode (right) error vs time for A0/A_BL/A_J:
    the letter's empirical story as two clean line panels. A_BL drawn dashed so
    its coincidence with the A0 line stays visible."""
    dt = 0.1
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)
    for arch in ("A0", "A_BL", "A_J"):
        rels, cms = [], []
        for p in sorted(RUNS.glob(f"urban_{regime}_{arch}_s*.npz")):
            # seeds 0-5 were overwritten by later patrol/dense config tests;
            # keep only untouched transect matrix episodes (plain-seed names)
            tag = p.stem.rsplit("_s", 1)[1]
            if not tag.isdigit() or int(tag) < 6:
                continue
            d = np.load(p)
            g = gauge_decompose(d["X_true"][:, :, :2], d["x_est"][:, :, :2])
            rels.append(g["relative"][step::step])   # skip the pre-init slot 0
            cms.append(g["common"][step::step])
        if not rels:
            print(f"skip decomp_time ({arch} missing)")
            return
        L = min(len(r) for r in rels)
        rels = [r[:L] for r in rels]
        cms = [c[:L] for c in cms]
        print(f"{arch}: {len(rels)} episodes, {L} points", flush=True)
        t = (1 + np.arange(L)) * step * dt
        for ax, series in zip(axes, (np.stack(rels), np.stack(cms))):
            m, se = series.mean(0), series.std(0) / np.sqrt(len(series))
            ls = (0, (4, 2)) if arch == "A_BL" else "-"
            ax.plot(t, m, color=COLOR[arch], ls=ls, lw=1.6,
                    label=LABEL[arch], zorder=3 if arch == "A_BL" else 2)
            ax.fill_between(t, m - se, m + se, color=COLOR[arch], alpha=0.15,
                            linewidth=0)
    axes[0].set_ylabel("relative error [m]")
    axes[1].set_ylabel("common-mode error [m]")
    for ax in axes:
        ax.set_xlabel("time [s]")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_decomp_time")



def fig_n16_common():
    """The headline: at N=16 the self-estimated field realizes the bound.
    Per-seed paired difference (estF minus A0) under common random numbers,
    mean and 95 percent CI over the 48 pre-registered confirmation seeds."""
    data = {}
    for pat in ("v3_n16conf_o8_4267180.out", "v3_n16conf_o24_4267181.out",
                "v3_n16conf_o40_4267182.out"):
        for arm, seeds in parse_m1(None, pat).items():
            data.setdefault(arm, {}).update(
                {f"{pat}_{s}": v for s, v in seeds.items()})
    a0, ef = data.get("A0", {}), data.get("estF", {})
    keys = sorted(set(a0) & set(ef))
    if not keys:
        print("skip fig_n16_common (no paired seeds)")
        return
    L = min(min(len(a0[k][0]) for k in keys), min(len(ef[k][0]) for k in keys))
    ts = np.array(a0[keys[0]][0][:L]) * 0.1
    diff = np.stack([np.array(ef[k][1][:L]) - np.array(a0[k][1][:L])
                     for k in keys])
    mu = diff.mean(0)
    ci = 1.96 * diff.std(0, ddof=1) / np.sqrt(len(keys))
    fig, ax = plt.subplots(figsize=(2.2, 1.75))
    ax.axhline(0, color="#949494", lw=0.8)
    ax.plot(ts, mu, color=COLOR["estF"], lw=1.5)
    ax.fill_between(ts, mu - ci, mu + ci, color=COLOR["estF"],
                    alpha=0.18, lw=0)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("paired $\\Delta$ common mode (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 305)
    save(fig, "fig_n16_paired")



def fig_stabilizer(log="v3_stabfisher_4327687.out"):
    """Proposition 3 measured: information along the gauge circle vanishes exactly at
    the stabilizer direction. Parses the q(phi)/mu1 sweeps out of the stabilizer-Fisher
    log, so the figure is reproducible from the run rather than transcribed."""
    txt = (LOGS / log).read_text()
    want = {"stratified-repo": "stratified field", "urban": "urban control",
            "warped-strata": "placebo, symmetry destroyed"}
    series, kind = {}, None
    for line in txt.splitlines():
        head = line.split()
        if head and head[0] in want:
            kind = head[0]
        if "q(phi)/mu1" in line and kind in want:
            series[kind] = [float(v) for v in line.split(":")[-1].split()]
            kind = None
    if "stratified-repo" not in series:
        print("skip fig_stabilizer (sweep not found)")
        return
    phi = np.arange(0, 180, 15)
    dense = np.linspace(0, 180, 361)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(dense, np.sin(np.deg2rad(dense)) ** 2, color="#949494", lw=1.0,
            dashes=(4, 2), label=r"theory $\sin^2\varphi$", zorder=1)
    style = {"stratified-repo": ("#0173B2", "o"), "warped-strata": ("#DE8F05", "s"),
             "urban": ("#029E73", "^")}
    for k in ("stratified-repo", "warped-strata", "urban"):
        if k not in series:
            continue
        c, mk = style[k]
        ax.plot(phi, series[k], color=c, lw=1.4, marker=mk, ms=3.4,
                label=want[k], zorder=2)
    # dropped into the empty band under all three curves; at y=0.16 the text ran
    # straight through the rising stratified curve
    ax.annotate("exact null at the predicted\nstabilizer direction",
                xy=(0, 0.012), xytext=(42, 0.115), fontsize=6.5, color="#3a3a3a",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#3a3a3a"))
    ax.set_xlabel(r"direction $\varphi$ in the translation plane (deg)",
                  fontsize=8)
    ax.set_ylabel("channel information\n(normalized)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-4, 168)
    ax.set_ylim(-0.05, 1.30)
    ax.set_xticks([0, 45, 90, 135])
    ax.legend(loc="upper center", ncol=2, fontsize=6.2, columnspacing=1.0,
              handlelength=1.8, borderaxespad=0.2)
    save(fig, "fig_stabilizer")


def main():
    fig_decomp_time()
    fig_topology()
    fig_gauge()
    fig_rti()
    fig_common_mode()
    fig_interface()
    fig_decomp("R4")
    fig_decomp("R1")
    fig_field()
    fig_worlds()
    fig_n16_common()
    print("DONE figures")


if __name__ == "__main__":
    main()
