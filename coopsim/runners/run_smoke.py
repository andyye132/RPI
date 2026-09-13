"""Build-gate smoke checks for AETHER v3 (gates B0-B3).

Stages: env, world, channel, detmodel. Each prints PASS/FAIL lines; the sbatch log is
the gate record. Run: python -u -m coopsim.runners.run_smoke --stage all
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from coopsim import channel, detmodel, world
from coopsim.config import config_hash, crn_keys, load_config, master_key

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def stage_env(cfg):
    print(f"devices: {jax.devices()}", flush=True)
    print(f"config hash: {config_hash(cfg)}", flush=True)
    k1 = np.asarray(master_key("urban", "R1", 0))
    k2 = np.asarray(master_key("urban", "R1", 0))
    k3 = np.asarray(master_key("urban", "R1", 1))
    check("env.key_deterministic", bool((k1 == k2).all()))
    check("env.key_seed_sensitive", bool((k1 != k3).any()))


def stage_world(cfg):
    from coopsim.field import line_integral
    for kind in ("urban", "terrain", "forest", "open"):
        cfg.world.kind = kind
        keys = crn_keys(kind, "R1", 0)
        wld = world.make_world(keys.world_gen, cfg)
        check(f"world.{kind}.finite", bool(np.isfinite(wld.F_true).all()),
              f"max={wld.F_true.max():.2f} dB/m nonzero={int((wld.F_true > 0).sum())}")

    cfg.world.kind = "urban"
    keys = crn_keys("urban", "R1", 0)
    wld = world.make_world(keys.world_gen, cfg)
    # per-crossing loss: integrate straight through each block's center
    losses = []
    for (x0, x1, y0, y1, loss) in wld.meta["blocks"]:
        cy = (y0 + y1) / 2
        li = float(line_integral(jnp.array([max(x0 - 3, 0.5), cy]),
                                 jnp.array([min(x1 + 3, cfg.world.extent - 0.5), cy]),
                                 jnp.asarray(wld.F_true), cfg.world.extent))
        losses.append((li, loss))
    med = np.median([abs(li - l) / l for li, l in losses])
    check("world.urban.crossing_loss", med < 0.5,
          f"median rel err {med:.2f} over {len(losses)} blocks")

    cfg.world.trajectory = "transect"
    Xt, _ = world.make_trajectories(keys.target_gen, cfg)
    pe_t = world.pe_score(Xt, wld, cfg)
    cfg.world.trajectory = "convoy"
    Xc, _ = world.make_trajectories(keys.target_gen, cfg)
    pe_c = world.pe_score(Xc, wld, cfg)
    check("world.pe_transect_gg_convoy", pe_t["crossings"] > 3 * max(pe_c["crossings"], 1),
          f"transect={pe_t['crossings']} convoy={pe_c['crossings']}")
    cfg.world.trajectory = "transect"

    for regime, nom in (("R1", 1), ("R4", 8)):
        cfg.targets.regime = regime
        tau, alive = world.gen_targets(keys.target_gen, cfg)
        counts = alive.sum(1)
        check(f"world.targets.{regime}", abs(counts.mean() - nom) < 0.5 * nom + 0.5,
              f"mean count {counts.mean():.2f} (nominal {nom})")
    cfg.targets.regime = "R1"


def _sched_tx(t, n, links=1):
    """Interference-free smoke schedule: `links` directed links, rotated by slot.
    One link per slot isolates the radio physics; concurrent links are the
    scheduler's job (conflict graph, gate B7)."""
    tx = np.zeros((n, n), bool)
    for k in range(links):
        i = (2 * k + t) % n
        j = (2 * k + 1 + t) % n
        if i != j:
            tx[i, j] = True
    return jnp.asarray(tx)


def stage_channel(cfg):
    cfg.world.kind = "open"
    keys = crn_keys("open", "R1", 0)
    wld = world.make_world(keys.world_gen, cfg)
    n = cfg.shapes.n_robots
    lut = channel.nakagami_lut(cfg.channel.nakagami_m.open)
    ext, dt = cfg.world.extent, cfg.time.dt
    st = channel.init_state(keys.fading, n)
    F = jnp.asarray(wld.F_true)
    step = jax.jit(lambda t, pos, vel, s, tx: channel.step(
        t, keys, pos, vel, F, s, tx, lut, cfg.channel, ext, dt))

    rng = np.random.default_rng(0)
    dists, succ, rssi = [], [], []
    for t in range(600):
        pos = jnp.asarray(rng.uniform(2, ext - 2, (n, 2)).astype(np.float32))
        vel = jnp.asarray(rng.normal(0, 1.0, (n, 2)).astype(np.float32))
        tx = _sched_tx(t, n)
        obs, st = step(t, pos, vel, st, tx)
        att = np.asarray(obs.attempted)
        d = np.linalg.norm(np.asarray(pos)[:, None] - np.asarray(pos)[None], axis=-1)
        s = np.asarray(obs.success)
        r = np.asarray(obs.rssi_q)
        dists += list(d[att]); succ += list(s[att]); rssi += list(r[att])
    dists, succ, rssi = np.array(dists), np.array(succ), np.array(rssi)
    # open world, scheduled links: high delivery, and mean RSSI follows path loss
    check("channel.open.delivery", succ.mean() > 0.9, f"delivery={succ.mean():.2f}")
    ok = succ & (dists > 3)
    slope = np.polyfit(np.log10(dists[ok]), rssi[ok], 1)[0]
    check("channel.open.pathloss_slope", abs(slope + 10 * cfg.channel.gamma) < 4.0,
          f"slope={slope:.1f} want~{-10 * cfg.channel.gamma:.0f}")

    # urban: NLoS links deliver less than LoS links at matched distance
    cfg.world.kind = "urban"
    wldu = world.make_world(crn_keys("urban", "R1", 0).world_gen, cfg)
    Fu = jnp.asarray(wldu.F_true)
    lut_u = channel.nakagami_lut(cfg.channel.nakagami_m.urban)
    stu = channel.init_state(keys.fading, n)
    step_u = jax.jit(lambda t, pos, vel, s, tx: channel.step(
        t, keys, pos, vel, Fu, s, tx, lut_u, cfg.channel, ext, dt))
    from coopsim.field import line_integral_links
    los_s, nlos_s = [], []
    for t in range(600):
        pos = jnp.asarray(rng.uniform(2, ext - 2, (n, 2)).astype(np.float32))
        vel = jnp.asarray(rng.normal(0, 1.0, (n, 2)).astype(np.float32))
        tx = _sched_tx(t, n)
        obs, stu = step_u(t, pos, vel, stu, tx)
        att = np.asarray(obs.attempted)
        li = np.asarray(line_integral_links(pos, Fu, ext))
        d = np.linalg.norm(np.asarray(pos)[:, None] - np.asarray(pos)[None], axis=-1)
        mid = att & (d > 10) & (d < 35)
        s = np.asarray(obs.success)
        los_s += list(s[mid & (li < 3)]); nlos_s += list(s[mid & (li > 10)])
    check("channel.urban.shadowing", np.mean(los_s) > np.mean(nlos_s) + 0.15,
          f"LoS={np.mean(los_s):.2f} NLoS={np.mean(nlos_s):.2f} "
          f"(n={len(los_s)}/{len(nlos_s)})")

    # GE dwell in the bad state ~ 1/(1-persist)
    stg = channel.init_state(keys.fading, n)
    bad = []
    pos0 = jnp.asarray(rng.uniform(2, ext - 2, (n, 2)).astype(np.float32))
    vel0 = jnp.zeros((n, 2), jnp.float32)
    tx0 = _sched_tx(0, n)
    for t in range(2000):
        obs, stg = step_u(t, pos0, vel0, stg, tx0)
        bad.append(np.asarray(stg.ge_bad))
    bad = np.stack(bad)
    entries = ((~bad[:-1]) & bad[1:]).sum()
    dwell = bad.sum() / max(entries, 1)
    want = 1.0 / (1.0 - cfg.channel.ge_persist)
    check("channel.ge_dwell", abs(dwell - want) < 0.4 * want,
          f"dwell={dwell:.1f} want~{want:.1f}")

    # CRN: same keys, different tx mask -> identical fading/blockage streams
    st1 = channel.init_state(keys.fading, n)
    st2 = channel.init_state(keys.fading, n)
    tx2 = jnp.zeros((n, n), bool).at[0, 1].set(True)
    for t in range(50):
        _, st1 = channel.step(t, keys, pos0, vel0, Fu, st1, tx0, lut_u, cfg.channel, ext, dt)
        _, st2 = channel.step(t, keys, pos0, vel0, Fu, st2, tx2, lut_u, cfg.channel, ext, dt)
    check("channel.crn_schedule_independent",
          bool(jnp.allclose(st1.u_fade, st2.u_fade) and (st1.ge_bad == st2.ge_bad).all()))


def stage_detmodel(cfg):
    cfg.world.kind = "urban"
    keys = crn_keys("urban", "R4", 0)
    wld = world.make_world(keys.world_gen, cfg)
    cfg.targets.regime = "R4"
    tau, alive = world.gen_targets(keys.target_gen, cfg)
    Xt, _ = world.make_trajectories(keys.target_gen, cfg)
    F = jnp.asarray(wld.F_sense)
    ext, D = cfg.world.extent, cfg.shapes.max_det
    gen = jax.jit(lambda t, X, tt, al: detmodel.gen_detections(
        t, keys, X, tt, al, F, ext, cfg.detection, D))

    hits, pds, clut_counts, overflow = [], [], [], 0
    for t in range(0, 1500, 5):
        dets = gen(t, jnp.asarray(Xt[t]), jnp.asarray(tau[t]), jnp.asarray(alive[t]))
        mt = tau.shape[1]
        pd = np.stack([[float(detmodel.p_detect(jnp.asarray(Xt[t, i, :2]),
                                                jnp.asarray(tau[t, k, :2]), F, ext,
                                                cfg.detection))
                        for k in range(mt)] for i in range(Xt.shape[1])])
        m = np.asarray(dets.mask)[:, :mt] & np.asarray(alive[t])[None, :]
        hits.append(np.asarray(dets.mask)[:, :mt][:, alive[t]])
        pds.append(pd[:, alive[t]])
        clut_counts += list(np.asarray(dets.mask)[:, mt:].sum(1))
        overflow += int(np.asarray(dets.overflow).sum())
    hits, pds = np.concatenate([h.ravel() for h in hits]), np.concatenate([p.ravel() for p in pds])
    lo, hi = pds < 0.3, pds > 0.7
    check("detmodel.pd_gating", hits[hi].mean() > 0.6 and hits[lo].mean() < 0.25,
          f"hit|highpd={hits[hi].mean():.2f} hit|lowpd={hits[lo].mean():.2f}")
    check("detmodel.clutter_rate", abs(np.mean(clut_counts) - cfg.detection.clutter_rate) < 0.5,
          f"mean={np.mean(clut_counts):.2f} want={cfg.detection.clutter_rate}")
    check("detmodel.overflow", overflow == 0, f"overflow={overflow}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "env", "world", "channel", "detmodel"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    stages = ["env", "world", "channel", "detmodel"] if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"\n== stage {s} ==", flush=True)
        globals()[f"stage_{s}"](cfg)
    print(f"\n{'ALL PASS' if all(RESULTS) else 'FAILURES PRESENT'} "
          f"({sum(RESULTS)}/{len(RESULTS)})")
    print("DONE smoke")


if __name__ == "__main__":
    main()
