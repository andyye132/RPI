"""Synthetic range-bearing point detector (jit, vmapped over robots).

The (z, R) interface: detections are body-frame (range, bearing) with covariance, so
the tracking layer is detector-agnostic by construction. p_D is gated softly through
the same attenuating structure that blocks the radio (F_sense == F_true unless the CH5
decoupling toggle). Fixed shape MAX_DET per robot per scan: true-target slots first,
clutter tail after, with a validity mask and an overflow counter.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coopsim.field import line_integral


class Dets(NamedTuple):
    z: jax.Array          # (N, D, 2) range, bearing (body frame)
    R: jax.Array          # (N, D, 2, 2)
    mask: jax.Array       # (N, D) bool
    is_clutter: jax.Array # (N, D) bool
    gt_idx: jax.Array     # (N, D) int32, -1 for clutter
    overflow: jax.Array   # (N,) clutter draws beyond capacity


def p_detect(xi, tau_xy, F_sense, extent, cfg_det):
    """Detection probability of one target from one robot position."""
    r = jnp.linalg.norm(tau_xy - xi)
    li = line_integral(xi, tau_xy, F_sense, extent)
    plos = jax.nn.sigmoid((cfg_det.los_gate_db - li) / cfg_det.los_gate_scale_db)
    return cfg_det.p0 * (r <= cfg_det.r_max) * plos


def gen_detections(t, keys, X, tau, alive, F_sense, extent, cfg_det, max_det):
    """X (N,3) poses, tau (Mt,4) targets, alive (Mt,). Returns Dets."""
    n, mt = X.shape[0], tau.shape[0]
    sig_b = jnp.deg2rad(cfg_det.sigma_beta_deg)
    R_one = jnp.diag(jnp.array([cfg_det.sigma_r ** 2, sig_b ** 2]))

    def per_robot(i, xi):
        k_det = jax.random.fold_in(jax.random.fold_in(keys.det_noise, t), i)
        k_clut = jax.random.fold_in(jax.random.fold_in(keys.clutter, t), i)
        kd1, kd2, kd3 = jax.random.split(k_det, 3)
        kc1, kc2, kc3 = jax.random.split(k_clut, 3)

        # true-target slots 0..Mt-1
        pd = jax.vmap(lambda ty: p_detect(xi[:2], ty[:2], F_sense, extent, cfg_det))(tau)
        hit = alive & (jax.random.uniform(kd1, (mt,)) < pd)
        d = tau[:, :2] - xi[None, :2]
        rng_true = jnp.linalg.norm(d, axis=1) + cfg_det.sigma_r * jax.random.normal(kd2, (mt,))
        bear_true = (jnp.arctan2(d[:, 1], d[:, 0]) - xi[2]
                     + sig_b * jax.random.normal(kd3, (mt,)))

        # clutter tail Mt..D-1, Poisson count uniform in the sensing disc
        cap = max_det - mt
        count = jax.random.poisson(kc1, cfg_det.clutter_rate)
        cmask = jnp.arange(cap) < jnp.minimum(count, cap)
        rng_c = cfg_det.r_max * jnp.sqrt(jax.random.uniform(kc2, (cap,)))
        bear_c = jax.random.uniform(kc3, (cap,), minval=-jnp.pi, maxval=jnp.pi)

        z = jnp.stack([jnp.concatenate([rng_true, rng_c]),
                       jnp.concatenate([bear_true, bear_c])], -1)
        mask = jnp.concatenate([hit, cmask])
        is_clut = jnp.concatenate([jnp.zeros(mt, bool), cmask])
        gt_idx = jnp.concatenate([jnp.arange(mt), jnp.full(cap, -1)]).astype(jnp.int32)
        return z, mask, is_clut, gt_idx, jnp.maximum(count - cap, 0)

    z, mask, is_clut, gt_idx, over = jax.vmap(per_robot)(jnp.arange(n), X)
    R = jnp.broadcast_to(R_one, (n, max_det, 2, 2))
    return Dets(z=z, R=R, mask=mask, is_clutter=is_clut, gt_idx=gt_idx, overflow=over)
