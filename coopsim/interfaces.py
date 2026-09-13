"""Architecture interfaces: the ONLY arch-conditional code (paper Sec. II).

A0   channel-blind: no channel factors, no field, scheduler sees no channel state.
A_BL bilevel: inner regime/field estimator emits per-link scalars (delivery prob,
     trust scale) that reweight task factors and bias the scheduler. Added at B9.
A_HY hybrid: inner field update + the Schur pose-information message (P-HY) into the
     outer pose-only system. Added at B9.
A_HYG gauge-projected hybrid: same message restricted to the three gauge generators
     (9 numbers per slot instead of 3N x 3N); the interface-dimension midpoint.
A_HYM marginalize-once hybrid: same interface shape as A_HY, but the message is the
     increment of a running current-pose marginal approximation. Those coordinates
     do not retain a moving trajectory or all shared-field correlations. The exact
     trajectory Gaussian construction is validated by run_message_validation;
     it is not a guarantee for this production arm.
A_J  joint: channel factors enter the joint pose-field system directly.

make_arch bakes the flags into jitted closures; everything downstream is shared code.
"""
from functools import partial

import jax

from coopsim import estimator


ARCHS = ("A0", "A_BL", "A_BLS", "A_BLF", "A_HY", "A_HYG", "A_HYR", "A_HYM",
         "A_HYK", "A_J")


def make_arch(cfg, arch):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch}")
    use_channel = arch == "A_J"
    use_msgs = arch.startswith("A_HY")      # A_HY, A_HYG, A_HYR, A_HYM, A_HYK
    estimate_field = arch == "A_J" and not cfg.estimator.oracle_field
    c = estimator.make_consts(cfg)
    # A_BL variants: /w (A_BL) reweights + delivery sched, /s (A_BLS) schedules
    # only, /f (A_BLF) hard-gates factors under the default schedule
    policy = {"A0": cfg.scheduler.policy, "A_BL": "delivery", "A_BLS": "delivery",
              "A_BLF": cfg.scheduler.policy,
              "A_HY": "logdet", "A_HYG": "logdet", "A_HYR": "logdet", "A_HYM": "logdet",
              "A_HYK": "logdet", "A_J": "logdet"}[arch]
    return dict(
        use_channel=use_channel,
        use_msgs=use_msgs,
        project_gauge=arch == "A_HYG",
        project_rand=arch == "A_HYR",
        estimate_field=estimate_field,
        consts=c,
        gn=jax.jit(partial(estimator.gn_update, c=c, use_channel=use_channel,
                           estimate_field=estimate_field,
                           iters=cfg.estimator.gn_iters, use_msgs=use_msgs)),
        warm=jax.jit(estimator.push_warmup),
        slide=jax.jit(partial(estimator.push_slide, c=c, use_channel=use_channel,
                              estimate_field=estimate_field, use_msgs=use_msgs)),
        sched_policy=policy,
    )
