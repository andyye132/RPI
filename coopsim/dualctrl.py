"""Dual control of the link schedule: probe the gauge, exploit the routes.

The schedule is the actuator (which <= B of the N(N-1) directed links to activate
this slot) and it serves two objectives at once:

  PROBE    collect gauge information, so map-referenced ETX predictions stay
           correct. The decode bit's Fisher term peaks at p = 1/2, so informative
           links are MARGINAL ones (see guidance.link_weights).
  EXPLOIT  deliver detections to the robots whose tracking they most improve,
           which wants RELIABLE links, p -> 1.

`lam_route` is the knob between them; 0 is pure probing. The exploit term is
scored on detections, the only traffic this simulator actually carries.

Reuses the existing seam: `guidance.greedy_gauge_links` already maximizes the
greedy logdet increment and accepts an additive per-link `bonus`, which is what
this module computes. Same MWIS conflict structure and budget, so no arm gets
extra airtime.
"""
import numpy as np



def detection_value(p_hat, n_det, aoi, eps=1e-9):
    """(L,) exploit value per DIRECTED link, from what is actually worth sending.

    p_hat (N,N) believed delivery probability, n_det (N,) live detections the
    sender holds this slot, aoi (N,N) slots since a delivery from i last reached
    j. Value of scheduling i -> j is the product

        v_ij = p_hat_ij * n_det_i * aoi_ij

    and each factor is necessary. Without p_hat the controller schedules links
    that will not arrive. Without n_det it spends airtime on a sender with
    nothing to say. Without aoi it starves whichever receiver is currently
    worst-informed, since a receiver that just heard from i gains little from
    hearing again. The product is age-of-information weighted by how much
    information there is and whether it will survive the channel; `scheduler.py`
    already names an "AoI index" as a later seam and this fills it.

    All three inputs are observable to the controller: p_hat comes from pr_hat at
    the ESTIMATED field and poses, n_det from the sender's own detector, and aoi
    from its own delivery history. No truth is used.

    Flattening an (N,N) matrix gives l = i*n + j, which is exactly
    scheduler.link_index's convention, so the result lines up elementwise with
    the conflict adjacency and the gauge sensitivities with no reindexing.
    """
    v = np.asarray(p_hat) * np.asarray(n_det)[:, None] * np.asarray(aoi)
    np.fill_diagonal(v, 0.0)
    return (v / (v.max() + eps)).reshape(-1)
