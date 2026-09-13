"""Detection fusion for AETHER v3.

Payload = raw detections (default): a delivered packet carries the sender's world-frame
(z, R) detections, which enter the receiver's PHD update alongside its own. One
conversion path (dets_to_world) serves local and received detections, so frame handling
cannot diverge. Provenance dedup lives in the harness (detections are delivered at most
once per link per slot in the gate runs; multi-hop forwarding is a later seam, as are
component-level AA payloads). The SOTA comparison rows live here too: ci_fuse_tracks
(track-level covariance intersection, the send-tracks alternative) and kla_fuse
(simplified consensus KLA), both numpy and called per robot per slot outside jit.
"""
import jax
import jax.numpy as jnp
import numpy as np

from coopsim.targets import PHD


def dets_to_world(z, R, mask, pose):
    """Body-frame (range, bearing) -> world (x, y) with first-order covariance.
    z (D,2), R (D,2,2), pose (3,). Returns (z_xy (D,2), R_xy (D,2,2), mask)."""
    a = pose[2] + z[:, 1]
    c, s = jnp.cos(a), jnp.sin(a)
    z_xy = pose[None, :2] + z[:, :1] * jnp.stack([c, s], -1)
    G = jnp.stack([jnp.stack([c, -z[:, 0] * s], -1),
                   jnp.stack([s, z[:, 0] * c], -1)], 1)          # (D,2,2)
    R_xy = G @ R @ G.transpose(0, 2, 1)
    return z_xy, R_xy, mask


def _fit(a, target, fill):
    k = a.shape[0]
    if k >= target:
        return a[:target]
    pad = jnp.full((target - k,) + a.shape[1:], fill, a.dtype)
    return jnp.concatenate([a, pad])


def fuse_sets(z_local, R_local, mask_local, z_recv, R_recv, mask_recv, max_fused):
    """Concatenate local + received world-frame detection sets to a FIXED shape
    (pad or truncate to max_fused). Received sets are pre-flattened. Truncation drops
    the tail of the received set (local detections always survive); the count is
    returned so the harness can log drops. Padded R rows stay identity so downstream
    inverses remain finite."""
    z = _fit(jnp.concatenate([z_local, z_recv]), max_fused, 0.0)
    R = _fit(jnp.concatenate([R_local, R_recv]), max_fused, 0.0)
    R = jnp.where(jnp.arange(max_fused)[:, None, None]
                  >= (z_local.shape[0] + z_recv.shape[0]),
                  jnp.eye(2)[None], R)
    mask = _fit(jnp.concatenate([mask_local, mask_recv]), max_fused, False)
    dropped = jnp.maximum(mask_local.sum() + mask_recv.sum() - mask.sum(), 0)
    return z, R, mask, dropped


def ci_fuse_tracks(local_est, local_mask, local_P, recv_est_list, recv_mask_list,
                   recv_P_list, thresh=2.0):
    """Track-level covariance intersection (Julier-Uhlmann send-tracks row).

    local_est (ML,5) rows [x,y,vx,vy,label], local_mask (ML,), local_P (ML,4,4);
    recv_*_list hold the same shapes per sender. Each received track is greedily
    associated to the nearest local track by position distance < thresh and fused
    by CI, omega picked from a coarse grid over [0.1, 0.9] minimizing det(P_f),
    P_f = inv(w inv(P1) + (1-w) inv(P2)). Unassociated received tracks fill free
    slots with the sender's label. Returns (est, mask)."""
    est = np.array(local_est, np.float64)
    mask = np.array(local_mask, bool)
    P = np.array(local_P, np.float64)
    omegas = np.linspace(0.1, 0.9, 9)
    for r_est, r_mask, r_P in zip(recv_est_list, recv_mask_list, recv_P_list):
        r_est = np.asarray(r_est, np.float64)
        r_P = np.asarray(r_P, np.float64)
        for k in np.nonzero(np.asarray(r_mask, bool))[0]:
            d = np.where(mask, np.linalg.norm(est[:, :2] - r_est[k, :2], axis=1),
                         np.inf)
            j = int(np.argmin(d))
            if d[j] < thresh:
                I1, I2 = np.linalg.inv(P[j]), np.linalg.inv(r_P[k])
                best = None
                for w in omegas:
                    Pf = np.linalg.inv(w * I1 + (1.0 - w) * I2)
                    detf = np.linalg.det(Pf)
                    if best is None or detf < best[0]:
                        best = (detf, w, Pf)
                _, w, Pf = best
                est[j, :4] = Pf @ (w * (I1 @ est[j, :4])
                                   + (1.0 - w) * (I2 @ r_est[k, :4]))
                P[j] = Pf
            else:
                free = np.nonzero(~mask)[0]
                if len(free):
                    s = free[0]
                    est[s], P[s], mask[s] = r_est[k], r_P[k], True
    return est, mask


def kla_fuse(phd_local, phd_recv_list, iters=3):
    """Simplified consensus Kullback-Leibler averaging on GM intensities
    (Battistelli-Chisci), first-order approximation.

    Per consensus iteration and per received PHD, components are greedily
    matched by squared Mahalanobis distance < 4 under the summed covariances.
    A matched pair is replaced by its weighted geometric mean, approximated as
    the CI fusion of the two Gaussians with omega = 0.5 and fused weight
    w_f = sqrt(w1 w2) (the standard first-order KLA approximation for matched
    components). Unmatched components on either side keep their parameters with
    weight scaled by 0.5, the flattening a geometric mean applies to mass with
    no counterpart. Received PHDs are held fixed across iterations (one
    exchange per slot, no re-broadcast). Operates on targets.PHD fields as
    numpy; returns a PHD with at most J components (top-J by weight, J from
    the local PHD shape), keeping the local next_label."""
    J = phd_local.w.shape[0]
    w_l, act_l = np.asarray(phd_local.w), np.asarray(phd_local.active)
    m_l, P_l = np.asarray(phd_local.m), np.asarray(phd_local.P)
    lab_l = np.asarray(phd_local.label)
    comps = [dict(w=float(w_l[i]), m=np.array(m_l[i], np.float64),
                  P=np.array(P_l[i], np.float64), label=int(lab_l[i]))
             for i in np.nonzero(act_l & (w_l > 0))[0]]
    recv = []
    for p in phd_recv_list:
        keep = np.nonzero(np.asarray(p.active) & (np.asarray(p.w) > 0))[0]
        recv.append((np.asarray(p.w, np.float64)[keep],
                     np.asarray(p.m, np.float64)[keep],
                     np.asarray(p.P, np.float64)[keep],
                     np.asarray(p.label)[keep]))
    for _ in range(iters):
        for rw, rm, rP, rlab in recv:
            taken = np.zeros(len(rw), bool)
            out = []
            for c in sorted(comps, key=lambda c: -c["w"]):
                j, mah_j = -1, np.inf
                if len(rw):
                    Ps = rP + c["P"][None]
                    diff = rm - c["m"][None]
                    mah = np.einsum("ki,ki->k", diff,
                                    np.linalg.solve(Ps, diff[..., None])[..., 0])
                    mah = np.where(taken, np.inf, mah)
                    j = int(np.argmin(mah))
                    mah_j = mah[j]
                if mah_j >= 4.0:
                    out.append(dict(c, w=0.5 * c["w"]))
                    continue
                taken[j] = True
                I1, I2 = np.linalg.inv(c["P"]), np.linalg.inv(rP[j])
                Pf = np.linalg.inv(0.5 * I1 + 0.5 * I2)
                mf = Pf @ (0.5 * (I1 @ c["m"]) + 0.5 * (I2 @ rm[j]))
                out.append(dict(w=float(np.sqrt(c["w"] * rw[j])), m=mf, P=Pf,
                                label=c["label"]))
            for j in np.nonzero(~taken)[0]:
                out.append(dict(w=0.5 * float(rw[j]), m=rm[j].copy(),
                                P=rP[j].copy(), label=int(rlab[j])))
            comps = out
    comps.sort(key=lambda c: -c["w"])
    w = np.zeros(J, np.float32)
    m = np.zeros((J, 4), np.float32)
    P = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    lab = np.zeros(J, np.int32)
    act = np.zeros(J, bool)
    for i, c in enumerate(comps[:J]):
        w[i], m[i], P[i], lab[i], act[i] = c["w"], c["m"], c["P"], c["label"], True
    return PHD(w=w, m=m, P=P, label=lab, active=act,
               next_label=np.int32(phd_local.next_label))
