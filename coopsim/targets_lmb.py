"""LMB-lite multi-target back end (pure numpy, per-robot python loop).

Track-oriented multi-Bernoulli with integer labels and a single-best-assignment
update: tracks in descending existence order each claim their best gated
measurement, each measurement at most once, with the claimed measurement's
likelihood standing in for the full association sum. This is an LMB-lite, not a
full delta-GLMB/LMB update: no ranked assignment, no hypothesis enumeration.
Existence follows the standard Bernoulli detection and misdetection posteriors
under that single hypothesis. Runs off the jit path; the spec allows the LMB
baseline on a numpy path.
"""
import numpy as np

H = np.zeros((2, 4))
H[0, 0] = H[1, 1] = 1.0
GATE = 9.21          # chi2(2, 0.99) on squared Mahalanobis


class LMB:
    def __init__(self, robot_id=0, r_birth=0.05, prune=1e-3, cap=64):
        self.tracks = []          # dicts r, m (4,), P (4,4), label int
        self.r_birth = r_birth
        self.prune = prune
        self.cap = cap
        self.next_label = robot_id << 20

    def predict(self, dt, sigma_a, survival):
        """NCV predict; existence decays by survival**dt (matches targets.py)."""
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = sigma_a ** 2
        Q = q * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
        for tr in self.tracks:
            tr["r"] *= survival ** dt
            tr["m"] = F @ tr["m"]
            tr["P"] = F @ tr["P"] @ F.T + Q

    def update(self, z, R, mask, p_d, kappa):
        """Single-best-assignment update. z (D,2), R (D,2,2), mask (D,),
        p_d scalar or (num_tracks,), kappa clutter density.

        Claimed measurement: Kalman update, r' = r p_d q / (kappa + r p_d q)
        with q the claimed likelihood. No claim: r' = r (1-p_d) / (1 - r p_d).
        Unclaimed measurements spawn births at r_birth; prune r < prune, cap by
        descending r."""
        mask = np.asarray(mask, bool)
        z_v = np.asarray(z, np.float64).reshape(-1, 2)[mask]
        R_v = np.asarray(R, np.float64).reshape(-1, 2, 2)[mask]
        pd = np.broadcast_to(np.asarray(p_d, np.float64).reshape(-1),
                             (len(self.tracks),))
        used = np.zeros(len(z_v), bool)
        order = sorted(range(len(self.tracks)),
                       key=lambda k: -self.tracks[k]["r"])
        for k in order:
            tr = self.tracks[k]
            best = None
            for d in range(len(z_v)):
                if used[d]:
                    continue
                S = H @ tr["P"] @ H.T + R_v[d]
                diff = z_v[d] - H @ tr["m"]
                mah = diff @ np.linalg.solve(S, diff)
                if mah > GATE:
                    continue
                q = np.exp(-0.5 * mah) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
                if best is None or q > best[0]:
                    best = (q, d, S, diff)
            r = tr["r"]
            if best is None:
                tr["r"] = r * (1.0 - pd[k]) / (1.0 - r * pd[k])
                continue
            q, d, S, diff = best
            used[d] = True
            K = tr["P"] @ H.T @ np.linalg.inv(S)
            tr["m"] = tr["m"] + K @ diff
            tr["P"] = (np.eye(4) - K @ H) @ tr["P"]
            tr["r"] = r * pd[k] * q / (kappa + r * pd[k] * q)
        for d in np.nonzero(~used)[0]:
            self.tracks.append(dict(r=self.r_birth,
                                    m=np.array([z_v[d, 0], z_v[d, 1], 0.0, 0.0]),
                                    P=np.diag([0.25, 0.25, 1.5**2, 1.5**2]),
                                    label=self.next_label))
            self.next_label += 1
        self.tracks = [t for t in self.tracks if t["r"] > self.prune]
        if len(self.tracks) > self.cap:
            self.tracks.sort(key=lambda t: -t["r"])
            del self.tracks[self.cap:]

    def extract(self, max_labels):
        """Tracks with r > 0.5 -> (est (max_labels,5) [x,y,vx,vy,label], mask)."""
        est = np.zeros((max_labels, 5), np.float32)
        mask = np.zeros(max_labels, bool)
        conf = sorted((t for t in self.tracks if t["r"] > 0.5),
                      key=lambda t: -t["r"])
        for i, tr in enumerate(conf[:max_labels]):
            est[i, :4] = tr["m"]
            est[i, 4] = float(tr["label"])
            mask[i] = True
        return est, mask
