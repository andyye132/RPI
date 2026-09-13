"""Per-slot link scheduling: conflict graph + greedy MWIS (jit).

Protocol interference model ported from experiments/run_mwis_sched.py: two directed
links conflict if either transmitter is within the conflict radius of the other link's
transmitter or receiver, or they share a receiver. Greedy maximum-weight independent
set under the budget inherits the submodular guarantee on the logdet part of DeltaV.
Weights: rr (round-robin ages), delivery (regime-belief expected delivery), logdet
(pose-information gain, the A-J value). CS-divergence term, Doppler lags, AoI index,
oracle-future, handoff lookahead are later seams (cfg.scheduler.lambda_div stub).
"""
import jax
import jax.numpy as jnp


def link_index(n):
    """Flattened directed-link indexing: l = i*n + j, i != j."""
    ii = jnp.arange(n)[:, None].repeat(n, 1).reshape(-1)
    jj = jnp.arange(n)[None, :].repeat(n, 0).reshape(-1)
    return ii, jj, ii != jj


def conflict_adj(pos, n, r_conflict):
    """(L,L) boolean conflict adjacency over flattened directed links, L = n*n."""
    ii, jj, valid = link_index(n)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    r2 = r_conflict ** 2
    tx_tx = d2[ii[:, None], ii[None, :]] < r2
    tx_rx = d2[ii[:, None], jj[None, :]] < r2
    rx_tx = d2[jj[:, None], ii[None, :]] < r2
    same_rx = jj[:, None] == jj[None, :]
    same_node = (ii[:, None] == ii[None, :]) | (ii[:, None] == jj[None, :]) | \
                (jj[:, None] == ii[None, :])
    adj = tx_tx | tx_rx | rx_tx | same_rx | same_node
    adj = adj & valid[:, None] & valid[None, :]
    return adj & ~jnp.eye(adj.shape[0], dtype=bool)


def greedy_mwis(weights, adj, budget, cand_mask):
    """Greedy MWIS: take the heaviest unblocked candidate, block its neighbors,
    repeat `budget` times. Returns (L,) bool chosen mask."""
    L = weights.shape[0]

    def body(_, carry):
        chosen, blocked = carry
        w = jnp.where(cand_mask & ~blocked & ~chosen, weights, -jnp.inf)
        k = jnp.argmax(w)
        ok = w[k] > -jnp.inf
        chosen = chosen.at[k].set(chosen[k] | ok)
        blocked = blocked | jnp.where(ok, adj[k], False)
        return chosen, blocked

    chosen, _ = jax.lax.fori_loop(0, budget, body,
                                  (jnp.zeros(L, bool), jnp.zeros(L, bool)))
    return chosen


def w_roundrobin(age):
    """age (L,): slots since last attempt per link; oldest first."""
    return age.astype(jnp.float32)


def w_delivery(belief, pr_hat, cfg_ch, age):
    """Regime-belief expected delivery probability, age-broken ties."""
    pr_s = pr_hat[..., None] - jnp.array([0.0, cfg_ch.block_extra_db])
    p_succ = jax.scipy.stats.norm.cdf((pr_s - cfg_ch.decode_dbm) / cfg_ch.sigma_f_db)
    exp_del = (belief * p_succ).sum(-1).reshape(-1)
    return exp_del + 1e-3 * age


def w_logdet(pose_cov_diag, age):
    """Cheap logdet proxy: sum of the two endpoints' position variances (links into
    uncertain robots are worth more), age-broken. pose_cov_diag (N,) trace of each
    robot's position covariance.

    NOTE: harness.run_episode currently passes a uniform vector, so this reduces to
    the round-robin ordering. See the comment there; the published comparisons are
    schedule-matched by design and none of them claims an information-gain schedule."""
    n = pose_cov_diag.shape[0]
    gain = pose_cov_diag[:, None] + pose_cov_diag[None, :]
    return gain.reshape(-1) + 1e-3 * age


def schedule(policy, pos, n, budget, r_conflict, age, extras):
    adj = conflict_adj(pos, n, r_conflict)
    _, _, valid = link_index(n)
    if policy == "rr":
        w = w_roundrobin(age)
    elif policy == "delivery":
        if extras.get("exp_del") is not None:
            w = extras["exp_del"].reshape(-1).astype(jnp.float32) + 1e-3 * age
        else:
            w = w_delivery(extras["belief"], extras["pr_hat"], extras["cfg_ch"], age)
    elif policy == "logdet":
        w = w_logdet(extras["pose_cov_diag"], age)
    else:
        raise ValueError(f"unknown policy {policy}")
    chosen = greedy_mwis(w, adj, budget, valid)
    return chosen.reshape(n, n)
