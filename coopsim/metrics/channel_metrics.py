"""Channel prediction metrics."""
import numpy as np
from scipy.stats import rankdata


def delivery_auc(pred, realized):
    """Rank AUC (Mann-Whitney) of predicted delivery score against realized
    delivery. pred (n,) float, realized (n,) bool. Ties get average ranks.
    Returns 0.5 on degenerate single-class input."""
    pred = np.asarray(pred, np.float64)
    realized = np.asarray(realized, bool)
    n1 = int(realized.sum())
    n0 = realized.size - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = rankdata(pred)
    return float((ranks[realized].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
