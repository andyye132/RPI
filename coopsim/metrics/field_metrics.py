"""Field reconstruction metrics. Fields are log-attenuation in dB per meter
(see world.py): the default IoU threshold of 1.0 dB/m is about 6 dB of
through-loss over a typical 6 m building crossing."""
import numpy as np


def field_iou(F_hat, F_true, thresh=1.0, observed_mask=None):
    """IoU of the thresholded masks {F > thresh}, thresh in dB/m. Restricted to
    observed_mask if given. Returns 1.0 when both masks are empty."""
    a = np.asarray(F_hat) > thresh
    b = np.asarray(F_true) > thresh
    if observed_mask is not None:
        a, b = a[observed_mask], b[observed_mask]
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def field_mae(F_hat, F_true, observed_mask=None):
    """Mean absolute error in dB/m, restricted to observed_mask if given."""
    err = np.abs(np.asarray(F_hat, np.float64) - np.asarray(F_true, np.float64))
    if observed_mask is not None:
        err = err[observed_mask]
    return float(err.mean())
