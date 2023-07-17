from typing import Tuple

import numpy as np
from utils import pmf


def modified_kw(horizon: int, lam0: float, lam1: float, th0: float, th1: float, th: float) -> Tuple[np.array, np.array]:
    """Approximate solution of modified Kiefer-Weiss problem."""
    z0 = pmf(horizon, th0)
    z1 = pmf(horizon, th1)

    lagr = np.full((horizon, horizon + 1), np.nan)
    accept = np.full((horizon, horizon + 1), False, dtype=bool)
    cont = np.full((horizon, horizon + 1), False, dtype=bool)

    lagr[horizon - 1] = np.minimum(lam0 * z0, lam1 * z1)
    accept[horizon - 1] = np.array(lam0 * z0 >= lam1 * z1, dtype=float)
    if horizon > 1:
        for n in range(horizon - 1, 0, -1):
            z0 = pmf(n, th0)
            z1 = pmf(n, th1)
            tmp = np.minimum(lam0 * z0, lam1 * z1)
            x = np.arange(n + 2)
            current_size = z0.size
            h = ((lagr[n])[:n + 2] / (n + 1) * (n + 1 - x))[:(n + 1)]
            t = ((lagr[n])[:n + 2] / (n + 1) * x)[-(n + 1):]
            lagr_p = np.hstack((np.minimum(pmf(n, th) + h + t, tmp), np.zeros(horizon - n)))
            accept[n - 1][:current_size] = lam0 * z0 >= lam1 * z1
            cont[n - 1][:current_size] = lagr_p[:n + 1] < tmp
            lagr[n - 1] = lagr_p

    return cont, accept
