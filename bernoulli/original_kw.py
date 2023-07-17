import time
from typing import Tuple

import numpy as np
from asn import average_sample_number
from config import DEFAULT_OPTIMIZATION_PRECISION_LEVEL
from core import modified_kw
from scipy.optimize import golden
from utils import horizon_bound


def maximize_asn(func, bounds, args=(), step_size=0.001, maxiter=1000):
    """Maximizes asn using step-by-step estimation."""
    start, stop = bounds[0], bounds[1]
    iters = 0
    best_param = None
    best_value = None
    for param in np.arange(start, stop, step_size):
        iters += 1

        value = func(param, *args)

        if best_value is None or value > best_value:
            best_value = value
            best_param = param

        if iters > maxiter:
            break

    return best_param, best_value


def negative_average_sample_number(x, cont):
    """Negative ASN for maximization in optimizer."""
    return -1 * average_sample_number(x, cont)


def original_kw(lam0, lam1, th0, th1, tol=DEFAULT_OPTIMIZATION_PRECISION_LEVEL, horizon=None) -> Tuple[float, float]:
    """
    Approximate Kiefer-Weiss problem solution.

    Returns optimal distribution parameter and average sample number for this value.
    """
    golden_constant = 0.2
    opt2 = th0 + (1 - golden_constant) * (th1 - th0)
    opt1 = th0 + golden_constant * (th1 - th0)
    opt1, opt2 = min(opt1, opt2), max(opt1, opt2)


    def delta(x):
        nonlocal horizon, lam0, lam1, th0, th1, opt1, opt2
        if horizon is None:
            horizon = horizon_bound(lam0, lam1, th0, th1, x)

        cont, accept = modified_kw(horizon, lam0, lam1, th0, th1, x)
        # TODO: in ASN func first argument must be x!!! This is necessary for optimizer
        # TODO: add '-' before func, now func calculates minimum value
        ASN_old = average_sample_number(x, cont)

        x_min, value = maximize_asn(
            average_sample_number,
            bounds=(th0, th1),
            args=(cont,)
        )

        return value - ASN_old
    t = time.time()
    x_min, value, iters = golden(
        delta,
        brack=(opt1, opt2),
        full_output=True,
    )
    return x_min, value
