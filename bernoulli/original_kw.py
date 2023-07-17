import scipy.optimize
import scipy.stats
from typing import Tuple

from asn import average_sample_number
from core import modified_kw
from utils import horizon_bound
from config import DEFAULT_OPTIMIZATION_PRECISION_LEVEL, MAX_OPTIMIZATION_COUNT


def negative_average_sample_number(x, cont):
    """Negative ASN for maximization in optimizer."""
    return -1 * average_sample_number(x, cont)


def original_kw(lam0, lam1, th0, th1, tol=DEFAULT_OPTIMIZATION_PRECISION_LEVEL, horizon=None) -> Tuple[float, float]:
    """
    Approximate Kiefer-Weiss problem solution.

    Returns optimal distribution parameter and average sample number for this value.
    """
    def delta(x):
        nonlocal horizon
        if horizon is None:
            horizon = horizon_bound(lam0, lam1, th0, th1, x)

        cont, accept = modified_kw(horizon, lam0, lam1, th0, th1, x)
        # TODO: in ASN func first argument must be x!!! This is necessary for optimizer
        # TODO: add '-' before func, now func calculates minimum value
        ASN_old = average_sample_number(x, cont)

        x_max, objective_value, funcalls = scipy.optimize.golden(
            negative_average_sample_number,
            args=(cont,),
            brack=(th0, th1),
            tol=tol,
            full_output=True,
            maxiter=MAX_OPTIMIZATION_COUNT
        )

        return objective_value - ASN_old

    x_min, objective_value, func_call = scipy.optimize.golden(delta, brack=(
        th0, th1), tol=tol, full_output=True, maxiter=MAX_OPTIMIZATION_COUNT)

    return x_min, objective_value
