import scipy.optimize
import scipy.stats
from typing import Tuple
from scipy.optimize import differential_evolution
from asn import average_sample_number
from core import modified_kw
from utils import horizon_bound
from config import DEFAULT_OPTIMIZATION_PRECISION_LEVEL, MAX_OPTIMIZATION_COUNT
import time

def negative_average_sample_number(x, cont):
    """Negative ASN for maximization in optimizer."""
    return -1 * average_sample_number(x, cont)


def original_kw(lam0, lam1, th0, th1, tol=DEFAULT_OPTIMIZATION_PRECISION_LEVEL, horizon=None) -> Tuple[float, float]:
    """
    Approximate Kiefer-Weiss problem solution.

    Returns optimal distribution parameter and average sample number for this value.
    """

    def delta(x):
        nonlocal horizon, lam0, lam1, th0, th1
        if horizon is None:
            horizon = horizon_bound(lam0, lam1, th0, th1, x)
        print(horizon, lam0, lam1, th0, th1, x)
        horizon = 102
        print('Modified_kw started')
        start = time.time()
        cont, accept = modified_kw(horizon, lam0, lam1, th0, th1, x)
        print('Modified_kw finished with time', time.time()-start)


        print("ASN started")

        start = time.time()

        ASN_old = average_sample_number(x, cont)

        print("ASN finished with time", time.time() - start)

        #creating bounds
        bounds = [(th0, th1)]

        print("Optimizer differential_evolution started")
        start = time.time()
        result = differential_evolution(negative_average_sample_number, args=(cont,), bounds=bounds)

        #optimizer returns dict, so we need to unpack it
        objective_value = result['fun']
        print("Optimizer differential_evolution finished wotk with time", time.time() - start)
        print("Result", result)


        return objective_value - ASN_old


    print("Golden started")
    start = time.time()
    x_min, objective_value, func_call = scipy.optimize.golden(delta, brack=(
        th0+tol, th1-tol), tol=tol, full_output=True, maxiter=MAX_OPTIMIZATION_COUNT)
    print("Golden ended work with time", time.time() - start)
    return x_min, objective_value

print(original_kw(157.696751972207, 193.349705609267, 0.05, 0.15))