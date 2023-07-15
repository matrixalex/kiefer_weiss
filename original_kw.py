import numpy as np
import scipy.stats
import scipy.optimize
from config import MAX_OPTIMIZATION_COUNT


def negative_average_sample_number(x, test):
    return -1*average_simple_number(x, test)


def original_kw(lam0, lam1, th0, th1, tol=0.0001, horizon=None,):

    def delta(x):
        if horizon == None:
            # TODO: implement Hbound
            pass

        test = modified_kw(horizon, lam0, lam1, th0, th1, x)
        # TODO: in ASN func first argument must be x!!! This is necessary for optimizer
        # TODO: add '-' before func, now func calculates minimum value
        ASN_old = average_sample_number(test, x)

        x_max, objective_value, funcalls = scipy.optimize.golden(negative_average_sample_number, args=(
            test,), brack=(th0, th1), tol=tol, full_output=True, maxiter=MAX_OPTIMIZATION_COUNT)

        return (objective_value - ASN_old)

    x_min, objective_value, func_call = scipy.optimize.golden(delta, brack=(
        th0, th1), tol=tol, full_output=True, maxiter=MAX_OPTIMIZATION_COUNT)

    if horizon == None:
        # TODO: add logic
        pass

    return (x_min, objective_value)
