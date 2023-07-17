import math

import numpy as np
from scipy.linalg import inv
from scipy.stats import binom


def pmf(sample_size, prob):
    """Binomial probability mass function."""
    sequence = np.arange(sample_size + 1)
    return binom.pmf(sequence, sample_size, prob)


def horizon_bound(lam0, lam1, th0, th1, th):
    """Horizon estimation under given parameters."""
    y = np.array([np.log(th / th0 / (1 - th) * (1 - th0)), np.log(th / th1 / (1 - th) * (1 - th1))])
    z = np.array([np.log((1 - th) / (1 - th0)), np.log((1 - th) / (1 - th1))])
    a = inv(np.vstack((y, z)))
    b = np.matmul(a, np.array([0, 1]))
    pmf0 = pmf(1, th0)
    pmf1 = pmf(1, th1)

    c = 0
    d = 0
    for i in range(pmf0.size):
        if pmf0[i] < pmf1[i]:
            c += pmf0[i]
        else:
            d += pmf1[i]

    result = max(
        1,
        math.ceil(b[0] * np.log(lam0) + b[1] * np.log(lam1) - (b[0] + b[1]) * (-np.log(1 - c - d)))
    )
    return result
