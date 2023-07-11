from scipy.stats import binom
import numpy as np

def pmf(sample_size, prob):
    """binomial probability mass function."""
    sequence = np.arange(sample_size + 1)
    return np.array(binom.pmf(sequence, sample_size, prob))

def modified_kw(horizon, lam0, lam1, theta0, theta1, theta):
    """Kiefer-Weiss modified problem function."""
    lagr = np.full([horizon + 1, horizon + 1], np.nan)
    accept = np.full([horizon + 1, horizon + 1], False)
    cont = np.full([horizon + 1, horizon + 1], False)
    prob0 = pmf(horizon, theta0)
    prob1 = pmf(horizon, theta1)
    lagr[horizon] = np.minimum(lam0 * prob0, lam1 * prob1)
    accept[horizon] = lam0 * prob0 >= lam1 * prob1


    if (horizon > 1):
        for n in range(horizon - 1, 0, -1):
            prob0 = pmf(n, theta0)
            prob1 = pmf(n, theta1)
            
