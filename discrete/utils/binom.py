from scipy.stats import binom
import numpy as np


def quant(prob, n, th, size=1):
    return binom.ppf(prob, size * n, th)

def cdf(s, n, th, size=1):
    return binom.cdf(s, size * n, th)

def pmf(s, n, th, size=1):
    return binom.pmf(s, size * n, th)

def rngen(n, th, size=1):
    return np.random.binomial(size, th, n)




