from scipy.stats import binom
import numpy as np

size = 1

def quant(prob, n, th):
    return binom.ppf(prob, size * n, th)

def cdf(s, n, th):
    return binom.cdf(s, size * n, th)

def pmf(s, n, th):
    return binom.pmf(s, size * n, th)

def rngen(n, th):
    return np.random.binomial(size, th, n)




