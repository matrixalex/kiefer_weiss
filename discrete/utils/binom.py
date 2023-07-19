import numpy as np
from scipy.stats import binom

from discrete.utils.core import BaseDistributionHandler


class BinomDistributionHandler(BaseDistributionHandler):
    @staticmethod
    def quant(prob, n, th, size=1):
        """Get p-th quantile of binomial density function."""
        return binom.ppf(prob, size * n, th)

    @staticmethod
    def cdf(s, n, th, size=1):
        """Get value of cumulative binomial density function."""
        return binom.cdf(s, size * n, th)

    @staticmethod
    def pmf(s, n, th, size=1):
        """Get probability mass function of binom dist."""
        return binom.pmf(s, size * n, th)

    @staticmethod
    def rngen(n, th, size=1):
        """Generate random variables from binom dist."""
        return np.random.binomial(size, th, n)
