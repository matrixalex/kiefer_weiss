import numpy as np
from scipy.stats import binom
from utils.core import BaseDistributionHandler



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
    
    @staticmethod
    def ubound(n, l0, th0, th, size = 1):
        return np.floor((-np.log(l0) - n * size * np.log((1 - th0)/(1 - th)))/(np.log(th0/th * (1 - th)/(1 - th0))))

    @staticmethod
    def lbound(n, l1, th1, th, size = 1):
        return max(0, np.ceil((-np.log(l1) - n * size * np.log((1 - th1)/(1 - th)))/(np.log(th1/th * (1 - th)/(1 - th1)))))

    @staticmethod
    def hbound(l0, l1, th0, th1, th, size = 1):
        return np.floor((np.log(l1)/np.log(th1/th * (1 - th)/(1 - th1)) - np.log(l0)/np.log(th0/th * (1 - th)/(1 - th0))) /
                    (np.log((1 - th0)/(1 - th))/np.log(th0/th * (1 - th)/(1 - th0)) -
                    np.log((1 - th1)/(1 - th))/np.log(th1/th * (1 - th)/(1 - th1))
                    ) / size)
