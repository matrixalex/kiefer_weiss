from scipy.stats import nbinom
from utils.core import BaseDistributionHandler
import numpy as np


class NBinomDistributionHandler(BaseDistributionHandler):
    @staticmethod
    def pmf(x, n, th, size=1):
        """Get probability mass function of N-binom distribution."""
        res = nbinom.pmf(x, size * n, 1 / (1 + th))
        return res

    @staticmethod
    def cdf(x, n, th, size=1):
        """Get value of cumulative N-binom density function."""
        res = nbinom.cdf(x, size * n, 1 / (1 + th))
        return res

    @staticmethod
    def rngen(n, th, size=1):
        """Generate random variables from N-binom dist."""
        res = nbinom.rvs(size, 1 / (1 + th), size=n)
        return res

    @staticmethod
    def quant(p, n, th, size=1):
        """Get p-th quantile of N-binom density function."""
        res = nbinom.ppf(p, size * n, 1 / (1 + th))
        return res
    
    @staticmethod
    def hbound(l0, l1, th0, th1, th, size = 1):
        res = np.floor(
        (np.log(l0) * np.log(th1/th * (th + 1)/(th1 + 1)) + np.log(l1) * np.log(th/th0 * (th0 + 1)/(th + 1))) /
        (np.log((th1 + 1)/(th + 1)) * np.log(th/th0 * (th0 + 1)/(th + 1))
        - np.log((th + 1)/(th0 + 1)) * np.log(th1/th * (th + 1)/(th1 + 1))) / size
        )
        return res

    @staticmethod
    def lbound(n, l1, th1, th, size = 1):
        res = max(
        np.ceil(
            (np.log(l1) + n * size * np.log((th + 1)/(th1 + 1))) / (np.log((th)/(th1) * (th1 + 1)/(th + 1)))
            ),
        0
        )
        return res

    @staticmethod
    def ubound(n, l0, th0, th, size = 1):
        res = np.floor(
        (np.log(l0) + n * size * np.log((th + 1)/(th0 + 1))) / (np.log((th) / (th0) * (th0 + 1) / (th + 1)))
        )
        return res

