from scipy.stats import poisson
import numpy as np
import math

from utils.core import BaseDistributionHandler


class PoissonDistributionHandler(BaseDistributionHandler):
    @staticmethod
    def pmf(s, n, th, size=1):
        """Get probability mass function of Poisson dist."""
        res = poisson.pmf(s, n * th)
        return res

    @staticmethod
    def quant(p, n, th, size=1):
        """Get p-th quantile of Poisson density function."""
        res = poisson.ppf(p, n * th)
        return res

    @staticmethod
    def cdf(s, n, th, size=1):
        """Get value of cumulative Poisson density function."""
        res = poisson.cdf(s, n * th)
        return res

    @staticmethod
    def rngen(n, th, size=1):
        """Generate random variables from Poisson dist."""
        res = poisson.rvs(th, size=n)
        return res
    
    @staticmethod
    def hbound(l0,l1,th0,th1,th, size = 1):
        res = np.floor(
            (np.log(l0) / (np.log(th) - np.log(th0)) + np.log(l1) /  (np.log(th1) - np.log(th))) /
        ((th1 - th)/(np.log(th1) - np.log(th)) - (th - th0)/(np.log(th) - np.log(th0)))
            )
        return res
    
    @staticmethod
    def ubound(n,l0,th0,th, size = 1):
        res = np.floor(
            np.log(l0) / np.log(th / th0) + n * (th - th0) / np.log(th / th0)
            )
        return res
    
    @staticmethod
    def lbound(n, l1, th1, th, size = 1):
        res = np.ceil(
            max(
                0,
                -np.log(l1) / np.log(th1 / th) + n * (th1 - th) / np.log(th1 / th)
                )
            )
        return(res)
    
    @staticmethod
    def d(n, s, x, size = 1):
        if n == 1:
            return 1
        res = math.comb(x + s, s) * ((1 - 1/n)**s)/n**x
        return res
        


