from scipy.stats import poisson

from discrete.utils.core import BaseDistributionHandler


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
