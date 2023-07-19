from scipy.stats import poisson


def pmf(s, n, th):
    """Get probability mass function of Poisson dist."""
    res = poisson.pmf(s, n * th)
    return res


def quant(p, n, th):
    """Get p-th quantile of Poisson density function."""
    res = poisson.ppf(p, n * th)
    return res


def cdf(s, n, th):
    """Get value of cumulative Poisson density function."""
    res = poisson.cdf(s, n * th)
    return res


def rngen(n, th):
    """Generate random variables from Poisson dist."""
    res = poisson.rvs(th, size=n)
    return res
