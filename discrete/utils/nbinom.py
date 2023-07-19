from scipy.stats import nbinom


def pmf(x, n, th, size=1):
    """Get probability mass function of N-binom distribution."""
    res = nbinom.pmf(x, size * n, 1 / (1 + th))
    return res


def cdf(x, n, th, size=1):
    """Get value of cumulative N-binom density function."""
    res = nbinom.cdf(x, size * n, 1 / (1 + th))
    return res


def rngen(n, th, size=1):
    """Generate random variables from N-binom dist."""
    res = nbinom.rvs(size, 1 / (1 + th), size=n)
    return res


def quant(p, n, th, size=1):
    """Get p-th quantile of N-binom density function."""
    res = nbinom.ppf(p, size * n, 1 / (1 + th))
    return res
