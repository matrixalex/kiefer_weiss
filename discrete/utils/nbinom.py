from scipy.stats import nbinom

def pmf(x, n, th,size = 1):
    res = nbinom.pmf(x, size * n, 1 / (1 + th))
    return res

def cdf(x, n, th,size = 1):
    res = nbinom.cdf(x, size * n, 1 / (1 + th))
    return res

def rngen(n,th,size = 1):
    res = nbinom.rvs(size, 1 / (1 + th),size =  n)
    return res

def quant(p, n, th,size = 1):
    res = nbinom.ppf(p, size * n,1 / (1 + th))
    return res