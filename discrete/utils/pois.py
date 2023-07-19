from scipy.stats import poisson

def pmf(s, n, th):
    res = poisson.pmf(s, n*th)
    return res

def quant(p, n, th):
    res = poisson.ppf(p, n*th)
    return res

def cdf(s, n, th):
    res = poisson.cdf(s,n*th)
    return res

def rngen(n,th):
    res =  poisson.rvs(th,size = n)
    return res



        



