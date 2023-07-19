from scipy.stats import poisson
import math

def pmf(s, n, th):
    res = poisson.pmf(s, n, th)
    return res

def quant(p, n, th):
    res = poisson.ppf(p, n, th)
    return res

def cdf(s, n, th):
    res = poisson.cdf(s,n,th)
    return res

def rngen(th,n):
    res =  poisson.rvs(th,size = n)
    return res

def d(n, s, x):
    if n ==1:
        return (1)
    else:
        res = math.comb(x+s,s)* (1 - 1/n)**s/n**x
        return  res
        



