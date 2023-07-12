# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:22:33 2023

@author: User
"""

from scipy.stats import binom
import numpy as np

def pmf(sample_number,th): 
    """ binomial probability mass function"""
    success_number = np.arange(0,sample_number+1)
    return binom(success_number,sample_number,th)

def modified_kw(horizon,lam0,lam1,th0,th1,th):
    lagr = np.full((horizon+1, horizon+1),np.nan)
    accept = np.zeros((horizon+1, horizon+1), dtype=bool)
    cont = np.zeros((horizon+1, horizon+1), dtype=bool)
    
    z0 = pmf(horizon, th0)
    z1 = pmf(horizon, th1)
    
    lagr[horizon] = np.minimum(lam0*z0, lam1*z1)
    accept[horizon] = lam0*z0>=lam1*z1
    cont[horizon] = np.zeros(horizon, dtype = bool)
    
    if horizon>1:
        for n in np.flip(np.arange(horizon)):
            z0 = pmf(n, th0)
            z1= pmf(n, th1)
            lagr_n = np.minimum(lam0*z0, lam1*z1)
            x = np.arange(n+2)
            
            h = (lagr[n+1]/(n+1)*(n+1-x))[:(n+1)]
            t = (lagr[n+1]/(n+1)*(x))[(-(n+1)):]
            
            lagr[n] = np.minimum(pmf(n, th) + h + t, lagr_n)
            accept[n] = lam0*th0>=lam1*th1
            cont[n] = lagr[n]<lagr_n
            
    return(cont,accept)
            
    
            
            
            
    
   
    