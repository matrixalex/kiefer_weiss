# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:38:43 2023

@author: User
"""

from scipy.stats import binom
import numpy as np


def pmf(sample_number,th): 
    """ binomial probability mass function"""
    success_number = np.arange(0,sample_number+1)
    return binom.pmf(success_number,sample_number,th)

def modified_kw(H,lam0,lam1,th0,th1,th):
    z0 = pmf(H,th0)
    z1 = pmf(H,th1)

    lagr = np.full((H, H+1),np.nan)
    accept = np.zeros((H, H+1),dtype = object)
    cont = np.zeros((H, H+1), dtype=object)

    lagr[H-1] = np.minimum(lam0*z0, lam1*z1)
    accept[H-1] = np.array(lam0*z0>=lam1*z1,dtype=float)
    cont[H-1] = np.full(H+1,0.0)
    if H>1:
        for n in range(H-1,0,-1):
            z0 = pmf(n,th0)
            z1 = pmf(n,th1)
            tmp = np.minimum(lam0*z0,lam1*z1)
            x = np.arange(n+2)
            
            h = ((lagr[n])[:n+2]/(n+1)*(n+1-x))[:(n+1)]
            t = ((lagr[n])[:n+2]/(n+1)*(x))[-(n+1):]
            
            lagr_p = np.hstack((np.minimum(pmf(n,th)+h+t,tmp),np.zeros(H-n)))
            accept[n-1] = np.hstack((lam0*z0>=lam1*z1, np.full(H-n,np.nan)))
            cont[n-1] = np.hstack((lagr_p[:n+1]<tmp, np.full(H-n,np.nan)))
            lagr[n-1] = lagr_p
            
        return(cont,accept)
    
