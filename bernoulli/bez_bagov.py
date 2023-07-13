# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:23:16 2023

@author: User
"""
from scipy.stats import binom
import numpy as np


def pmf(sample_number,th): 
    """ binomial probability mass function"""
    success_number = np.arange(0,sample_number+1)
    return binom.pmf(success_number,sample_number,th)

H = 5
lam0 = 5
lam1 = 10
th0 = 0.1
th1 = 0.2
th = 0.15

z0 = pmf(H,th0)
z1 = pmf(H,th1)

lagr = np.full((H, H+1),np.nan)
accept = np.zeros((H, H+1),dtype = bool)
cont = np.zeros((H, H+1), dtype=bool)

lagr[H-1] = np.minimum(lam0*z0, lam1*z1)
accept[H-1] = lam0*z0>=lam1*z1

if H>0:
    for n in np.flip(np.arange(H-2)):
        z0 = pmf(n, th0)
        z1= pmf(n, th1)
        lagr_n = np.hstack((np.minimum(lam0*z0, lam1*z1),np.zeros(H-n)))
        x = np.arange(H+1)
        
        h =np.hstack(((lagr[n+1]/(n+2)*(n+2-x))[:(n+2)],np.zeros(H-n-1)))
        t = np.hstack(((lagr[n+1]/(n+2)*(x))[-(n+2):],np.zeros(H-n-1)))
        
        lagr[n] = np.minimum(np.hstack((pmf(n,th),np.zeros(H-n))),lagr_n)
        accept[n] = np.hstack((lam0*z0>=lam1*z1,np.full(H-n,1)))
        cont[n] = lagr[n]<lagr_n
        print(lagr[n],accept[n],cont[n])
    

