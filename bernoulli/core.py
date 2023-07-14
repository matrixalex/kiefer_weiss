import numpy as np
from utils import pmf

def modified_kw(horizon,lam0,lam1,th0,th1,th):
    z0 = pmf(horizon,th0)
    z1 = pmf(horizon,th1)

    lagr = np.full((horizon, horizon+1),np.nan)
    accept = np.zeros((horizon, horizon+1),dtype = object)
    cont = np.zeros((horizon, horizon+1), dtype=object)

    lagr[horizon-1] = np.minimum(lam0*z0, lam1*z1)
    accept[horizon-1] = np.array(lam0*z0>=lam1*z1,dtype=float)
    cont[horizon-1] = np.full(horizon+1,0.0)
    if horizon>1:
        for n in range(horizon-1,0,-1):
            z0 = pmf(n,th0)
            z1 = pmf(n,th1)
            tmp = np.minimum(lam0*z0,lam1*z1)
            x = np.arange(n+2)
            
            h = ((lagr[n])[:n+2]/(n+1)*(n+1-x))[:(n+1)]
            t = ((lagr[n])[:n+2]/(n+1)*(x))[-(n+1):]
            
            lagr_p = np.hstack((np.minimum(pmf(n,th)+h+t,tmp),np.zeros(horizon-n)))
            accept[n-1] = np.hstack((lam0*z0>=lam1*z1, np.full(horizon-n,np.nan)))
            cont[n-1] = np.hstack((lagr_p[:n+1]<tmp, np.full(horizon-n,np.nan)))
            lagr[n-1] = lagr_p
            
        return(cont,accept)