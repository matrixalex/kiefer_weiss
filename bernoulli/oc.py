import numpy as np
from utils import pmf


def operating_characteristic(cont,accept,th):
    horizon = np.shape(cont)[0]
    a = np.zeros((horizon, horizon+1),dtype = float)

        
    for i in range(0,horizon+1):
        if accept[horizon-1][i] == 1:
            a[horizon-1][i] = pmf(horizon,th)[i]
        else:a[horizon-1][i] = 0
        


    if horizon>1:
        for n in range(horizon-1,0,-1):
            x = np.arange(n+2)
                    
            h = ((a[n])[:n+2]/(n+1)*(n+1-x))[:(n+1)]
            t = ((a[n])[:n+2]/(n+1)*(x))[-(n+1):]

            for i in range(0,horizon+1):
                if cont[n-1][i] == 1:
                    a[n-1][i] = (h+t)[i]
                else:
                    if accept[n-1][i] == 1:
                        a[n-1][i] = pmf(n,th)[i]
                    else:a[n-1][i] = 0 
                    
    return sum(a[0])
