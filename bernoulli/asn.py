import numpy as np
from core import modified_kw
from utils import pmf


def average_sample_number(th, cont):
    """average sample number of a test giver theta = th"""
    horizon = np.shape(cont)[0]
    asn = np.full((horizon, horizon + 1), 0.0)

    if (horizon == 1):
        return 1
    
    for n in range(horizon-2,-1,-1):
        x = np.arange(n + 2)

        h = ((asn[n+1])[:n+2]/(n+1)*(n+1-x))[:(n+2)]
        t = ((asn[n+1])[:n+2]/(n+1)*(x))[-(n+2):]

        for k in range(horizon+1):
            if cont[n][k] == 1.0:
                asn[n][k] = (pmf(n+1, th) + h + t)[k]
            else:
                asn[n][k] = 0.0
        
    return (1 + sum(asn[0]))

    