import numpy as np
from core import modified_kw
from utils import pmf


def average_sample_number(th, cont):
    """average sample number of a test giver theta = th"""
    horizon = np.shape(cont)[0]
    asn = np.full((horizon, horizon + 1), 0)

    if (horizon == 1):
        return 1
    
    for n in range(horizon - 1, 0, -1):
        x = np.arange(n + 2)

        h = ((asn[n])[:n+2]/(n+1)*(n+1-x))[:(n+1)]
        t = ((asn[n])[:n+2]/(n+1)*(x))[-(n+1):]

        for k in range(horizon + 1):
            if cont[n][k] == 1.0:
                asn[n] = np.hstack((pmf(n, th) + h + t, np.zeros(horizon - n)))
            elif cont[n][k] == 0.0:
                asn[n] = np.zeros(horizon + 1)
            else:
                continue
        
        return (1 + asn[1][1] + asn[1][horizon])

