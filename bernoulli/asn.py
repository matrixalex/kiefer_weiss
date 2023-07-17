import numpy as np

from utils import pmf


def average_sample_number(th: float, cont: np.array) -> float:
    """
    Average sample number (ASN) approximate calculation under given theta.

    :param th: float - theta, must be in [0,1]
    :param cont: nparray - boolean matrix from modified_kw
    :return: float - approximate calculation of ASN
    """
    horizon = len(cont)
    asn = np.zeros((horizon, horizon + 1), dtype=np.float64)
    if horizon == 1:
        return 1

    for n in range(horizon - 2, -1, -1):
        # recurrent calculation
        x = np.arange(n + 3)
        x_size = x.size
        asn_n = asn[n + 1][:x_size]  # take values from last step
        asn_n = asn_n / (n + 2)

        h = (asn_n * (n + 2 - x))[:x_size - 1]
        t = (asn_n * x)[-(x_size - 1):]
        pmf0 = pmf(n + 1, th)  # vector of success probabilities from 0 to n + 1 inclusive
        tmp = h + t + pmf0
        cont_vector = cont[n]
        for i in range(n + 2):
            if cont_vector[i]:
                asn[n][i] = tmp[i]

    return 1 + np.sum(asn[0])



