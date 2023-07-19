import numpy as np


def hbound(l0,l1,th0,th1,th):
    res = np.floor(
        (np.log(l0) / (np.log(th) - np.log(th0)) + np.log(l1) /  (np.log(th1) - np.log(th))) /
    ((th1 - th)/(np.log(th1) - np.log(th)) - (th - th0)/(np.log(th) - np.log(th0)))
        )
    return res

def ubound(n,l0,th0,th):
    res = np.floor(
        np.log(l0) / np.log(th / th0) + n * (th - th0) / np.log(th / th0)
        )
    return res

def lbound(n, l1, th1, th):
    res = np.ceil(
        max(
            0,
            -np.log(l1) / np.log(th1 / th) + n * (th1 - th) / np.log(th1 / th)
            )
        )
    return(res)
