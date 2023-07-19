import numpy as np
from math import floor, ceil

size = 1

def Ubound(n, l0, th0, th):
    return floor((-np.log(l0) - n * size * np.log((1 - th0)/(1 - th)))/(np.log(th0/th * (1 - th)/(1 - th0))))


def Lbound(n, l1, th1, th):
    return max(0, ceil((-np.log(l1) - n * size * np.log((1 - th1)/(1 - th)))/(np.log(th1/th * (1 - th)/(1 - th1)))))

def Hbound(l0, l1, th0, th1, th):
    return floor((np.log(l1)/np.log(th1/th * (1 - th)/(1 - th1)) - np.log(l0)/np.log(th0/th * (1 - th)/(1 - th0))) /
                 (np.log((1 - th0)/(1 - th))/np.log(th0/th * (1 - th)/(1 - th0)) -
                 np.log((1 - th1)/(1 - th))/np.log(th1/th * (1 - th)/(1 - th1))
                 ) / size)
