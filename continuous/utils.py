import numpy as np
from scipy.special import erf

def b_normal(thx):
    return thx ** 2 / 2

def bound(l, tx0, tx1, n):
    return np.log(l) / (tx1 - tx0) + n * (b_normal(tx1) - b_normal(tx0)) / (tx1 - tx0)

def calc_center_bound(l0, l1, n, th0, th1):
    return bound(l0 / l1, th0, th1, n)

def calc_left_bound(l1, n, th1, th):
    return bound(1 / l1, th, th1, n)

def calc_right_bound(l0, n, th0, th):
    return bound(l0, th0, th, n)

def d(s, thx, n):
    return np.exp(thx * s - (n * thx ** 2) / 2)

def L(s, l0, l1, th0, th1, horizon):
    B = horizon * (th0 + th1) / 2 + np.log(l1 / l0) / (th0 - th1)
    rp1 = l0 * d(s, th0, horizon - 1) * (1 - erf((B - s - th0) / np.sqrt(2)))
    rp2 = l1 * d(s, th1, horizon - 1) * (1 + erf((B - s - th1) / np.sqrt(2)))
    return ((rp1 + rp2) / 2)

def h_theoretical(l0, l1, th0, th1, th):
    res = ( ( np.log(l0) / (th - th0) + np.log(l1) / (th1 - th))/
    (
      (b_normal(th1) - b_normal(th)) / (th1 - th) -
      (b_normal(th) - b_normal(th0)) / (th - th0)
    ) )
    return np.floor(res)