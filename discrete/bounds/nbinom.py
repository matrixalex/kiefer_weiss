import numpy as np

size = 1

def hbound(l0, l1, th0, th1, th):
    res = np.floor(
    (np.log(l0) * np.log(th1/th * (th + 1)/(th1 + 1)) + np.log(l1) * np.log(th/th0 * (th0 + 1)/(th + 1))) /
    (np.log((th1 + 1)/(th + 1)) * np.log(th/th0 * (th0 + 1)/(th + 1))
      - np.log((th + 1)/(th0 + 1)) * np.log(th1/th * (th + 1)/(th1 + 1))) / size
    )
    return res

def lbound(n, l1, th1, th):
    res = max(
    np.ceil(
        (np.log(l1) + n * size * np.log((th + 1)/(th1 + 1))) / (np.log((th)/(th1) * (th1 + 1)/(th + 1)))
        ),
     0
     )
    return res

def ubound(n, l0, th0, th):
    res = np.floor(
    (np.log(l0) + n * size * np.log((th + 1)/(th0 + 1))) / (np.log((th) / (th0) * (th0 + 1) / (th + 1)))
    )
    return res
