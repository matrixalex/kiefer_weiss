import support as sup
from step_data import StepData
import utils as ut
import numpy as np

def modified_kw(l0, l1, th0, th1, th, horizon = 0, precision=0.01):
    if th <= th0 or th >= th1:
        raise Exception("Only case th0 < th < th1 is supported!")
    
    theor_h = ut.h_theoretical(l0, l1, th0, th1, th)

    if horizon > theor_h or horizon == 0:
        horizon = theor_h

    test = {}
    n = horizon - 1
    while True:
        if sup.step_n_has_continuation(l0, l1, th0, th1, th, n = n):
            break
        else:
            n -= 1
        
        if n == 1:
            return test
        
    acceptance_constant = ut.bound(l0/l1, th0, th1, n + 1)
    test[n + 1] = StepData(n = n + 1, grid = acceptance_constant, val = None)

    continuation_interval = sup.find_first_continuation_interval(l0, l1, th0, th1, th, n)
    a = continuation_interval[0]
    b = continuation_interval[1]
    nint = int(np.ceil((b - a) / precision))
    grid = np.linspace(start = a, stop = b, num = nint + 1)
    val = ut.d(grid, th, n) + ut.L(grid, l0, l1, th0, th1, n + 1)
    step_data = StepData(n = n, grid = grid, val = val)
    test[n] = step_data

    while True:
        continuation_interval = sup.find_continuation_interval(step_data, l0, l1, th0, th1, th)
        n = step_data.n - 1
    
        a = continuation_interval[0]
        b = continuation_interval[1]
        nint = int(np.ceil((b - a) / precision))
        grid = np.linspace(start = a, stop = b, num = nint + 1)
        val = ut.d(grid, th, n) + np.vectorize(sup.calculate_integral)(grid, step_data, l0, l1, th0, th1)
        step_data = StepData(n = n, grid = grid, val = val)
        test[n] = step_data
        if n == 1:
            break
    
    return test

thx = 0.4
tx0 = 0.25
tx1 = 0.53
n = 10
l = 23
l0 = 15
l1 = 18
th0 = 0.12
th1 = 0.36
th = 0.27
s = 7
horizon = 16

print(modified_kw(l0, l1, th0, th1, th)[3])
# print(np.ceil(3.7878))