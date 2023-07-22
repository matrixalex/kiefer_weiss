import numpy as np
from scipy.optimize import brentq
import utils as ut
from scipy.stats import norm
from scipy.special import erf

def find_first_continuation_interval(l0, l1, th0, th1, th, n):

    def current_loss(s): return min(l0 * ut.d(s, th0, n), l1 * ut.d(s, th1, n))
    def future_loss(s): return ut.d(s, th, n) + ut.L(s, l0, l1, th0, th1, n + 1)
    def to_solve(s): return current_loss(s) - future_loss(s)

    c = ut.calc_center_bound(l0, l1, n, th0, th1)
    a = ut.calc_left_bound(l1, n, th1, th)
    b = ut.calc_right_bound(l0, n, th0, th)

    if (np.sign(to_solve(c)) == np.sign(to_solve(b))):
        print("There is no continuation!")
        return None
    
    left_solution = brentq(to_solve, a, c)
    right_solution = brentq(to_solve, c, b)

    return left_solution, right_solution
    
def step_n_has_continuation(l0, l1, th0, th1, th, n):
    
    def current_loss(s):
        res = min(l0 * ut.d(s,th0, n), l1 * ut.d(s, th1, n))
        return res
    
    def future_loss(s):
        res = ut.d(s, th, n) + ut.L(s, l0, l1, th0, th1, n + 1)
        return res
    
    def to_solve(s):
        res = current_loss(s) - future_loss(s)
        return res
    
    c = ut.calc_center_bound(l0, l1, n, th0, th1)
    b = ut.calc_right_bound(n, l0, th0, th)
    
    if np.sign(to_solve(c)) == np.sign(to_solve(b)):
        return False
    else:
        return True
    
def intgr(spts, vls, t):
    delta = spts[1] - spts[0]
    PhiVec = norm.cdf(spts - t)
    phiVec = norm.pdf(spts - t)
    diffPhi = np.diff(PhiVec)
    diffphi = np.diff(phiVec)
    tmp = sum(vls[:-1] * diffPhi)
    vec = diffPhi * (spts - t)[:-1] + diffphi
    tmp = tmp - sum((np.diff(vls)/delta) * vec)
    return tmp

def calculate_integral(s, stepdata, l0, l1, th0, th1):
    if np.size(s) != 1:
        print("s should be a number here")
    else:
        n = stepdata.n
        size_grid = np.size(stepdata.grid)
        a = stepdata.grid[0]
        b = stepdata.grid[size_grid - 1]
        int3 = intgr(stepdata.grid,stepdata.val,s)
        int1 = l0 * ut.d(s, th0,  n - 1) * (1 - erf((b - s - th0) / np.sqrt(2))) / 2
        int2 = l1 * ut.d(s, th1,  n - 1) * (1 + erf((a - s - th1) / np.sqrt(2))) / 2
        return int1 + int2 + int3 
    
def find_continuation_interval(stepdata, l0, l1, th0, th1, th):
    n = stepdata.n - 1
    
    def current_loss(s):
        res = min(l0 * ut.d(s,th0, n), l1 * ut.d(s, th1, n))
        return res
    
    def future_loss(s):
        res = ut.d(s, th, n) + calculate_integral(s, stepdata, l0, l1, th0, th1)
        return res
    
    def to_solve(s):
        res = current_loss(s) - future_loss(s)
        return res
    
    c = ut.calc_center_bound(l0, l1, n, th0, th1)
    a = ut.calc_left_bound(n, l1, th1, th)
    b = ut.calc_right_bound(n, l0, th0, th)
    
    left_solution = brentq(to_solve, a, c)
    right_solution = brentq(to_solve, c, b)
    
    return np.array([left_solution,  right_solution])

def intgr_trapezoidal(spts, vls):
    x_diff = spts[1:] - spts[:-1]
    y_diff = vls[1:] + vls[:-1]
    trapezes = x_diff * y_diff / 2
    return sum(trapezes)
                  