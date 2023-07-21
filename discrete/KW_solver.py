from helpers import DistributionController
from step_data import StepData
from scipy.optimize import golden
import numpy as np

class KieferWeissSolver:
    
    
    
    def __init__(self, l0, l1, th0, th1, dist_type, size=1, th=None, horizon=None):
        self.l0 = l0
        self.l1 = l1
        self.th0 = th0
        self.th1 = th1
        self.dist_class = DistributionController.get_distribution_handler(dist_type)
        self.horizon = horizon
        self.size = size
        self.th = th
        self.cont = None
        self.accept = None
        self.step_helper = StepHelper(self.l0, self.l1, self.th0, self.th1, self.th, self.dist_class)

    def maximize_asn(self, test, step_size=0.01, maxiter=1000):
        """Maximizes asn using step-by-step estimation."""
        start, stop = self.th0, self.th1
        iters = 0
        best_param = None
        best_value = None
        for param in np.arange(start, stop, step_size):
            iters += 1

            value = self.avarage_sample_number(param, test)

            if best_value is None or value > best_value:
                best_value = value
                best_param = param

            if iters > maxiter:
                break

        return best_param, best_value
    
    def solve_original(self, horizon):
        golden_constant = 0.2
        opt2 = self.th0 + (1 - golden_constant) * (self.th1 - self.th0)
        opt1 = self.th0 + golden_constant * (self.th1 - self.th0)
        opt1, opt2 = min(opt1, opt2), max(opt1, opt2)

        def delta(x):
            nonlocal horizon, opt1, opt2
            if horizon is None:
                horizon = self.dist_class.hbound(self.lam0, self.lam1, self.th0, self.th1, x)

            test = self.modified_kw(x, horizon)
            # TODO: in ASN func first argument must be x!!! This is necessary for optimizer
            # TODO: add '-' before func, now func calculates minimum value
            ASN_old = self.average_sample_number(x, test)

            x_min, value = self.maximize_asn(test)

            return value - ASN_old
        x_min, value, iters = golden(
            delta,
            brack=(opt1, opt2),
            full_output=True,
        )
        return x_min, value
    
    def solve_modified(self, th, horizon = None):
        test = {}
        if self.th1 > th > self.th0:
            if self.horizon is None:
                h = self.dist_class.hbound(self.l0, self.l1, self.th0, self.th1, th, self.size)
            else:
                h = horizon

            while (
                    self.dist_class.lbound(h, self.l1, self.th1, th, self.size) >
                    self.dist_class.ubound(h, self.l0, self.th0, th, self.size)
            ):
                h -= 1


            stepdata = StepData(
                h=h+1,
                accept_at=self.dist_class.ubound(h + 1, self.l0, self.th0, th, self.size),
                last_step=True
            )
            test[h+1] = stepdata

            while True:
                nocont = True
                for i in np.arange(
                        self.dist_class.lbound(h, self.l1, self.th1, th, self.size),
                        self.dist_class.ubound(h, self.l0, self.th0, th, self.size) + 1
                ):
                    if self.step_helper.step_effect(stepdata, h, i) < 0:
                        nocont = False
                        a = i
                        break

                if nocont:
                    h -= 1
                    if h == 0:
                        raise Exception("Stop after 1 observation")
                    test = {}
                    stepdata = StepData(
                        h=h + 1,
                        accept_at=self.dist_class.ubound(h, self.l0 / self.l1, self.th0, th, self.size),
                        last_step=True
                    )
                    test[h+1] = stepdata
                    continue

                s = self.dist_class.ubound(h, self.l0, self.th0, th, self.size)
                while self.step_helper.step_effect(stepdata, h, s) >= 0:
                    s -= 1

                b = s
                stepdata = self.fill_in(stepdata, a, b, h)
                test[h] = stepdata
                if h == 1:
                    break
                h -= 1

        return test

    def average_sample_number(self):
        pass
    
    def operating_characteristics(self, test):
        h = len(test)
        stepdata = t[h]
        h = h - 1
        
        while h >= 1:
            for i in np.arange(t[h].frm, t[h].frm + t[h].length):
                t[h].val[int(i - t[h].frm)] = self.step_helper.back_step_int_oc(stepdata, i)
            stepdata = t[h]
            h -= 1 
                
        
        res =  self.step_helper.back_step_int_oc(stepdata, 0)
        
        return res
        


    def fill_in(self, stepdata, a, b, h):
        new_data = StepData(h=h, frm=a, length=int(b - a + 1), val=np.zeros(int(b - a + 1)), last_step=False)
        for i in range(int(new_data.frm), int(new_data.frm + new_data.length)):
            val1 = self.step_helper.back_step_int(stepdata, h + 1, i) + self.dist_class.pmf(i, h, self.th, self.size)
            val2 = self.l0 * self.dist_class.pmf(i, h, self.th0)
            val3 = self.l1 * self.dist_class.pmf(i, h, self.th1)
            new_data.val[int(i - new_data.frm)] = min(val1, val2, val3)
        new_data.accept_at = self.dist_class.ubound(h, self.l0 / self.l1, self.th0, self.th1)+1
        return new_data

    def calculate_results(self):
        self.solve_original()
        self.solve_modified()
        
        self.avarage_sample_number()
        self.operating_characteristics()
        
        res = {}
        
        return res


class StepHelper:
    
    def __init__(self, l0, l1, th0, th1, th, dist):
        self.l0 = l0
        self.l1 = l1
        self.th0 = th0
        self.th1 = th1
        self.th = th
        self.dist = dist
    
    def back_step_int(self, stepdata, n, s):
        
        def incorp(x):
            if not stepdata.last_step:
                if (
                    x >= stepdata.frm 
                    and 
                    x - stepdata.frm + 1 <= stepdata.length
                    ):
                    return(stepdata.val[int(x - stepdata.frm)])
                else:
                    return(
                        min(
                        self.l0 * self.dist.pmf(x, n, self.th0),
                               self.l1 * self.dist.pmf(x, n, self.th1)
                               )
                        )
            else:
                return(
                    min(
                        self.l0 * self.dist.pmf(x, n, self.th0),
                        self.l1 * self.dist.pmf(x, n, self.th1)
                        )
                    )
            
        sum = 0
        k = 0
           
        while True:
            sumold = sum
            sum = sum + incorp(s + k) * self.dist.d(n, s, k)
            if abs(sum - sumold) < 0.000000000001:
                break
            k = k + 1
            
        return sum
     
    def back_step_int_oc(self, stepdata, s):
        
        if not stepdata.last_step:
            sum = self.dist.cdf(
                min(
                    stepdata.frm - 1 - s,
                    stepdata.accept_at - s
                    ),
                1,
                self.th
                )
            
            for k in np.arange(stepdata.frm - s,
                               stepdata.frm - s + stepdata.length, 1):
                if k >= 0:
                    sum = (sum + 
                    stepdata.val[int(k + s - stepdata.frm)] * self.dist.pmf(k, 1, self.th))
                    
        else:
            sum = self.dist.cdf(stepdata.accept_at - s, 1, self.th)
        
        return sum
    
    def back_step_int_asn(self, stepdata, s):
        if not stepdata.last_step:
            k = 0
            sum = 0
            while True:
                if(k + s >= stepdata.frm and
                   k + s <= stepdata.frm + stepdata.length - 1):
                    
                    sum = (sum +
                           stepdata.val[k + s - stepdata.frm + 1] * self.dist.pmf(k, 1, self.th))
                
                if k + s >= stepdata.frm + stepdata.length - 1:
                    break
                k = k + 1
            
        else:
            sum = 0
        
        return sum
    
    def step_effect(self, stepdata, n, s):
        res = (
        self.back_step_int(stepdata, n + 1, s) +
        self.dist.pmf(s, n, self.th) -
        min(
            self.l0 * self.dist.pmf(s, n, self.th0),
            self.l1 * self.dist.pmf(s, n, self.th1)
           )
             )
        return res
    

solver = KieferWeissSolver(49, 59, 0.2, 0.4, th=0.3, dist_type='binom')

t = solver.solve_modified()

d = solver.operating_characteristics(t)

print(t[1],d)
