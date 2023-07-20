from helpers import DistributionController
from step_data import StepData
import numpy as np

class KieferWeissSolver:
    
    
    
    def __init__(self, l0, l1, th0, th1, dist_type, horizon, th, cont, accept):
        self.l0 = l0
        self.l1 = l1
        self.th0 = th1
        self.th1 = th1
        self.dist_class = DistributionController.get_distribution_handler(dist_type)
        self.horizon = None
        self.th = None
        self.cont = None
        self.accept = None
        pass
    
    def solve_original(self):
        pass
    
    def solve_modified(self):
        pass
    
    def avarage_sample_number(self):
        pass
    
    def operating_characteristics(self):
        pass
    
    def calculate_results(self):
        self.solve_original()
        self.solve_modified()
        
        self.avarage_sample_number()
        self.operating_characteristics()
        
        res = {}
        
        return res
    
class StepHelper:
    
    
    
    def __init__(self, n, s, l0, l1, th0, th1, th, dist):
        self.n = n
        self.s = s
        self.l0 = l0
        self.l1 = l1
        self.th0 = th0
        self.th1 = th1
        self.th = th
        self.dist = dist
    
    def back_step_int(self, stepdata):
        
        def incorp(x):
            if not stepdata.last_step:
                if (
                    x >= stepdata.frm 
                    and 
                    x - stepdata.frm + 1 <= stepdata.length
                    ):
                    return(stepdata.val[x - stepdata.frm + 1])
                else:
                    return(
                        min(
                        self.l0 * self.dist.pmf(x, self.n, self.th0), 
                               self.l1 * self.dist.pmf(x, self.n, self.th1)
                               )
                        )
            else:
                return(
                    min(
                        self.l0 * self.dist.pmf(x, self.n, self.th0), 
                        self.l1 * self.dist.pmf(x, self.n, self.th1)
                        )
                    )
            
        sum = 0
        k = 0
           
        while True:
            sumold = sum
            sum = sum + incorp(self.s + k) * self.dist.d(self.n, self.s, k)
            if(sum == sumold):
                break
            k = k + 1
            
        return sum
     
    def back_step_int_oc(self, stepdata):
        
        if not stepdata.last_step:
            sum = self.dist.cdf(
                min(
                    stepdata.frm - 1 - self.s,
                    stepdata.acceptAt - self.s
                    )
                )
            
            for k in np.arange(stepdata.frm - self.s,
                               stepdata.frm - self.s + stepdata.length, 1):
                if k >= 0:
                    sum = (sum + 
                    stepdata.val[k + self.s - stepdata.frm + 1] * self.dist.pmf(k, 1, self.th))
                    
        else:
            sum = self.dist.cdf(stepdata.acceptAt - self.s, 1, self.th)
        
        return sum
    
    def back_step_int_asn(self, stepdata):
        if not stepdata.last_step:
            k = 0
            sum = 0
            while True:
                if(k + self.s >= stepdata.frm and 
                   k + self.s <= stepdata.frm + stepdata.length - 1):
                    
                    sum = (sum +
                           stepdata.val[k + self.s - stepdata.frm + 1] * self.dist.pmf(k, 1, self.th))
                
                if k + self.s >= stepdata.frm + stepdata.length - 1:
                    break
                k = k + 1
            
        else:
            sum = 0
        
        return sum
    
    def step_effect(self, stepdata):
        res = (
        self.back_step_int(stepdata, self.n + 1, self.s, self.l0, self.l1, self.th0, self.th1) +
        self.dist.pmf(self.s, self.n, self.th) - 
        min(
            self.l0 * self.dist.pmf(self.s, self.n, self.th0), 
            self.l1 * self.dist.pmf(self.s, self.n, self.th1)
           )
             )
        return res
    
data = StepData(False, 2, 5, 6, np.array([1,2]))

a = StepHelper(5,4,3,3,0.2,0.5,0.3,DistributionController.get_distribution_handler('binom'))
res = a.back_step_int(data)
