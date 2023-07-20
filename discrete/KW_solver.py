from helpers import DistributionController

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
    
    def __init__(self,stepdata, n, s, l0, l1, th0, th1, th, dist):
        self.stepdata = stepdata
        self.n = n
        self.s = s
        self.l0 = l0
        self.l1 = l1
        self.th0 = th0
        self.th1 = th1
        self.th = th
        self.dist = dist
    
    def back_step_int(self):
        
        def incorp(x):
            if not self.stepdata.laststep:
                if (
                    x >= self.stepdata.frm 
                    and 
                    x - self.stepdata.frm + 1 <= self.stepdata.length
                    ):
                    return(self.stepdata.val[x - self.stepdata.frm + 1])
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
                sum = sum + incorp(self.s + self.k) * d(self.n, self.s, self.k)
                if(sum == sumold):
                    break
                k = k + 1
                
            return sum
                
        
        
                
        