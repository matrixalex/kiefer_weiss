from helpers import DistributionController


def calculate(l0, l1, th0, th1, distribution='binom'):
    dist_helper = DistributionController.get_distribution_handler(distribution)
    p = dist_helper.pmf(4, 5, 0.5, size=1)
    print(p)
    
    
calculate(2, 1, 0.1, 0.2, distribution='nbinom')