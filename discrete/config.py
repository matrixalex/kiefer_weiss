class DistributionEnum():
    BINOM = 1
    POIS = 2
    NBINOM = 3

def get_pmf(distribution_type):
    if distribution_type == DistributionEnum.BINOM:
        from utils.binom import pmf
    elif distribution_type == DistributionEnum.POIS:
        from utils.pois import pmf
    else:
        from utils.nbinom import pmf
    
    return pmf


    