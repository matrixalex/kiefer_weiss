from discrete.helpers import DistributionController


def calculate(l0, l1, th0, th1, distribution='binom'):
    dist_helper = DistributionController.get_distribution_handler(distribution)
