from scipy.stats import binom
import numpy as np

def pmf(sample_size, prob):
    """binomial probability mass function."""
    sequence = np.arange(sample_size + 1)
    return binom.pmf(sequence, sample_size, prob)
