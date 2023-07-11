from scipy.stats import binom
import numpy as np

def pmf(sampleSize, prob):

    #binomial probability mass function

    sequence = np.arange(sampleSize + 1)
    return binom.pmf(sequence, sampleSize, prob)
