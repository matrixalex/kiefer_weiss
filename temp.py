# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.stats import binom
import numpy as np

def pmf(sample_number,probability):
    
    """ binomial probability mass function"""
    
    success_number = np.arange(0,sample_number+1)
    return binom(success_number,sample_number,probability)
    