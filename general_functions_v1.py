# -*- coding: utf-8 -*-
"""
General Use Functions

Author: Alan Nguyen

Date: 28-05-2020
"""

import numpy as np

def array_summary(array):
    mean = np.mean(array)
    std = np.std(array)
    minimum = np.min(array)
    maximum = np.max(array)
    median = np.median(array)
    
    print("             Count:   {}".format(len(array)))
    print("              Mean:   {}".format(mean))
    print("            Median:   {}".format(median))
    print("Standard Deviation:   {}".format(std))
    print("           Minimum:   {}".format(minimum))
    print("           Maximum:   {}".format(maximum))