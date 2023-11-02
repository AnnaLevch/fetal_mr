import numpy as np


def list_avg(lst):
    if not lst:
        return float('nan')
    else:
        return np.average(lst)


def list_max(lst):
    if not lst:
        return float('nan')
    else:
        return np.max(lst)