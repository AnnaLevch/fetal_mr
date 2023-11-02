import math
import numpy as np



def get_object_slices(y_true):
    """
    Calculate first and last non-nan slices
    :param points:
    :return:
    """
    for i in range(0, y_true.shape[2]):
        indices_truth = np.nonzero(y_true[:,:,i]>0)
        if ((len(indices_truth[0])) != 0 ):
            break
    first_slice = i
    for j in reversed(range(0, y_true.shape[2])):
        indices_truth = np.nonzero(y_true[:,:,j]>0)
        if ((len(indices_truth[0])) != 0 ):
            break
    last_slice = j
    return first_slice,last_slice
