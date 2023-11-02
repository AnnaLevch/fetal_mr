from scipy import ndimage
import numpy as np


def connected_components_data_features(data):
    """
    This function extracts connected componnents and calculates number of connected components and their
    :param data:
    :return:
    """
    components = ndimage.label(data)
    values, counts = np.unique(components[0], return_counts=True)
    return components[0], values, counts


#def connected_components_comparison_features(truth, result):
