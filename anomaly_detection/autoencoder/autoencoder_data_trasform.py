import numpy as np
from anomaly_detection.anomaly_utils.utils import get_object_slices


def extract_context_rep(y_true):
    """
    This function extracts non-zero slices along with 2 consecutive slices from each side
    :param y_true:
    :return: current slice with 2 consecutive slices
    """
    slices_representation = []
    [first_slice, last_slice] = get_object_slices(y_true)
    context_window = 2
    for i in range(first_slice+context_window,last_slice-context_window):
        slices_representation.append(np.copy(y_true[:,:,i-context_window:i+context_window+1]))

    return slices_representation


def extract_context_rep_pairs(volume, mask, input_size):
    """
    This function extracts non-zero slices along with 2 consecutive slices from each side
    :param mask:
    :return: current slice with 2 consecutive slices
    """
    slices_representation = []
    [first_slice, last_slice] = get_object_slices(mask)
    context_window = 2
    z_size = int(input_size[2]/2)
    for i in range(first_slice+context_window,last_slice-context_window):
        data = np.empty(input_size)
        data[:,:,:z_size] = np.copy(volume[:, :, i - context_window:i + context_window + 1])
        data[:,:,z_size:] = np.copy(mask[:, :, i - context_window:i + context_window + 1])
        slices_representation.append(data)

    return slices_representation


def extract_single_slice_rep(y_true):
    """
    This function extracts a list of slices
    :param y_true:
    :return: list of slices
    """
    slices_representation = []
    for i in range(0, y_true.shape[2]):
        slices_representation.append(y_true[:,:,i])

    return slices_representation


def pad_to_size(y_true, slice_size):
    truth_size = y_true.shape[0:2]
    if(list(truth_size)==slice_size):
        return y_true, [0,0]
    new_shape = list(y_true.shape)
    new_shape[0:2] = slice_size
    padded = np.zeros(new_shape, dtype=float)
    x_delta = int(np.ceil((slice_size[0]-truth_size[0])/2))
    y_delta = int(np.ceil((slice_size[1]-truth_size[1])/2))
    padded[x_delta:(x_delta+truth_size[0]), y_delta:(y_delta+truth_size[1]),:] = y_true

    return padded, [x_delta, y_delta]


def unpad_to_original_size(mask, delta, origin_size):
    [x_delta, y_delta] = delta

    if(x_delta==0 and y_delta==0):
        return mask

    if(x_delta!=0 and y_delta!=0):
        return mask[x_delta:-(mask.shape[0]-origin_size[0]-x_delta), y_delta:-(mask.shape[1]-origin_size[1]-y_delta)]
    if(x_delta!=0):
        return mask[x_delta:-(mask.shape[0]-origin_size[0]-x_delta), :]
    #y_delta!=0
    return mask[:, y_delta:-(mask.shape[1]-origin_size[1]-y_delta)]