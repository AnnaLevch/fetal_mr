import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, binary_fill_holes
from scipy.ndimage.measurements import label


def get_main_connected_component(data):
    """
    Extract main component
    :param data:
    :return: segmentation data
    """
    labeled_array, num_features = label(data)
    i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
    return labeled_array == i


def fill_holes_2d(data):
    """
    Fill holes for each slice separately
    :param data: data
    :return: filled data
    """

    filled_data = np.zeros_like(data).astype(bool)
    for i in range(data.shape[2]):
        slice = data[:,:,i]
        slice_filled = binary_fill_holes(slice).astype(bool)
        filled_data[:,:,i] = slice_filled
    return filled_data


def remove_small_components(mask, min_count):
    """
    Remove small connected components
    :param mask: mask
    :param min_count: minimum number of component voxels
    :return: result with removed small components
    """
    seg_res = np.copy(mask)
    s = generate_binary_structure(2,2)
    labeled_array, num_features = label(mask, structure=s)
    (unique, counts) = np.unique(labeled_array, return_counts=True)
    for i in range(0,len(counts)):
        if counts[i] < min_count:
           seg_res[labeled_array == i] = 0

    return seg_res


def remove_small_components_2D(mask, min_count=5):
    """
    Remove small components in 2D slices
    :param mask: mask
    :return: largest component for ach slice
    """
    seg = np.zeros_like(mask, dtype=np.uint8)
    for i in range(0,mask.shape[2]):
        if len(np.nonzero(mask[:,:,i])[0]) > 0:
            seg[:,:,i] = remove_small_components(mask[:,:,i], min_count)

    return seg


def postprocess_prediction(pred, threshold=0.5, fill_holes=True, connected_component=True, remove_small=False, fill_holes_2D=False):
    """
    postprocessing prediction
    :param pred:
    :param threshold: prediction mask
    :param fill_holes: should fill holes
    :param connected_component: should extract one main component
    :param remove_small: should remove small components in 2D
    :return: postprocessed mask
    """
    pred = pred > threshold

    if(np.count_nonzero(pred)==0):#no nonzero elements
        return pred

    if fill_holes:
        pred = binary_fill_holes(pred)

    if connected_component:
        pred = get_main_connected_component(pred)

    if remove_small:
        pred = remove_small_components_2D(pred)

    if fill_holes_2D:
        pred = fill_holes_2d(pred)

    return pred
