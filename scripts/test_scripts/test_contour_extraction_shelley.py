import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from skimage.filters import sobel_h, sobel_v
import numpy as np
from utils.read_write_data import save_nifti

def get_contour(gt):
    """
    :param gt: A binary image that represents a volumetric segmentation of a slice
    :return: A binary image that represents a contour segmentation of a slice
    """
    magnitudes = np.zeros(gt.shape)
    for i in range(gt.shape[2]):
        if gt.sum() == 0:
            continue
        else:
            dx = sobel_v(gt[ :, :, i])
            dy = sobel_h(gt[ :, :, i])
            magnitudes[ :, :, i] = np.sqrt((dx ** 2) + (dy ** 2))
            magnitudes[ :, :, i][magnitudes[ :, :, i] > 0] = 1
    return magnitudes

if __name__ == "__main__":

    subject_folder = '/home/bella/Phd/tmp/18'
    truth_filename = 'truth.nii'

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)

    y_true_contour = get_contour(y_true)

    truth_contour = swap_to_original_axis(swap_axis, y_true_contour)
    save_nifti(truth_contour, os.path.join(subject_folder, 'truth_shelly_contour.nii.gz'))