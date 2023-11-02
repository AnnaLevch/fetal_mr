import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from training.train_functions.metrics import extract_volume_2D_contours_tf, extract_volume_2D_bands_tf
import tensorflow as tf
from utils.read_write_data import save_nifti
import numpy as np


if __name__ == "__main__":
    subject_folder = '/home/bella/Phd/code/code_bella/log/351/output/Placenta_FIESTA_all_test_corrected_tta/test/37/'
    truth_filename = 'truth.nii.gz'
    prediction_filename = 'prediction.nii.gz'

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    y_pred = nib.load(os.path.join(subject_folder, prediction_filename)).get_data()
    y_pred, swap_axis = move_smallest_axis_to_z(y_pred)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #contour extraction
        input4D_truth = tf.cast(tf.expand_dims(y_true,0), tf.int32)
        y_true_contour_tf = extract_volume_2D_contours_tf(input4D_truth, y_true.shape, 3)
        y_true_contour = tf.cast(y_true_contour_tf[0,:,:,:], tf.int32).eval()

        input4D_pred = tf.cast(tf.expand_dims(y_pred,0), tf.int32)
        y_pred_contour_tf = extract_volume_2D_contours_tf(input4D_pred, y_pred.shape, 3)
        y_pred_contour = tf.cast(y_pred_contour_tf[0,:,:,:], tf.int32).eval()

        y_true_band_tf = extract_volume_2D_bands_tf(input4D_truth, y_true.shape, 3)
        y_true_band = tf.cast(y_true_band_tf[0,:,:,:], tf.int32).eval()

        y_pred_band_tf = extract_volume_2D_bands_tf(input4D_pred, y_pred.shape, 3)
        y_pred_band = tf.cast(y_pred_band_tf[0,:,:,:], tf.int32).eval()


    unified = np.zeros_like(y_true_contour)

    # unified[y_true_contour == 1] = 1
    # unified[y_pred_contour == 1] = 2
    # unified[(y_pred_contour == 1) & (y_true_contour == 1)] = 3

    unified[y_true_band == 1] = 1
    unified[y_pred_band == 1] = 2
    unified[y_true_contour == 1] = 3
    unified[y_pred_contour == 1] = 4
    unified[(y_pred_contour == 1) & (y_true_band == 1)] = 5
    unified[(y_pred_band == 1) & (y_true_contour == 1)] = 5

    truth_contour = swap_to_original_axis(swap_axis, unified)

    save_nifti(unified, os.path.join(subject_folder, 'unified_truth_pred_contour.nii.gz'))
