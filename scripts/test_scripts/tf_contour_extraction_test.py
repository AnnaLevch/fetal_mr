import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from training.train_functions.metrics import extract_volume_2D_contours_tf_test, extract_volume_2D_bands_tf_test
import tensorflow as tf
from utils.read_write_data import save_nifti
import numpy as np


if __name__ == "__main__":
    subject_folder = '/home/bella/Phd/data/brain/hemispheres/Pat03_Se05_Res0.7422_0.7422_Spac5/'
    truth_filename = 'truth_all.nii.gz'

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    casting_test_data = [0.3,0.8,0.5, 1.0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #casting test
        casting_test = tf.cast(tf.round(casting_test_data), dtype=tf.int32).eval()
     #   casting_test = tf.cast(casting_test_data, dtype=tf.int32).eval()
        #contour extraction test
        input4D = tf.cast(tf.expand_dims(y_true,0), tf.int32)
        y_true_contour_tf, erroded = extract_volume_2D_contours_tf_test(input4D, y_true.shape)
        y_true_contour = tf.cast(y_true_contour_tf[0,:,:,:], tf.int32).eval()
        y_true_eroded = tf.cast(erroded[0,:,:,:], tf.int32).eval()
        #Band
        y_true_band_tf, dilated_band_tf, eroded_band_tf = extract_volume_2D_bands_tf_test(input4D, y_true.shape)
        y_true_band = tf.cast(y_true_band_tf[0,:,:,:], tf.int32).eval()
        dilated_band = tf.cast(dilated_band_tf[0,:,:,:], tf.int32).eval()
        eroded_band = tf.cast(eroded_band_tf[0,:,:,:], tf.int32).eval()


    print('casting result: ' + str(casting_test))
    truth_contour = swap_to_original_axis(swap_axis, y_true_contour)
    save_nifti(truth_contour, os.path.join(subject_folder, 'truth_tf_contour.nii.gz'))
 #   save_nifti(y_true_eroded, os.path.join(subject_folder, 'truth_tf_eroded.nii.gz'))

    save_nifti(y_true_band, os.path.join(subject_folder, 'truth_band.nii.gz'))
    save_nifti(dilated_band, os.path.join(subject_folder, 'truth_dilated_band.nii.gz'))
    save_nifti(eroded_band, os.path.join(subject_folder, 'truth_eroded_band.nii.gz'))