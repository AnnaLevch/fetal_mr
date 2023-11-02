import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
import tensorflow as tf
from utils.read_write_data import save_nifti
import numpy as np


if __name__ == "__main__":

    subject_folder = '/home/bella/Phd/code/code_bella/log/531/output/placenta_FIESTA_unsupervised_cases/test/Pat16_Se17_Res1.875_1.875_Spac4.1/'
    truth_filename = 'prediction_soft.nii.gz'
    uncertainty = 'uncertainty.nii.gz'

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    uncertainty = nib.load(os.path.join(subject_folder, uncertainty)).get_data()
    uncertainty, swap_axis = move_smallest_axis_to_z(uncertainty)
    casting_test_data = [0.3,0.8,0.5, 1.0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #uncertainty test
        y_true_tf = tf.cast(tf.expand_dims(y_true,0), tf.int32)
        uncertainty_tf = tf.cast(tf.expand_dims(uncertainty,0), tf.double)
        indices = tf.where(uncertainty_tf > 0.5)
        size = tf.size(indices).eval()
        band = -1
        result = tf.cond(tf.greater(tf.size(indices), 0), lambda: 5, lambda: -1).eval()
        if result is True:
            print('true')
        y_true_indices_only = tf.gather_nd(y_true_tf, indices).eval()
        print('stop')