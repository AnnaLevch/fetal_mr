import os

import nibabel as nib
import numpy as np

from anomaly_detection.calc_features_per_slice import calculate_connected_component_max_min
from data_curation.helper_functions import move_smallest_axis_to_z
from evaluation.eval_utils.feature_extraction import connected_components_data_features
from utils.visualization import visualize_slice

if __name__ == "__main__":

    subject_folder = '/home/bella/Phd/code/code_bella/log/27/output/FIESTA_origin_gt_errors/test/225/'
    truth_filename = 'truth.nii.gz'
    slice_num = 56

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)

    components, values, counts = connected_components_data_features(y_true[:,:,slice_num-1])
    components, values, counts = connected_components_data_features(y_true[:,:,slice_num-1])
    num_components =len(values)-1
    componnents_size_std = np.std(counts[1:])
    componnents_size_mean = np.mean(counts[1:])

    visualize_slice(components)


    hausdorff = calculate_connected_component_max_min(components, values)

    print('hausdorff is: ' + str(hausdorff))