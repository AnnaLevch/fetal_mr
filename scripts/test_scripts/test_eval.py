import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from evaluation.eval_utils.eval_functions import *

if __name__ == "__main__":
    subject_folder = '/home/bella/Phd/code/code_bella/log/anomaly_datection/17/output/FIESTA_origin_gt_errors/test/117/'
    truth_filename = 'truth.nii.gz'
    result_filename = 'prediction.nii.gz'
    slice = 38
    scaling = (1.56,1.56)

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_pred = nib.load(os.path.join(subject_folder, result_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    y_pred, swap_axis = move_smallest_axis_to_z(y_pred)

    #specific slice
    dice_val = dice(y_true[:,:,slice-1], y_pred[:,:,slice-1])
    hausdorff_val = hausdorff_lits(y_true[:,:,slice-1], y_pred[:,:,slice-1], scaling)

    # #2D calculation for volume
    dice_vals = calc_overlap_measure_per_slice(y_true, y_pred, dice)
    hausdorff_vals = calc_distance_measure_per_slice(y_true, y_pred, scaling, hausdorff_lits)
    pixel_diff = calc_overlap_measure_per_slice(y_true, y_pred, pixel_difference)
    print("")