import glob
import os
import nibabel as nib
import numpy as np
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from utils.read_write_data import save_nifti
import shutil


def update_partial_case(partial_case_dir, data_path, truth_filename):
    case_id = os.path.basename(partial_case_dir)
    print('updating case: ' + case_id)
    if os.path.exists(os.path.join(data_path, case_id, truth_filename)) is False:
        truth_filename = truth_filename + '.gz'
    y_true = nib.load(os.path.join(data_path, case_id, truth_filename)).get_data()
    partial_uncertainty = nib.load(os.path.join(partial_case_dir, 'uncertainty.nii.gz')).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    partial_uncertainty, swap_axis = move_smallest_axis_to_z(partial_uncertainty)
    nonzero_slices = set(np.nonzero(y_true)[2])
    min_slice = min(nonzero_slices)
    max_slice = max(nonzero_slices)
    #update partial uncertainty
    partial_uncertainty[:,:,:min_slice] = 0
    partial_uncertainty[:,:,max_slice:] = 0
    partial_uncertainty = swap_to_original_axis(swap_axis, partial_uncertainty)
    save_nifti(partial_uncertainty, os.path.join(partial_case_dir, 'uncertainty_with_borders.nii.gz'))


if __name__ == "__main__":
    """
    Add border slices to partial annotations
    """
    partial_path = '/media/bella/8A1D-C0A6/Phd/data/placenta/ASL/partial_9/detection/'
    whole_path = '/home/bella/Phd/data/placenta/ASL/placenta_training_valid_set/'
    truth_filename = 'truth.nii'

    partial_dirs = glob.glob(os.path.join(partial_path, "*"))

    for partial_dir in partial_dirs:
        print('updating path '+ partial_dir)
        cases_dirs = glob.glob(os.path.join(partial_dir, "*"))
        for case_dir in cases_dirs:
            if os.path.exists(os.path.join(case_dir, 'partial_truth.nii.gz')) is False:
                shutil.copy(os.path.join(case_dir, 'uncertainty.nii.gz'), os.path.join(case_dir, 'uncertainty_with_borders.nii.gz'))
            update_partial_case(case_dir, whole_path, truth_filename)