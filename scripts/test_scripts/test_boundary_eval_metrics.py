from evaluation.eval_utils.eval_functions import added_path_length_contour, false_negative_path_length_contour, surface_intersection_contour
import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from utils.read_write_data import save_nifti


""""
This script gets ground truth and result segmentation and calculates APL, FNPL and SDCS metrics. Results visualization is saved in the directory of the scan
"""""
if __name__ == "__main__":
    subject_folder = '/home/bella/Phd/code/code_bella/log/92/output/FIESTA/test/95'
    truth_filename = 'truth.nii.gz'
    result_filename = 'prediction.nii.gz'

    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_pred = nib.load(os.path.join(subject_folder, result_filename)).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)
    y_pred, swap_axis = move_smallest_axis_to_z(y_pred)

    APL_contour = added_path_length_contour(y_true, y_pred)
    FNPL_contour = false_negative_path_length_contour(y_true, y_pred)
    SDSC_contour = surface_intersection_contour(y_true, y_pred)

    APL_contour = swap_to_original_axis(swap_axis, APL_contour)
    FNPL_contour = swap_to_original_axis(swap_axis, FNPL_contour)
    SDSC_contour = swap_to_original_axis(swap_axis, SDSC_contour)

    save_nifti(APL_contour, os.path.join(subject_folder, 'apl_contour.nii.gz'))
    save_nifti(FNPL_contour, os.path.join(subject_folder,'FNPL_contour.nii.gz'))
    save_nifti(SDSC_contour, os.path.join(subject_folder, 'SDSC_contour.nii.gz'))