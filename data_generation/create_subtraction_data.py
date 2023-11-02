import glob
import os
import nibabel as nib
import shutil
from utils.read_write_data import save_nifti
from data_curation.helper_functions import move_smallest_axis_to_z,swap_to_original_axis
import numpy as np

if __name__ == '__main__':
    src_dir = '/media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/TRUFI2020/TRUFI_42'
    prediction_filename = 'prediction.nii'
    truth_filename = "truth.nii"
    prediction_data_path = '/media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/TRUFI2020/TRUFI_42_annotations'

    dirs_path =  glob.glob(os.path.join(src_dir, '*'))

    for subject_dir in dirs_path:
        basedir = os.path.basename(subject_dir)
        shutil.copy(os.path.join(prediction_data_path, basedir, 'prediction.nii.gz'), os.path.join(subject_dir, 'prediction.nii.gz'))

    for subject_dir in dirs_path:
        print('case ' + subject_dir)
        if os.path.exists(os.path.join(os.path.join(subject_dir, truth_filename))):
            y_true = nib.load(os.path.join(subject_dir, truth_filename)).get_data()
        elif os.path.exists(os.path.join(os.path.join(subject_dir, truth_filename + '.gz'))):
            y_true = nib.load(os.path.join(subject_dir, truth_filename + '.gz')).get_data()
        else:
            print('Truth data was not found for case ' + subject_dir)
        if os.path.exists(os.path.join(subject_dir, prediction_filename)):
            y_pred = nib.load(os.path.join(subject_dir, prediction_filename)).get_data()
        elif os.path.exists(os.path.join(subject_dir, prediction_filename + '.gz')):
            y_pred = nib.load(os.path.join(subject_dir, prediction_filename + '.gz')).get_data()
        else:
            print('prediction data was not found for case ' + subject_dir)
        diff_data =  (1-y_pred)*y_true + (1-y_true)*y_pred
        save_nifti(diff_data, os.path.join(subject_dir, 'diff.nii.gz'))