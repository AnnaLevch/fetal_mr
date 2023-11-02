import os
import nibabel as nib
from utils.read_write_data import *

if __name__ == "__main__":
    subject_folder = '/home/bella/Downloads/recolution_bug_data/'
    filename = 'Pat15053_Se87_Res0.78_0.78_Spac2'
    case_path = os.path.join(subject_folder, filename + '.nii.gz')
    y_true = nib.load(case_path).get_data()
    save_nifti(y_true, os.path.join(subject_folder, filename + '_basic_save.nii.gz'))

    y_true, affine, header = read_nifti_vol_meta(case_path)
    save_nifti_with_metadata(y_true, affine, header, os.path.join(subject_folder, filename + '_updated_save.nii.gz'))