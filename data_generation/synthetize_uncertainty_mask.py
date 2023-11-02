import glob
import os
import nibabel as nib
import numpy as np
from utils.read_write_data import save_nifti


if __name__ == "__main__":
    data_dir = '/home/bella/Phd/data/placenta/placenta_clean_cutted/'
    data_pathes = glob.glob(os.path.join(data_dir,"*"))

    for path in data_pathes:
        truth_filename = os.path.join(path, 'truth.nii')
        if os.path.exists(truth_filename) is False:
            truth_filename = os.path.join(path, 'truth.nii.gz')
        y_pred = nib.load(truth_filename).get_data()
        uncertainty = np.zeros_like(y_pred)
        save_nifti(uncertainty, os.path.join(path, 'uncertainty.nii.gz'))