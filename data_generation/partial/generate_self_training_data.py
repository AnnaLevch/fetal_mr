import glob
import os
import shutil
import nibabel as nib
import numpy as np


def get_dict(pathes):
    ret_dict = {}
    for path in pathes:
        subject_id = os.path.basename(path)
        ret_dict.append(subject_id, path)
    return ret_dict


if __name__ == '__main__':
    """
    unify partial annotations with self training
    """
    self_training_results_path = '/media/bella/8A1D-C0A6/Phd/log/946/output/placenta_trufi_training/test'
    partial_data_path = '/home/bella/Phd/data/placenta/ASL/partial_9/partial0.2_0whole_0'
    unified_data_path = '/home/bella/Phd/data/placenta/ASL/partial_9_self_training/partial_self_training_0'

    partial = glob.glob(os.path.join(partial_data_path, "*"))
    self_training_res = glob.glob(os.path.join(self_training_results_path, "*"))

    partial_cases = get_dict(partial)
    train_cases_pseudo_labels = get_dict(self_training_res)

    #unify partial annotations with self training results
    for training_case in train_cases_pseudo_labels:
        shutil.copy(os.path.join(train_cases_pseudo_labels[training_case], "data.nii.gz"), os.path.join(unified_data_path, training_case, "volume.nii.gz"))
        partial_truth = nib.load(os.path.join(partial_cases[training_case], "partial_truth.nii.gz")).get_data()
        partial_uncertainty = nib.load(os.path.join(partial_cases[training_case], "uncertainty.nii.gz")).get_data()
        st_truth = nib.load(os.path.join(train_cases_pseudo_labels[training_case], "prediction.nii.gz")).get_data()
        st_uncertainty = nib.load(os.path.join(train_cases_pseudo_labels[training_case], "uncertainty.nii.gz")).get_data()
        ground_truth_indices = np.where(partial_uncertainty == 0)
        st_uncertainty[ground_truth_indices]=0
        st_truth[ground_truth_indices] = partial_truth[ground_truth_indices]

        save_nifti(volume, os.path.join(det_out_dir, 'volume.nii.gz'))