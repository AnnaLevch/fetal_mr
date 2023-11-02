import os
from glob import glob
import nibabel as nib
import numpy as np
from utils.read_write_data import save_nifti


def extract_bounding_box(vol, truth):
    non_zero_indices = np.where(vol>2)
    min_x = np.min(non_zero_indices[0])
    max_x = np.max(non_zero_indices[0])
    min_y = np.min(non_zero_indices[1])
    max_y = np.max(non_zero_indices[1])
    min_z = np.min(non_zero_indices[2])
    max_z = np.max(non_zero_indices[2])

    return vol[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1], truth[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]


if __name__ == "__main__":
    tag_index = 7
    data_path = '/media/bella/8A1D-C0A6/Phd/data/FeTA/feta_2.1/'
    roi_path = '/media/bella/8A1D-C0A6/Phd/data/FeTA/binary_data/feta_bs/'
    for sample_folder in glob(os.path.join(data_path, 'sub*')):
        case_id = os.path.basename(sample_folder)
        if os.path.exists(os.path.join(sample_folder,'anat', case_id + '_rec-irtk_T2w.nii.gz')):
            vol_nifti = nib.load(os.path.join(sample_folder,'anat', case_id + '_rec-irtk_T2w.nii.gz'))
            vol = vol_nifti.get_data()
            truth_nifti = nib.load(os.path.join(sample_folder,'anat', case_id + '_rec-irtk_dseg.nii.gz'))
            truth = truth_nifti.get_data()
        else:
            vol_nifti = nib.load(os.path.join(sample_folder,'anat', case_id + '_rec-mial_T2w.nii.gz'))
            vol = vol_nifti.get_data()
            truth_nifti = nib.load(os.path.join(sample_folder,'anat', case_id + '_rec-mial_dseg.nii.gz'))
            truth = truth_nifti.get_data()
        case_dir = os.path.join(roi_path, case_id)
        roi_vol, roi_truth = extract_bounding_box(vol, truth)
        if not os.path.exists(case_dir):
            os.mkdir(case_dir)

        #make brain predictions binary
        nonzero = np.nonzero(roi_truth==tag_index)
        roi_truth[roi_truth!=tag_index]=0
        roi_truth[roi_truth==tag_index]=1

        nib.save(nib.Nifti1Image(roi_vol, vol_nifti.affine), os.path.join(case_dir,'volume.nii.gz'))
        nib.save(nib.Nifti1Image(roi_truth, truth_nifti.affine), os.path.join(case_dir,'truth.nii.gz'))