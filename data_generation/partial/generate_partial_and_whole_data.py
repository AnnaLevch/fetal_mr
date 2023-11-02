import glob
import os
from utils.read_write_data import list_load, save_nifti
import shutil
import nibabel as nib
import numpy as np
from data_generation.cut_roi_data import cut_with_roi_data_np


def create_partial_whole_data(cases_pathes, all_cases_ids, out_det_path, out_seg_path):
    for case_dir in cases_pathes:
        case_id = os.path.basename(case_dir)
        if case_id not in all_cases_ids:
            continue  # use only training and validation cases

        out_case_det_dir = os.path.join(out_det_path, case_id)
        if os.path.exists(out_case_det_dir) is False:
            os.mkdir(out_case_det_dir)
        out_case_seg_dir = os.path.join(out_seg_path, case_id)
        if os.path.exists(out_case_seg_dir) is False:
            os.mkdir(out_case_seg_dir)
        volume_path = os.path.join(case_dir, volume_filename)
        if os.path.exists(volume_path):
            volume = nib.load(volume_path).get_data()
        else:
            volume = nib.load(volume_path + '.gz').get_data()
        save_nifti(volume, os.path.join(out_det_path, case_id, volume_filename + '.gz'))
        truth_path = os.path.join(case_dir, truth_filename)
        if os.path.exists(truth_path):
            y_true = nib.load(truth_path).get_data()
        else:
            y_true = nib.load(truth_path + '.gz').get_data()

        if case_id in whole_cases_ids:
            # detection
            truth_path = os.path.join(case_dir, truth_filename)
            save_nifti(y_true, os.path.join(out_det_path, case_id, truth_filename + '.gz'))
            uncertainty = np.zeros_like(y_true)
            save_nifti(uncertainty, os.path.join(out_det_path, case_id, 'uncertainty.nii.gz'))
            # segmentation
            gt_save_path = os.path.join(out_seg_path, case_id, truth_filename + '.gz')
            vol_save_path = os.path.join(out_seg_path, case_id, volume_filename + '.gz')
            mask_save_path = os.path.join(out_seg_path, case_id, uncertainty_filename + '.gz')
            cut_with_roi_data_np(y_true, volume, uncertainty, gt_save_path, vol_save_path, mask_save_path,
                                 uncertainty_filename, padding)

        else:
            # detection
            shutil.copy(os.path.join(case_dir, partial_filename + '.gz'),
                        os.path.join(out_det_path, case_id, partial_filename + '.gz'))
            shutil.copy(os.path.join(case_dir, uncertainty_filename + '.gz'),
                        os.path.join(out_det_path, case_id, uncertainty_filename + '.gz'))
            # segmentation
            uncertainty = nib.load(os.path.join(case_dir, uncertainty_filename + '.gz')).get_data()
            partial_truth = nib.load(os.path.join(case_dir, partial_filename + '.gz')).get_data()
            gt_save_path = os.path.join(out_seg_path, case_id, truth_filename + '.gz')
            vol_save_path = os.path.join(out_seg_path, case_id, volume_filename + '.gz')
            uncertainty_save_path = os.path.join(out_seg_path, case_id, uncertainty_filename + '.gz')
            partial_save_path = os.path.join(out_seg_path, case_id, partial_filename + '.gz')
            cut_with_roi_data_np(y_true, volume, uncertainty, gt_save_path, vol_save_path, uncertainty_save_path,
                                 uncertainty_filename, padding)
            cut_with_roi_data_np(y_true, volume, partial_truth, gt_save_path, vol_save_path, partial_save_path,
                                 partial_filename, padding)
            if os.path.exists(os.path.join(out_seg_path, case_id, 'truth.nii')):
                os.remove(os.path.join(out_seg_path, case_id, 'truth.nii'))
            elif os.path.exists(os.path.join(out_seg_path, case_id, 'truth.nii.gz')):
                os.remove(os.path.join(out_seg_path, case_id, 'truth.nii.gz'))


if __name__ == '__main__':
    data_path = '/home/bella/Phd/data/placenta/placenta_partial_annotations0.2'
    out_det_path = '/home/bella/Phd/data/placenta/placenta_partial0.2_6whole_1/'
    out_seg_path = '/home/bella/Phd/data/placenta/placenta_partial0.2_6whole_1_cutted/'
    debug_split_whole_cases = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/partial/partial_split/debug_split_6/'
    origin_split_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/debug_split'
    volume_filename = 'volume.nii'
    truth_filename = 'truth.nii'
    partial_filename = 'partial_truth.nii'
    uncertainty_filename = 'uncertainty.nii'
    padding = np.array([16, 16, 8])

    whole_training = list_load(os.path.join(debug_split_whole_cases, 'training_ids.txt'))
    origin_training = list_load(os.path.join(origin_split_path, 'training_ids.txt'))
    whole_validation = list_load(os.path.join(debug_split_whole_cases, 'validation_ids.txt'))
    whole_cases_ids = whole_training + whole_validation
    all_cases_ids = origin_training + whole_validation
    cases_pathes = glob.glob(os.path.join(data_path, '*'))

    create_partial_whole_data(cases_pathes, all_cases_ids)





