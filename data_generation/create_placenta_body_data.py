import glob
import os
import nibabel as nib
from data_curation.helper_functions import move_smallest_axis_to_z
from utils.read_write_data import save_nifti


if __name__ == '__main__':
    placenta_path = '/home/bella/Phd/data/placenta/placenta_clean/'
    body_path = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin_clean'
    out_path = '/home/bella/Phd/data/body_placenta/FIESTA/'
    pred_filename = 'volume.nii'
    truth_filename = 'truth.nii'

    placenta_cases = glob.glob(os.path.join(placenta_path, "*"))
    for case_path in placenta_cases:
        volume_path = os.path.join(case_path, pred_filename)
        if os.path.exists(volume_path) is False:
            volume_path = os.path.join(case_path, pred_filename + '.gz')
        volume = nib.load(volume_path).get_data()
        volume, _ = move_smallest_axis_to_z(volume)

        truth_path = os.path.join(case_path, truth_filename)
        if os.path.exists(truth_path) is False:
            truth_path = os.path.join(case_path, truth_filename + '.gz')
        truth = nib.load(truth_path).get_data()
        truth, _ = move_smallest_axis_to_z(truth)

        id = os.path.basename(case_path)
        body_case_path = glob.glob(os.path.join(body_path, str(id)))
        if len(body_case_path) != 1:
            print('no exact match for case ' + str(id))
            continue
        mask_path = os.path.join(body_case_path[0], 'truth.nii')
        if os.path.exists(mask_path) is False:
            mask_path = os.path.join(body_case_path[0], truth_filename + '.gz')
        body_mask = nib.load(mask_path).get_data()
        body_mask, _ = move_smallest_axis_to_z(body_mask)

        out_dir = os.path.join(out_path, str(id))
        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)
        save_nifti(volume, os.path.join(out_dir, 'volume.nii.gz'))
        save_nifti(truth, os.path.join(out_dir, 'truth.nii.gz'))
        save_nifti(body_mask, os.path.join(out_dir, 'body_mask.nii.gz'))