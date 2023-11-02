import os
import glob
import nibabel as nib
from utils.read_write_data import save_nifti

if __name__ == "__main__":
    """
    Script to trim soft result based on binary result
    """
    log_dir_path = "/media/bella/8A1D-C0A6/Phd/log"
    pred_filename = 'prediction.nii.gz'
    soft_pred_filename = 'prediction_soft.nii.gz'
    save_filename ='prediction_soft_trimmed.nii.gz'
    dirs = [531,532,533,534,535]

    for dir_id in dirs:
        cases_path = os.path.join(log_dir_path, str(dir_id), 'output', 'placenta_FIESTA_unsupervised_cases', 'chosen_cases_soft')
        cases_dirs = glob.glob(os.path.join(cases_path, '*'))
        for case_dir in cases_dirs:
            y_pred = nib.load(os.path.join(case_dir, pred_filename)).get_data()
            y_pred_soft = nib.load(os.path.join(case_dir, soft_pred_filename)).get_data()
            y_pred_soft[y_pred==0]=0
            save_nifti(y_pred_soft, os.path.join(case_dir, save_filename))