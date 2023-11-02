import glob
import os
import nibabel as nib
import numpy as np
from utils.read_write_data import save_nifti
from evaluation.eval_utils.postprocess import postprocess_prediction

if __name__ == "__main__":
    body_gt_path = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin_clean/'
    runs = [351,365]

    for i in range(0,len(runs)):
        placenta_res_path = '/home/bella/Phd/code/code_bella/log/{run}/output/Placenta_FIESTA_det_361/test'.format(run=runs[i])
        dirs = glob.glob(os.path.join(placenta_res_path,'*/'))
        for dir in dirs:
            dir_id = os.path.basename(os.path.dirname(dir))
            print('run {run}, dirname {dirname}'.format(run=runs[i], dirname=dir_id))
            placenta_res = nib.load(os.path.join(dir, 'prediction.nii.gz')).get_data()
            body_gt = nib.load(os.path.join(body_gt_path, dir_id, 'truth.nii')).get_data()
            body_indices = np.nonzero(body_gt)
            placenta_res[body_indices]=0
            placenta_res = np.uint8(postprocess_prediction(placenta_res, remove_small=True, fill_holes_2D=True))
            save_nifti(placenta_res, os.path.join(dir, 'prediction_postprocess.nii.gz'))
