import glob
import os
import nibabel as nib
from utils.read_write_data import save_nifti


if __name__ == '__main__':
    dirs = [531,532,533,534,535]
    log_dir_path = '/media/bella/8A1D-C0A6/Phd/log'
    filename = 'prediction_soft.nii.gz'
    folds = 5

    for i in range(folds):
      print('updating fold ' + str(i))
      cases_pathes = glob.glob(os.path.join(log_dir_path, str(dirs[i]), 'output', 'placenta_FIESTA_unsupervised_cases',
                                            'test', '*'))
      for case in cases_pathes:
          print('case ' + case)
          tta_pred = nib.load(os.path.join(case, filename)).get_data()
          if len(tta_pred.shape) == 4:
            save_nifti(tta_pred[-1,:,:,:], os.path.join(case,'prediction_soft.nii.gz'))