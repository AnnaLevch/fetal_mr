import argparse
import glob
import os
import nibabel as nib
import numpy as np
from utils.read_write_data import save_nifti


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", help="specifies directory of TTA results ",
                        type=str, required=True)
    parser.add_argument("--variability_dir", help="specifies directory of variability results ",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    """
   This scripts creates possible, consensus and variability masks for TTA results and outputs them in TTA result folder
    """
    opts = parse_eval_arguments()

    dirs = glob.glob(os.path.join(opts.results_dir,'*'))

    for case_dir in dirs:
        subject_id = os.path.basename(case_dir)
        print('calcuating variability of case ' + subject_id)
        median_prediction = nib.load(os.path.join(case_dir, 'prediction.nii.gz')).get_data()
        possible = np.zeros_like(median_prediction)
        consensus = np.copy(median_prediction)
        tta_predictions_pathes = glob.glob(os.path.join(case_dir,'tta*'))

        for tta_path in tta_predictions_pathes:
            tta_res = nib.load(tta_path).get_data()
            possible[np.nonzero(tta_res)] = 1
            consensus[tta_res == 0] = 0

        variability = np.copy(possible)
        variability[np.nonzero(consensus)]=0

        os.mkdir(os.path.join(opts.variability_dir, subject_id))
        save_nifti(np.int16(possible), os.path.join(opts.variability_dir, subject_id,'possible_TTA.nii.gz'))
        save_nifti(np.int16(consensus), os.path.join(opts.variability_dir, subject_id, 'consensus_TTA.nii.gz'))
        save_nifti(np.int16(variability), os.path.join(opts.variability_dir, subject_id, 'variability_TTA.nii.gz'))
