import argparse
import glob
import os
import nibabel as nib
import numpy as np
import shutil
from utils.read_write_data import save_nifti


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--truth_path", help="specifies the path of ground truth segmentation",
                        type=str, required=True)
    parser.add_argument("--diff_filename", help="specifies the filename of diff result",
                        type=str, default='prediction.nii.gz')
    parser.add_argument("--prediction_dir", help="if prediction mask is in another path, specify it",
                        type=str, default=None)
    parser.add_argument("--prediction_filename", help="specifies the filename of prediction",
                        type=str, default='mask.nii.gz')
    parser.add_argument("--out_path", help="specifies the path of unified result. If None, it is the same as input path",
                        type=str, default=None)

    return parser.parse_args()

"""
This script unifies results of original network result and diff (error) network result and copies ground truth to the
same directory
"""
if __name__ == '__main__':

    opts = parse_arguments()

    if opts.out_path is None:
        opts.out_path = opts.diff_dir
    if opts.prediction_dir is None:
        opts.prediction_dir = opts.diff_dir

    scan_dirs = glob.glob(opts.diff_dir + '/*/')

    for subject_dir in scan_dirs:
        subject_id = os.path.basename(subject_dir[:-1])
        y_diff = nib.load(os.path.join(opts.diff_dir, subject_dir, opts.diff_filename)).get_data()
        y_pred = nib.load(os.path.join(opts.prediction_dir, subject_dir, opts.prediction_filename)).get_data()
        y_diff_pos = np.nonzero(y_diff)
        y_pred[y_diff_pos]= abs(y_pred[y_diff_pos]-y_diff[y_diff_pos])
        save_nifti(y_pred, os.path.join(subject_dir, os.path.join(opts.out_path, subject_dir,'prediction_unified.nii.gz')))
        shutil.copy(os.path.join(opts.truth_path, subject_id, 'truth.nii.gz'), os.path.join(opts.out_path, subject_dir,'truth_unified.nii.gz'))