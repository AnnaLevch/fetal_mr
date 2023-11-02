import argparse
import os
import glob
import nibabel as nib
import numpy as np
from utils.read_write_data import save_nifti


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variability_dir", help="path to variability estimation",
                        type=str, required=True)
    parser.add_argument("--annotator1", help="annotator 1 filename",
                        type=str, required=True)
    parser.add_argument("--annotator2", help="annotator 2 filename",
                        type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    """
    This script calculates possible, consensus and variability between 2 annotators
    """
    opts = parse_eval_arguments()

    dirs = glob.glob(os.path.join(opts.variability_dir,'*'))
    for case_dir in dirs:
        annotator1_seg = nib.load(os.path.join(case_dir,opts.annotator1)).get_data()
        annotator2_seg = nib.load(os.path.join(case_dir,opts.annotator2)).get_data()
        possible = np.copy(annotator1_seg)
        possible[np.nonzero(annotator2_seg)] = 1
        consensus = np.copy(annotator1_seg)
        consensus[annotator2_seg==0] = 0
        variability = np.copy(possible)
        variability[np.nonzero(consensus)]=0

        save_nifti(np.int16(possible), os.path.join(case_dir, 'possible_manual.nii.gz'))
        save_nifti(np.int16(consensus), os.path.join(case_dir, 'consensus_manual.nii.gz'))
        save_nifti(np.int16(variability), os.path.join(case_dir, 'variability_manual.nii.gz'))