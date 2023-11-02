import argparse
import glob
import os
import nibabel as nib
from data_curation.helper_functions import get_resolution
import pandas as pd
from evaluation.eval_utils.eval_functions import volume
import numpy as np


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", help="specifies directory of TTA results that includes uncertainty",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="specifies path to metadata for volume calculation ",
                        type=str, required=True)
    parser.add_argument("--out_path", help="variability evaluation output path ",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_eval_arguments()

    dirs = glob.glob(os.path.join(opts.results_dir,'*'))
    df = pd.read_csv(opts.metadata_path, encoding ="unicode_escape")
    variability_ratios = {}

    for case_dir in dirs:
        subject_id = os.path.basename(case_dir)
        print('calcuating variability of case ' + subject_id)
        variability = nib.load(os.path.join(case_dir, 'uncertainty.nii.gz')).get_data()
        truth = nib.load(os.path.join(case_dir, 'truth.nii.gz')).get_data()

        variability[variability>0.5] = 1
        res = get_resolution(subject_id, df=df)
        variability_vol = volume(variability, res)
        truth_vol = volume(truth, res)

        variability_ratio = variability_vol/truth_vol
        variability_ratios[subject_id] = variability_ratio

    out_df = pd.DataFrame(variability_ratios, index=[0]).T
    out_df.to_csv(opts.out_path)