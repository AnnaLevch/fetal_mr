import os
import glob
import argparse
from utils.arguments import str2bool
import pandas as pd
from data_curation.helper_functions import get_metadata_by_subject_id


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--output_path", help="specifies output csv path",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to metadata",
                        type=str, required=False)
    return parser.parse_args()

if __name__ == '__main__':

    opts = get_arguments()
    scan_dirs = glob.glob(os.path.join(opts.input_dir, '*'))
    data_df = pd.read_csv(opts.metadata_path)
    new_df = None
    for scan_dir in scan_dirs:
        scan_df = get_metadata_by_subject_id(os.path.basename(scan_dir), df=data_df)
        if(scan_df is None):
            continue
        if new_df is None:
            new_df = scan_df
        else:
            new_df.append(scan_df)

    new_df.to_csv(opts.output_path)