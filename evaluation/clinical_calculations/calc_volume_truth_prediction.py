import argparse
import glob
import pandas as pd
import os
from evaluation.clinical_calculations.calc_volume_dir import calc_volume

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to a metadata file where resolutions are specified",
                        type=str, required=True)
    parser.add_argument("--out_path", help="specifies the path of output volume calculation",
                        type=str, required=True)
    parser.add_argument("--truth_filename", help="filename of the volume mask",
                        type=str, default='truth.nii')
    parser.add_argument("--prediction_filename", help="filename of the volume mask",
                        type=str, default='prediction.nii')
    parser.add_argument("--ids_list", help="file with needed ids",
                        type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":

    opts = parse_arguments()
    scan_dirs = glob.glob(os.path.join(opts.src_dir, '*'))
    df = pd.read_csv(opts.metadata_path, encoding ="unicode_escape")

    ids_lst = None
    if opts.ids_list is not None:
        df_ids = pd.read_csv(opts.ids_list, encoding ="unicode_escape")
        ids_lst = set(df_ids['Subject'].to_list())

    volume_metrics = {}
    volume_metrics['truth'] = {}
    volume_metrics['prediction'] = {}
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)

        if ids_lst is not None and (scan_name) not in ids_lst:
            continue
        if os.path.isdir(scan_dir) is False:
            continue

        print('calculating id: ' + scan_name)

        volume_metrics['truth'][scan_name] = calc_volume(opts.truth_filename, scan_dir, df, scan_name)/1000
        volume_metrics['prediction'][scan_name] = calc_volume(opts.prediction_filename, scan_dir, df, scan_name)/1000

    out_df = pd.DataFrame.from_dict(volume_metrics)
    out_df.to_csv(opts.out_path)