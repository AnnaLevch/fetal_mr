import argparse
import glob
import os
import nibabel as nib
from evaluation.eval_utils.eval_functions import volume
from data_curation.helper_functions import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to a metadata file where resolutions are specified",
                        type=str, required=True)
    parser.add_argument("--out_path", help="specifies the path of output volume calculation",
                        type=str, required=True)
    parser.add_argument("--mask_filename", help="filename of the volume mask",
                        type=str, default='truth.nii')
    parser.add_argument("--ids_list", help="file with needed ids",
                        type=str, default=None)
    return parser.parse_args()


def calc_volume(filename, scan_dir, df, scan_name):
    mask_path = os.path.join(scan_dir, filename)
    if os.path.exists(mask_path) is False:
        mask_path = mask_path + '.gz'
    mask = nib.load(mask_path).get_data()

    res = get_resolution(scan_name, df=df, subject_id_scan_id=True)
    if res is None:
        res = default_resolution
        print('case ' + scan_name + " not found, using default resolution of " + str(default_resolution))
    else:
        print('case: ' + scan_name + ', resolution is: '+ str(res))
    return volume(mask, res)


if __name__ == "__main__":
   # default_resolution = [0.78125, 0.78125, 2]#use carefully, check if it is brain or body
    default_resolution = [1.56,1.56,3]
    opts = parse_arguments()
    scan_dirs = glob.glob(os.path.join(opts.src_dir, '*'))
    df = pd.read_csv(opts.metadata_path, encoding ="unicode_escape")

    ids_lst = None
    if opts.ids_list is not None:
        df_ids = pd.read_csv(opts.ids_list, encoding ="unicode_escape")
        ids_lst = set(df_ids['Subject'].to_list())

    volume_metrics = {}
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)

        if ids_lst is not None and (scan_name) not in ids_lst:
            continue
        if os.path.isdir(scan_dir) is False:
            continue

        print('calculating id: ' + scan_name)

        volume_metric = calc_volume(opts.mask_filename, scan_dir, df, scan_name)
        volume_metrics[scan_name] = volume_metric

    out_df = pd.DataFrame(volume_metrics, index=[0]).T
    out_df.to_csv(opts.out_path)
