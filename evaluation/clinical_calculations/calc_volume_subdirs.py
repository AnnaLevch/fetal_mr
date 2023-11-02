import argparse
import glob
import os
import nibabel as nib
from evaluation.eval_utils.eval_functions import volume
from data_curation.helper_functions import *
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to a metadata file where resolutions are specified",
                        type=str, required=False)
    parser.add_argument("--out_path", help="specifies the path of output inference",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """
    This script applies network and calculates volume for scans inside a subdirectory and outputs results to excel file
    """
    default_resolution = [0.7812,0.7812,2]#use carefully, check if it is brain or body
    opts = parse_arguments()
    scans_dirs = glob.glob(os.path.join(opts.src_dir, '*'))
    res = None

    volume_metrics = {}
    volume_metrics['pat']={}
    volume_metrics['volume']={}
    for scans_dir in scans_dirs:
        out_dir = os.path.join(opts.out_path, os.path.basename(scans_dir))+'/'
        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)
        args = "--input_path {input_path} --output_folder {out_dir}" \
           " --config_dir /home/bella/Phd/code/code_bella/log/636/ --preprocess window_1_99 --labeled False" \
            " --scan_series_id_from_name False --all_in_one_dir True".format(input_path=scans_dir, out_dir=out_dir)
        print('running with arguments:')
        print(args)
        subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

        scans_pathes = glob.glob(os.path.join(out_dir,'test', '*','prediction.nii.gz'))
        pat = os.path.basename(scans_dir)
        for scan_path in scans_pathes:
            mask = nib.load(scan_path).get_data()
         #   res = get_resolution(pat, df=df)
            scan_name = os.path.basename(os.path.dirname(scan_path))
            if res is None:
                res = default_resolution
                print('case ' + scan_path + " not found, using default resolution!")
            else:
                print('case: ' + scan_path + ', resolution is: '+ str(res))
            volume_metric = volume(mask, res)
            volume_metrics['pat'][scan_name] = pat
            volume_metrics['volume'][scan_name] = volume_metric

    out_df = pd.DataFrame.from_dict(volume_metrics)
    out_df.to_csv(os.path.join(opts.out_path, 'volume_calc.csv'))
