import argparse
import glob
import os
from evaluation.unsupervised_eval.variability_estimation import calc_prior_based_var_est
from utils.arguments import str2bool
import nibabel as nib
import pandas as pd
from evaluation.eval_utils.eval_functions import volume
from data_curation.helper_functions import get_resolution
from utils.read_write_data import list_load

DEFAULT_SCALE = [1.56,1.56,3]


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="specifies directory of data",
                        type=str, required=True)
    parser.add_argument("--var_est_dir", help="specifies directory of variability results ",
                        type=str, required=True)
    parser.add_argument("--vol_filename", help="volume file name",
                        type=str, required=False, default='volume.nii')
    parser.add_argument("--truth_filename", help="truth file name",
                        type=str, required=False, default='truth.nii')
    parser.add_argument("--ids_list", help="list of ids on which to apply variability estimation",
                        type=str, required=False, default=None)

    #options for variability volume calculation
    parser.add_argument("--save_volume_calc", help="Shuold we save volume calculations for Possible, Consensus and Variability"
                                                    "The calculations will be saved in 'var_est_dir' directory",
                        type=str2bool, required=False, default=False)
    parser.add_argument("--var_calculated", help="whether variability estimation was already applied and saved",
                        type=str2bool, required=False, default=False)
    parser.add_argument("--use_default_res", help="is it allowed to use default resolution of [1.56,1.56,3]. Use in case of FIESTA old ids",
                        type=str2bool, required=False, default=False)
    parser.add_argument("--metadata_path", help="path to a metadata file where resolutions are specified. Needed only for volume calculation option",
                        type=str, required=False)
    parser.add_argument("--in_plane_res_name", help="in-plane resolution column name in metadata file",
                        type=str, default='PixelSpacing')
    return parser.parse_args()


def load_calc_volume(scan_dir, filename, res):
    """
    Load file and calculate its' volume
    :param scan_dir: path to scan directory
    :param filename: mask filename
    :param res: resolution
    :return:
    """
    filepath = os.path.join(scan_dir, filename)
    if os.path.exists(filepath) is False:
        filepath += '.gz'
    mask = nib.load(filepath).get_data()
    return volume(mask, res)


def calc_possible_consensus_var_volumes(out_path, scan_dirs, ids_list, truth_filename, in_plane_res_name='PixelSpacing', allow_default_scale=False):
    """
    calculate volume of mask and variability volumes (possible, consensus, variability)
    """
    if opts.metadata_path is not None:
        df = pd.read_csv(opts.metadata_path, encoding ="unicode_escape")
    else:
        df = None
    volume_metrics = {}
    volume_metrics['truth']={}
    volume_metrics['possible']={}
    volume_metrics['consensus']={}
    volume_metrics['variability']={}
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(os.path.dirname(scan_dir))
        if (ids_list is not None) and (scan_name not in ids_list):
            continue
        res = get_resolution(scan_name, df=df, in_plane_res_name=in_plane_res_name)
        if res is None:
            print('resolution for case ' + scan_name + ' cannot be extracted.')
            if allow_default_scale is True:
                print('using default resolution of ' + str(DEFAULT_SCALE))
                res = DEFAULT_SCALE
            else:
                continue
        volume_metrics['truth'][scan_name] = load_calc_volume(scan_dir, truth_filename, res)
        volume_metrics['possible'][scan_name] = load_calc_volume(scan_dir, 'possible.nii.gz', res)
        volume_metrics['consensus'][scan_name] = load_calc_volume(scan_dir, 'consensus.nii.gz', res)
        volume_metrics['variability'][scan_name] = load_calc_volume(scan_dir, 'variability.nii.gz', res)

    out_df = pd.DataFrame.from_dict(volume_metrics)
    out_df.to_csv(os.path.join(out_path, 'volume_estimations.csv'))


if __name__ == '__main__':
    """
    This script performs variability estimation using priors method
    It outputs Possible, Consensus and Variability of the segmentation mask given the input
    """
    opts = parse_eval_arguments()
    dirs = glob.glob(os.path.join(opts.data_dir,'*'))

    if opts.ids_list is not None:
        ids = list_load(opts.ids_list)
    else:
        ids=None

    if opts.var_calculated is False:
        for case_dir in dirs:
            scan_id = os.path.basename(case_dir)
            if (opts.ids_list is None) or (scan_id in ids):
                print('calculating variability for case ' + scan_id)
                calc_prior_based_var_est(case_dir, os.path.join(opts.var_est_dir, scan_id), vol_filename=opts.vol_filename, truth_filename=opts.truth_filename, T_quality=1)

    var_dirs = glob.glob(os.path.join(opts.var_est_dir,'*/'))
    if opts.save_volume_calc is True:
        calc_possible_consensus_var_volumes(opts.var_est_dir, var_dirs, ids, opts.truth_filename, opts.in_plane_res_name, allow_default_scale=opts.use_default_res)





