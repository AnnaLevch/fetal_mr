import argparse
from utils.arguments import str2bool
import pickle
from pathlib import Path
from glob import glob
import os
from evaluation.evaluate import evaluate_all, write_to_excel
from evaluation.eval_utils.eval_functions import *


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variability_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to scans details for correct resolution",
                        type=str, required=True)
    parser.add_argument("--num_key_imgs", help="specifies source directory for evaluation",
                        type=int, required=False, default=4)
    parser.add_argument("--vol_filename", help="name of the volume file",
                        type=str, required=False, default='data.nii.gz')
    parser.add_argument("--scan_series_id_from_name", help="should we extract scan and series id from name to get metadata. Set to True if metadata has this values",
                        type=str2bool, required=False, default=True)
    return parser.parse_args()


def get_calc_scores(variability_dir, metadata_path, postfix, metrics_without_rescaling, without_rescaling_2D, metrics_with_rescaling_2D, metrics_with_rescaling):
    """
    Calculate scores if they do not already exist
    It is assumed that reference manual segmentation has a postfix 'manual' in the filename
    :param variability_dir: directory of variability estimation and reference variability
    :param postfix: pkl files postfix
    :return: volumetric and 2D scores
    """
    vol_scores_pkl = os.path.join(variability_dir,'pred_scores_vol_' + postfix + '.pkl')
    slice_scores_pkl = os.path.join(variability_dir,'pred_scores_per_slice_' + postfix + '.pkl')

    if Path(slice_scores_pkl).exists() and Path(vol_scores_pkl).exists():
        print('scores were already calculated, loading')
        with open(slice_scores_pkl, 'rb') as f:
            pred_scores_per_slice = pickle.load(f)
        with open(vol_scores_pkl, 'rb') as f:
            pred_scores_vol = pickle.load(f)
    else:
        pred_scores_per_slice, pred_scores_vol = evaluate_all([_ for _ in (glob(os.path.join(variability_dir, '*/')))], truth_filename= postfix + '_manual.nii.gz',
                                                              result_filename=postfix + '.nii.gz', metadata_path=metadata_path,
                                                              scan_series_id_from_name=opts.scan_series_id_from_name, metrics_without_rescaling=metrics_without_rescaling,
                                                              without_rescaling_2D=without_rescaling_2D, metrics_with_rescaling_2D=metrics_with_rescaling_2D,
                                                              metrics_with_rescaling=metrics_with_rescaling)
        print('--------------------\nsaving...')
        with open(slice_scores_pkl, 'wb') as f:
            pickle.dump(pred_scores_per_slice, f)
        with open(vol_scores_pkl, 'wb') as f:
            pickle.dump(pred_scores_vol, f)

    return pred_scores_per_slice, pred_scores_vol


if __name__ == '__main__':
    """
    This script evaluates variability estimation - metrics for possible, consensus and variability
    """
    #TODO: currently does not work because write_to_excel is not scalable to different metrics!
    opts = parse_eval_arguments()

    #set variability estimation relevant metrics
    metrics_without_rescaling= [dice, vod]
    without_rescaling_2D=[dice]
    metrics_with_rescaling_2D=[]
    metrics_with_rescaling=[volume_difference, volume_difference_ratio]


    scores_per_slice_possible, scores_vol_possible = get_calc_scores(opts.variability_dir, opts.metadata_path, 'possible',
                                                                     metrics_without_rescaling, without_rescaling_2D, metrics_with_rescaling_2D, metrics_with_rescaling)
    scores_per_slice_consensus, scores_vol_possible = get_calc_scores(opts.variability_dir, opts.metadata_path, 'consensus',
                                                                      metrics_without_rescaling, without_rescaling_2D, metrics_with_rescaling_2D, metrics_with_rescaling)
    scores_per_slice_variability, scores_vol_possible = get_calc_scores(opts.variability_dir, opts.metadata_path, 'variability',
                                                                        metrics_without_rescaling, without_rescaling_2D, metrics_with_rescaling_2D, metrics_with_rescaling)

    write_to_excel(scores_per_slice_possible,  scores_per_slice_consensus, os.path.join(opts.variability_dir, 'possible_eval.xlsx'), opts.variability_dir,
                   opts.variability_dir, opts.vol_filename, 'possible_manual.nii.gz','possible.nii.gz', opts.metadata_path)
    write_to_excel(scores_per_slice_possible,  scores_per_slice_consensus, os.path.join(opts.variability_dir, 'consensus_eval.xlsx'), opts.variability_dir,
                   opts.variability_dir, opts.vol_filename, 'consensus_manual.nii.gz','consensus.nii.gz', opts.metadata_path)
    write_to_excel(scores_per_slice_possible,  scores_per_slice_consensus, os.path.join(opts.variability_dir, 'variability_eval.xlsx'), opts.variability_dir,
                   opts.variability_dir, opts.vol_filename, 'variability_manual.nii.gz','variability.nii.gz', opts.metadata_path)