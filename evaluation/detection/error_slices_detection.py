import argparse
import glob
import os
from multiprocess.dummy import Pool
from tqdm import tqdm_notebook as tqdm
import nibabel as nib
import pandas as pd
from data_curation.helper_functions import move_smallest_axis_to_z
from evaluation.eval_utils.eval_functions import ErrorDetMetrics, dice, volume_difference_ratio


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", help="specifies test directory of segmentation errors data",
                        type=str, required=True)
    parser.add_argument("--output_path", help="specifies output path, expected to be .xlsx format",
                        type=str, required=True)
    parser.add_argument("--diff_truth_filename", help="specifies the filename of truth errors",
                        type=str, default='')
    return parser.parse_args()


def evaluate_error_slices_detection(cases_pathes):
    error_pred_scores = {}
    metrics = ['precision_all', 'recall_all', 'precision_100', 'recall_100', 'precision_300', 'recall_300',
               'precision_500', 'recall_500', 'precision_800', 'recall_800',
               'pred_num_error_slices',  'num_slices', 'dice_before_correction', 'dice_after_correction',
               'dice_after_rand_correction', 'dice_after_rand_nonzero_correction', 'vdr_before_correction',
               'vdr_after_correction', 'vdr_after_rand_correction', 'vdr_after_rand_nonzero_correction',
               'dice_percentage_correction', 'vdr_percentage_correction', 'dice_rand_nonzero_percentage_correction',
               'vdr_rand_nonzero_percentage_correction', 'dice_rand_percentage_correction',
               'vdr_rand_percentage_correction', 'dice_seq_percentage_correction', 'vdr_seq_percentage_correction']

    for metric in metrics:
        error_pred_scores[metric] = {}

    def process_sub(case_dir):
        subject_id = os.path.basename(case_dir)
        print('processing case ' + subject_id)
        prediction = nib.load(os.path.join(case_dir, 'prediction.nii.gz')).get_data()
        prediction, swap_axis = move_smallest_axis_to_z(prediction)
        prediction_soft = nib.load(os.path.join(case_dir, 'prediction_soft.nii.gz')).get_data()
        prediction_soft, swap_axis = move_smallest_axis_to_z(prediction_soft)
        truth = nib.load(os.path.join(case_dir, 'truth.nii.gz')).get_data()
        truth, swap_axis = move_smallest_axis_to_z(truth)
        mask = nib.load(os.path.join(case_dir, 'mask.nii.gz')).get_data()
        mask, swap_axis = move_smallest_axis_to_z(mask)
        truth_unified = nib.load(os.path.join(case_dir, 'truth_unified.nii.gz')).get_data()
        truth_unified, swap_axis = move_smallest_axis_to_z(truth_unified)

        error_det_metrics = ErrorDetMetrics()
        precision_all = error_det_metrics.error_slices_det_precision(truth, prediction, 0)
        error_pred_scores['precision_all'][subject_id] = precision_all
        recall_all = error_det_metrics.error_slices_det_recall(truth, prediction, 0, relaxation=False)
        error_pred_scores['recall_all'][subject_id] = recall_all
        precision_100 = error_det_metrics.error_slices_det_precision(truth, prediction, 100)
        error_pred_scores['precision_100'][subject_id] = precision_100
        recall_100 = error_det_metrics.error_slices_det_recall(truth, prediction, 100, relaxation=False)
        error_pred_scores['recall_100'][subject_id] = recall_100
        precision_300 = error_det_metrics.error_slices_det_precision(truth, prediction, 300)
        error_pred_scores['precision_300'][subject_id] = precision_300
        recall_300 = error_det_metrics.error_slices_det_recall(truth, prediction, 300, relaxation=False)
        error_pred_scores['recall_300'][subject_id] = recall_300
        precision_500 = error_det_metrics.error_slices_det_precision(truth, prediction, 500)
        error_pred_scores['precision_500'][subject_id] = precision_500
        recall_500 = error_det_metrics.error_slices_det_recall(truth, prediction, 500, relaxation=False)
        error_pred_scores['recall_500'][subject_id] = recall_500
        precision_800 = error_det_metrics.error_slices_det_precision(truth, prediction, 800)
        error_pred_scores['precision_800'][subject_id] = precision_800
        recall_800 = error_det_metrics.error_slices_det_recall(truth, prediction, 800, relaxation=False)
        error_pred_scores['recall_800'][subject_id] = recall_800
        dice_after_correction, vdr_after_correction, num_slices = error_det_metrics.eval_after_slice_correction(mask, prediction, truth_unified)
        error_pred_scores['pred_num_error_slices'][subject_id] = num_slices
        error_pred_scores['num_slices'][subject_id] = truth_unified.shape[2]
        error_pred_scores['dice_before_correction'][subject_id] = dice(truth_unified, mask)
        error_pred_scores['dice_after_correction'][subject_id] = dice_after_correction
        dice_after_rand_correction, vdr_after_rand_correction = error_det_metrics.eval_after_random_correction(mask, truth_unified, num_slices)
        error_pred_scores['dice_after_rand_correction'][subject_id] = dice_after_rand_correction
        dice_after_rand_nonzero_correction, vdr_after_rand_nonzero_correction = error_det_metrics.metrics_after_random_nonzero_correction(mask, truth_unified, num_slices)
        error_pred_scores['dice_after_rand_nonzero_correction'][subject_id] = dice_after_rand_nonzero_correction
        error_pred_scores['vdr_before_correction'][subject_id] = abs(volume_difference_ratio(truth_unified, mask))
        error_pred_scores['vdr_after_correction'][subject_id] = abs(vdr_after_correction)
        error_pred_scores['vdr_after_rand_correction'][subject_id] = abs(vdr_after_rand_correction)
        error_pred_scores['vdr_after_rand_nonzero_correction'][subject_id] = abs(vdr_after_rand_nonzero_correction)
        dice_scores, vdr_scores = error_det_metrics.eval_after_slice_correct_different_percentages(mask, prediction_soft, truth_unified)
        error_pred_scores['dice_percentage_correction'][subject_id] = dice_scores
        error_pred_scores['vdr_percentage_correction'][subject_id] = vdr_scores
        dice_scores_rand_nonzero, vdr_scores_rand_nonzero = error_det_metrics.eval_after_rand_nonzero_correct_percentages(mask, truth_unified)
        error_pred_scores['dice_rand_nonzero_percentage_correction'][subject_id] = dice_scores_rand_nonzero
        error_pred_scores['vdr_rand_nonzero_percentage_correction'][subject_id] = vdr_scores_rand_nonzero
        dice_scores_rand, vdr_scores_rand = error_det_metrics.eval_after_rand_correct_percentages(mask, truth_unified)
        error_pred_scores['dice_rand_percentage_correction'][subject_id] = dice_scores_rand
        error_pred_scores['vdr_rand_percentage_correction'][subject_id] = vdr_scores_rand
        dice_scores_seq, vdr_scores_seq = error_det_metrics.eval_after_sequential_correction_different_percentages(mask, truth_unified)
        error_pred_scores['dice_seq_percentage_correction'][subject_id] = dice_scores_seq
        error_pred_scores['vdr_seq_percentage_correction'][subject_id] = vdr_scores_seq

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_sub, cases_pathes), total=len(cases_pathes)))

    return error_pred_scores


if __name__ == '__main__':

    opts = get_arguments()
    dirs = glob.glob(os.path.join(opts.test_dir,'*'))
    cases_pathes = []
    for case_dir in dirs:
        if os.path.isdir(case_dir):
            cases_pathes.append(case_dir)
    error_pred_scores = evaluate_error_slices_detection(cases_pathes)
    pred_df = pd.DataFrame.from_dict(error_pred_scores)

    writer = pd.ExcelWriter(opts.output_path, engine='xlsxwriter')
    pred_df.to_excel(writer,  sheet_name='detection_eval')
    dice_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['dice_percentage_correction']).T
    dice_percentage_correction.to_excel(writer, sheet_name='dice_percentage_correction')
    vdr_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['vdr_percentage_correction']).T
    vdr_percentage_correction.to_excel(writer, sheet_name='vdr_percentage_correction')
    dice_rand_nonzero_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['dice_rand_nonzero_percentage_correction']).T
    dice_rand_nonzero_percentage_correction.to_excel(writer, sheet_name='dice_rand_nonzero_percent')
    vdr_rand_nonzero_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['vdr_rand_nonzero_percentage_correction']).T
    vdr_rand_nonzero_percentage_correction.to_excel(writer, sheet_name='vdr_rand_nonzero_percent')
    dice_rand_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['dice_rand_percentage_correction']).T
    dice_rand_percentage_correction.to_excel(writer, sheet_name='dice_rand_percent')
    vdr_rand_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['vdr_rand_percentage_correction']).T
    vdr_rand_percentage_correction.to_excel(writer, sheet_name='vdr_rand_percent')
    dice_seq_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['dice_seq_percentage_correction']).T
    dice_seq_percentage_correction.to_excel(writer, sheet_name='dice_seq_percent')
    vdr_seq_percentage_correction = pd.DataFrame.from_dict(error_pred_scores['vdr_seq_percentage_correction']).T
    vdr_seq_percentage_correction.to_excel(writer, sheet_name='vdr_seq_percent')
    writer.save()


