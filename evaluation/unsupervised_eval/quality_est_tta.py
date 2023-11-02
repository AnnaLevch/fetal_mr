import os
import numpy as np
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from evaluation.eval_utils.eval_functions import dice
import nibabel as nib
import glob
from scripts.run_multiple_inferences import run_inference_tta_unlabeled
from evaluation.unsupervised_eval.estimate_metrics_TTA import estimate_metrics_all, write_estimation_to_excel
import copy
import pandas as pd


class QualityEstTTA:

    def __init__(self, tta_dir):
        self.tta_dir = tta_dir

    def run_tta_quality_est(self, detection_dir,segmentation_dir, log_dir_path, input_path, out_dirname, ids_list):
        """
        run network with TTA and save all TTA results
        :param detection_dir:
        :param segmentation_dir:
        :param log_dir_path:
        :param input_path:
        :param out_dirname:
        :param ids_list:
        :return:
        """
        run_inference_tta_unlabeled(detection_dir, segmentation_dir, log_dir_path, input_path, out_dirname,
                                    return_preds='True', ids_list=ids_list)

    def estimate_partial_dir(self, annotations_ratio):
        cases_dirs = glob.glob(os.path.join(self.tta_dir,'*'))
        locations_to_annotate = {}
        i=0
        for case_dir in cases_dirs:
            # if i==2:
            #     break
            max_dice_diff, max_start, max_end, estimated_dice_origin = self.estimate_partial(case_dir, annotations_ratio)
            locations_to_annotate[case_dir]={}
            locations_to_annotate[case_dir]['max_dice_diff'] = max_dice_diff
            locations_to_annotate[case_dir]['max_start'] = max_start
            locations_to_annotate[case_dir]['max_end'] = max_end
            locations_to_annotate[case_dir]['estimated_dice_origin'] = estimated_dice_origin
            i+=1

        return locations_to_annotate


    def clean_tta_results(self):
        cases_pathes = glob.glob(os.path.join(self.tta_dir,'*'))
        for case_path in cases_pathes:
            tta_files = glob.glob(os.path.join(case_path,'tta*.nii.gz'))
            for tta_res in tta_files:
                os.remove(tta_res)


    def load_all_tta(self, tta_predictions_pathes):
        """
        Load all TTA results
        :param tta_predictions_pathes:
        :return:
        """
        tta_data = {}
        for tta_path in tta_predictions_pathes:
            tta = nib.load(tta_path).get_data()
            tta, swap_axis = move_smallest_axis_to_z(tta)
            tta_data[tta_path] = tta
        return tta_data


    def estimate_partial(self, case_dir, annotations_ratio):
        """
        perform quality estimation on partial data at each one of the non-zero locations
        :param case_dir: location of the case directory with TTA results
        :param annotations_ratio: percentage of annotated slices from the whole scan
        :return:
        """
        tta_predictions_pathes = glob.glob(os.path.join(case_dir,'tta*'))
        subject_id = os.path.basename(case_dir)
        print('partial estimation of case ' + subject_id)
        mean_prediction = nib.load(os.path.join(case_dir, 'prediction.nii.gz')).get_data()
        mean_prediction, swap_axis = move_smallest_axis_to_z(mean_prediction)

        nonzero_slices = set(np.nonzero(mean_prediction)[2])
        num_slices = int(np.round(len(nonzero_slices) * annotations_ratio))
        max_dice_diff = 0.0
        tta_data = self.load_all_tta(tta_predictions_pathes)

        for slice_number in nonzero_slices:
            start_ind = slice_number - int(np.round(num_slices/2))
            if start_ind < 0:
                start_ind = 0
            elif start_ind + num_slices >= mean_prediction.shape[2]:
                start_ind = mean_prediction.shape[2]-num_slices-1
            end_ind = start_ind+num_slices

            tta_dice_diff = []
            tta_dice_origin = []
            for tta_path in tta_predictions_pathes:
                tta = copy.deepcopy(tta_data[tta_path])
                tta_dice = dice(mean_prediction[:,:,:], tta[:,:,:])#dice before replacement
                tta_dice_origin.append(tta_dice)

                tta[:,:,start_ind:(end_ind+1)] = mean_prediction[:,:,start_ind:(end_ind+1)]
                tta_dice_after_replacement = dice(mean_prediction[:,:,:], tta[:,:,:])
                tta_dice_diff.append(tta_dice_after_replacement-tta_dice)


            estimated_dice_diff = np.median(tta_dice_diff)
            estimated_dice_origin = np.median(tta_dice_origin)
            if estimated_dice_diff > max_dice_diff:
                max_dice_diff = estimated_dice_diff
                max_start = start_ind
                max_end = end_ind

        print('maximum dice diff is: ' + str(max_dice_diff) + 'between slices: ' + str(max_start) + ':' + str(max_end))

        return max_dice_diff, max_start, max_end, estimated_dice_origin


    def estimate_whole_dir(self, metadata_dir, output_path):
        """
        Estimate dice for all case and save estimations to excel file
        :param metadata_dir: path to metadata
        :param output_path: path to output excel file
        :return:
        """
        print('calculating metrics')
        cases_dirs = glob.glob(os.path.join(self.tta_dir,'*'))
        estimated_vol_metrics, estimated_vol_metrics_with_scaling,  estimated_2D_metrics, estimated_calculations = \
            estimate_metrics_all(cases_dirs, metadata_path=metadata_dir)
        print('writing to excel')
        write_estimation_to_excel(output_path, estimated_vol_metrics, estimated_vol_metrics_with_scaling,
                                  estimated_2D_metrics, estimated_calculations, self.tta_dir, None, False)


    def pick_best_cases(self, quality_est_path, num_samples):
        """
        Given quality estimation file, pick best num_samples cases
        :param quality_est_path: path to quality estimation file
        :param min_dice: minimum estimated dice
        :param num_supervised_cases: number of supervised training cases
        :return:
        """
        qe_df = pd.read_excel(quality_est_path)
        qe_df = qe_df.sort_values(by=['dice'], ascending=False)
        chosen_cases = qe_df.iloc[:,0].to_list()
        chosen_cases = chosen_cases[:num_samples]

        return chosen_cases
