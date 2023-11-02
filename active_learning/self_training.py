import subprocess
import os
import glob
import pandas as pd
import shutil
import json
import numpy as np
from utils.read_write_data import list_load, list_dump
from data_generation.cut_roi_data import extract_mask_roi
from training.train_functions.training import get_last_model_path
from evaluation.features_extraction.feature_extractor import FeatureExtractor
import nibabel as nib
from utils.read_write_data import list_load, save_nifti
from scipy import ndimage
from active_learning.active_learning_uniqueness import ActiveLearningUniqueness


class SelfTraining:

    def __init__(self, tta_dir):
        self.tta_dir = tta_dir


    def run_eval_estimation_tta(self, output_path, metadata_path=None):
        """
        apply quality estimation using test time augmentations
        :param tta_dir: directory of tta results
        :param output_path: path of output excel file
        :return:
        """
        args = "--tta_dir {tta_dir} --output_path {output_path} --metadata_path {metadata_path}"\
            .format(tta_dir=self.tta_dir, output_path=output_path, metadata_path=metadata_path)
        print('running estimation using tta with arguments {args}'.format(args=args))
        subprocess.call("python -m evaluation.unsupervised_eval.estimate_metrics_TTA " + args, shell=True)


    def clean_tta_results(self):
        cases_pathes = glob.glob(os.path.join(self.tta_dir,'*'))
        for case_path in cases_pathes:
            tta_files = glob.glob(os.path.join(case_path,'tta*.nii.gz'))
            for tta_res in tta_files:
                os.remove(tta_res)


    def pick_best_dice_uniqueness(self, path, min_dice, num_supervised_cases, config, training_data_pathes,
                                  detection_network_path, unsupervised_scans_dir, layer_name='concatenate_1'):
        """
        Given quality estimation file and training examples pathes, pick best cases with estimated dice above min_dice
        and the most unique cases based on distances of intermediate feature vectors
        :param path:
        :param min_dice:
        :param num_supervised_cases:
        :param training_cases:
        :return:
        """

        max_num_cases = int(num_supervised_cases*1.5)#make sure the number of unsupervised cases is bounded. Save 5 cases for validation
        qe_df = pd.read_excel(path)
        qe_df.sort_values(by=['dice'], ascending=False)
        qe_df = qe_df[qe_df['dice']>=min_dice]
        above_thresh_cases = set(qe_df.iloc[:,0])
        print('number of retrieved unsupervised cases is: ' + str(len(above_thresh_cases)))
        unsupervised_pathes = []
        for unsupervised_case in above_thresh_cases:
            unsupervised_pathes.append(os.path.join(unsupervised_scans_dir, unsupervised_case))
        scale = config.get('scale', None)
        patch_size = config.get('input_shape')[1:]
        active_learning = ActiveLearningUniqueness(detection_network_path + '/')
        training_data_features = active_learning.load_extract_features(training_data_pathes, layer_name, scale, patch_size)
        unsupervised_data_features = active_learning.load_extract_features(unsupervised_pathes, layer_name, scale, patch_size)

        training_cases = []
        for num_unsupevised_cases in range(0,max_num_cases):
            samples_uniqueness = {}
            for unsup_case_path in above_thresh_cases:
                unsp_case_id = os.path.basename(unsup_case_path)
                distance_from_training = active_learning.calc_distance_from_training(unsupervised_data_features[unsp_case_id], training_data_features.values())
                samples_uniqueness[unsp_case_id] = distance_from_training
            df_unsupervised = pd.DataFrame(samples_uniqueness, index=[0]).T
            df_unsupervised.columns = ['distance']
            df_unsupervised = df_unsupervised.sort_values(by=['distance'], ascending=False)
            most_unique_case = df_unsupervised.iloc[0].name
            training_data_features[most_unique_case] = unsupervised_data_features[most_unique_case]
            training_cases.append(most_unique_case)
            del unsupervised_data_features[most_unique_case]
            above_thresh_cases.remove(most_unique_case)

        validation_cases = list(above_thresh_cases)[:5]
        return training_cases, validation_cases

    def pick_best_cases(self, path, min_dice, num_supervised_cases):
        """
        Given quality estimation file, pick best cases with dice above min_dice and mo more training cases than 1.5 supervised cases
        :param path: path to quality estimation file
        :param min_dice: minimum estimated dice
        :param num_supervised_cases: number of supervised training cases
        :return:
        """
        max_num_cases = int(num_supervised_cases*1.5 + 5)#make sure the number of unsupervised cases is bounded. Save 5 cases for validation
        qe_df = pd.read_excel(path)
        qe_df = qe_df.sort_values(by=['dice'], ascending=False)
        qe_df = qe_df[qe_df['dice']>=min_dice]
        chosen_cases = qe_df.iloc[:,0].to_list()
        if len(chosen_cases)> max_num_cases:
            chosen_cases = chosen_cases[:max_num_cases]
        training_cases = chosen_cases[:-5]
        validation_cases = chosen_cases[-5:]

        return training_cases, validation_cases

    def copy_best_cases(self, training_lst, valid_lst, input_dir, out_dir, soft=True):
        """
        Copy chosen cases
        :param training_lst:
        :param valid_lst:
        :param input_dir:
        :param out_dir:
        :return:
        """
        print('number of training cases: ' + str(len(training_lst)))
        print('number of validation cases: ' + str(len(valid_lst)))
        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)
        for case_id in training_lst:
            case_out_dir = os.path.join(out_dir, case_id)
            if os.path.exists(case_out_dir) is False:
                os.mkdir(case_out_dir)
            shutil.copy(os.path.join(input_dir, case_id, 'data.nii.gz'), os.path.join(case_out_dir,'data.nii.gz'))
            shutil.copy(os.path.join(input_dir, case_id, 'prediction.nii.gz'), os.path.join(case_out_dir,'prediction.nii.gz'))
            if soft is True:
                soft_data = nib.load(os.path.join(input_dir, case_id, 'prediction_soft.nii.gz')).get_data()
                save_nifti(soft_data[-1,:,:,:], os.path.join(case_out_dir,'prediction_soft.nii.gz'))

        for case_id in valid_lst:
            case_out_dir = os.path.join(out_dir, case_id)
            if os.path.exists(case_out_dir) is False:
                os.mkdir(case_out_dir)
            shutil.copy(os.path.join(input_dir, case_id, 'data.nii.gz'), os.path.join(case_out_dir, 'data.nii.gz'))
            shutil.copy(os.path.join(input_dir, case_id, 'prediction.nii.gz'), os.path.join(case_out_dir, 'prediction.nii.gz'))
            if soft is True:
                soft_data = nib.load(os.path.join(input_dir, case_id, 'prediction_soft.nii.gz')).get_data()
                save_nifti(soft_data[-1,:,:,:], os.path.join(case_out_dir,'prediction_soft.nii.gz'))



    def save_semi_supervised_configs(self, detection_dir, segmentation_dir, semi_supervised_cfgs_path,
                                     training_cases,validation_cases, chosen_cases_dir, log_path_dir, config_dir, soft,
                                     uncertainty, weighting):
        """
        Create student config and train/test config based on teacher config and chosen semi-supervised cases
        Change Training set list, data pathes, hdf5 save path
        :param detection_dir: path to detection network directory
        :param segmentation_dir: path to segmentation network directory
        :param semi_supervised_cfgs_path: output path to new semi-supervised configurations
        :return:
        """
        detection_dir = os.path.join(log_path_dir, str(detection_dir))
        segmentation_dir = os.path.join(log_path_dir, str(segmentation_dir))

        with open(os.path.join(detection_dir,'config.json'), 'r') as f:
            det_cfg = json.load(f)
        with open(os.path.join(segmentation_dir,'config.json'), 'r') as f:
            seg_cfg = json.load(f)

        cross_valid_dirname = det_cfg['split_dir'].split('/')[-1]

        #update training list
        train_set_lst = list_load(os.path.join(detection_dir, cross_valid_dirname,'training_ids.txt'))
        num_labeled = len(train_set_lst)
        num_pseudo_labeled = len(training_cases)
        train_set_lst = train_set_lst + training_cases
        test_set_lst = list_load(os.path.join(detection_dir, cross_valid_dirname,'test_ids.txt'))
        valid_set_lst = validation_cases
        base_split_dir = os.path.join(semi_supervised_cfgs_path, 'debug_split')
        if os.path.exists(base_split_dir) is False:
            os.mkdir(base_split_dir)
        split_dir = os.path.join(base_split_dir, cross_valid_dirname)
        relative_split_dir = split_dir.replace(config_dir, './config')

        if os.path.exists(split_dir) is False:
            os.mkdir(split_dir)
        list_dump(train_set_lst, os.path.join(split_dir,'training_ids.txt'))
        list_dump(valid_set_lst, os.path.join(split_dir,'validation_ids.txt'))
        list_dump(test_set_lst, os.path.join(split_dir,'test_ids.txt'))
        if weighting is True:
            pseudo_labeled_weight = num_labeled/num_pseudo_labeled
            labeled_weights = np.ones(num_labeled)
            pseudo_labeled_weights = np.ones(num_pseudo_labeled)*pseudo_labeled_weight
            weights = np.concatenate((labeled_weights, pseudo_labeled_weights))
            list_dump(weights, os.path.join(split_dir,'weights.txt'))
            det_cfg['samples_weights'] = os.path.join(relative_split_dir,'weights.txt')
            seg_cfg['samples_weights'] = os.path.join(relative_split_dir,'weights.txt')


        det_cfg['split_dir']=relative_split_dir
        seg_cfg['split_dir']=relative_split_dir
        det_cfg['training_file']=os.path.join(relative_split_dir,'training_ids.txt')
        det_cfg['validation_file']=os.path.join(relative_split_dir,'validation_ids.txt')
        det_cfg['test_file']=os.path.join(relative_split_dir,'test_ids.txt')
        seg_cfg['training_file']=det_cfg['training_file']
        seg_cfg['validation_file']=det_cfg['validation_file']
        seg_cfg['test_file']=det_cfg['test_file']
        #update scans dir and data dir
        relative_cases_dir = chosen_cases_dir.replace(log_path_dir, '../log')
        det_cfg['scans_dir'] = det_cfg['scans_dir'] + ';' + relative_cases_dir
        seg_cfg['scans_dir'] = seg_cfg['scans_dir'] + ';' + relative_cases_dir + '_cutted'

        if soft is True:
            if uncertainty is True:
                det_cfg['data_dir'] = det_cfg['data_dir'][:-1] + '_self_supervised_soft_uncertainty'
                seg_cfg['data_dir'] = seg_cfg['data_dir'][:-1] + '_self_supervised_soft_uncertainty'
                det_cfg['training_modalities'] = ["volume;data","truth;prediction_soft", "uncertainty"]
                seg_cfg['training_modalities'] = ["volume;data","truth;prediction_soft", "uncertainty"]
                det_cfg['mask_shape']=[1,128,128,48]
                seg_cfg['mask_shape']=[1,128,128,48]
                det_cfg['loss'] = 'dice_with_uncertainty_loss'
                seg_cfg['loss'] = 'contour_dice_tolerance_and_dice_uncertainty_loss'
                det_cfg['u_th'] = 0
                seg_cfg['u_th'] = 0
            else:
                det_cfg['data_dir'] = det_cfg['data_dir'][:-1] + '_self_supervised_soft'
                seg_cfg['data_dir'] = seg_cfg['data_dir'][:-1] + '_self_supervised_soft'
                det_cfg['training_modalities'] = ["volume;data","truth;prediction_soft"]
                seg_cfg['training_modalities'] = ["volume;data","truth;prediction_soft"]
        else:
            det_cfg['data_dir'] = det_cfg['data_dir'][:-1] + '_self_supervised'
            seg_cfg['data_dir'] = seg_cfg['data_dir'][:-1] + '_self_supervised'
            det_cfg['training_modalities'] = ["volume;data","truth;prediction"]
            seg_cfg['training_modalities'] = ["volume;data","truth;prediction"]


        #update old model path
        det_model_path = get_last_model_path(os.path.join(detection_dir,'epoch_'))
        det_model_path = det_model_path.replace(log_path_dir, '../log')
        det_cfg['old_model_path'] = det_model_path
        seg_model_path = get_last_model_path(os.path.join(segmentation_dir,'epoch_'))
        seg_model_path = seg_model_path.replace(log_path_dir, '../log')
        seg_cfg['old_model_path'] = seg_model_path

        if os.path.exists(os.path.join(semi_supervised_cfgs_path, 'configs')) is False:
            os.mkdir(os.path.join(semi_supervised_cfgs_path, 'configs'))
        det_cfg_path = os.path.join(semi_supervised_cfgs_path, 'configs', 'config_all_semi_supervised_' + cross_valid_dirname + '.json')
        with open(det_cfg_path, mode='w') as f:
            json.dump(det_cfg , f,  indent=2)
        seg_cfg_path = os.path.join(semi_supervised_cfgs_path, 'configs', 'config_roi_semi_supervised_' + cross_valid_dirname + '.json')
        with open(seg_cfg_path, mode='w') as f:
            json.dump(seg_cfg , f,  indent=2)


    def create_roi_data(self, data_path, mask_filename):
        """
        Extract ROI data for segmentation network training and save in a directory with "cutted" postfix
        :param data_path: path to data with label mask
        :return:
        """
        padding = np.array([16, 16, 8])
        extract_mask_roi(data_path, data_path + '_cutted', padding, 'data.nii', 'prediction_soft.nii', mask_filename)






