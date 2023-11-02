import subprocess
import os
import glob
from scripts.run_multiple_inferences import run_inference_tta, run_inference_tta_unlabeled
from scripts.run_multiple_evals import evalute_dir
from active_learning.self_training import SelfTraining
import json
from utils.read_write_data import list_load, save_nifti


def load_training_pathes_config(cfg_path, fold_id, input_path):
    """
    Load training pathes and network config
    """
    training_scans_list = list_load(os.path.join(cfg_path, fold_id, 'training_ids.txt'))
    training_samples_pathes = []
    for training_dir in training_scans_list:
        training_samples_pathes.append(os.path.join(input_path, training_dir))
    with open(os.path.join(cfg_path, 'config.json'), 'r') as f:
        config = json.load(f)

    return training_samples_pathes, config


if __name__ == "__main__":
    """
    This script evaluates teacher network and generates self-training configuration with unlabeled cases from teacher
    network results. cross validation
    It first runs teacher segmentation networks, then estimates dice and picks estimated dice examples
    """
    folds = 5
    detection_dirs = [525,526,528,529,530]
    segmentation_dirs = [531,532,533,534,535]

    log_dir_path = '/media/bella/8A1D-C0A6/Phd/log'
    out_dirname = 'placenta_FIESTA_unsupervised_cases'
    input_path = '/home/bella/Phd/data/placenta/placenta_clean/'
    st_input_path = '/home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/'
    metadata_path = '/home/bella/Phd/data/data_description/index_all_unified.csv'
    num_supervised_cases=30
    semi_supervised_cfgs_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks_uncertainty_above_th0.94_weights/'
    unsupervised_scans_dir = '/home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/'
    config_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config'
    min_dice=0.94
    include_uniqueness = False
    soft=True
    uncertainty = True
    weighting = True

    #evaluate teacher network on test ids
    #for i in range(folds):
     #    run_inference_tta(detection_dirs[i], segmentation_dirs[i], i, log_dir_path, input_path, out_dirname,
     #                      return_preds='True')
     #    res_dir = os.path.join(log_dir_path, str(segmentation_dirs[i]), 'output', out_dirname, 'test')
     #    evalute_dir(res_dir, metadata_path)

    #run teacher network on unlabeled cases
    for i in range(folds):
        # run_inference_tta_unlabeled(detection_dirs[i], segmentation_dirs[i], log_dir_path, st_input_path, out_dirname, 'True')
        tta_dir = os.path.join(log_dir_path, str(segmentation_dirs[i]), 'output', out_dirname)
        test_dir = os.path.join(tta_dir, 'test')
        st = SelfTraining(test_dir)
        dice_estimation_path = os.path.join(tta_dir, 'estimated_dice.xlsx')
        # st.run_eval_estimation_tta(dice_estimation_path, metadata_path=metadata_path)
        # st.clean_tta_results()
        if include_uniqueness is True:
            detection_network_path = os.path.join(log_dir_path, str(detection_dirs[i]))
            sup_training_cases, config = load_training_pathes_config(detection_network_path, str(i), input_path)
            training_cases, validation_cases = st.pick_best_dice_uniqueness(dice_estimation_path, min_dice, num_supervised_cases,
                                                                            config, sup_training_cases, detection_network_path,
                                                                            unsupervised_scans_dir)
        else:
            training_cases, validation_cases = st.pick_best_cases(dice_estimation_path, min_dice, num_supervised_cases)
        roi_dir = test_dir + '_cutted'
        if os.path.exists(roi_dir) is False:
            if uncertainty is True:
                 st.create_roi_data(test_dir, "uncertainty.nii")
            else:
                st.create_roi_data(test_dir, None)
      #  st.copy_best_cases(training_cases, validation_cases, os.path.join(tta_dir, 'test'), chosen_cases_dir, soft=soft)
        st.save_semi_supervised_configs(detection_dirs[i], segmentation_dirs[i],semi_supervised_cfgs_path,
                                        training_cases, validation_cases, test_dir, log_dir_path, config_dir,
                                        soft, uncertainty, weighting)
