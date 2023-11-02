from utils.read_write_data import list_load, list_dump
import random
import os
import json


if __name__ == "__main__":
    """
    Choose additional examples from training set and create cross validation split
    """
    existing_examples_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/partial/partial_split/debug_split_6/'
    samples_to_choose_from_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/debug_split/'
    cross_valid_save_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/partial/partial_split/cross_valid/'
    similar_configs_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/cross_valid/teacher_networks/configs/'
    config_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config/'
    num_new_examples = 2
    num_folds = 5

    existing_examples = set(list_load(os.path.join(existing_examples_path, 'training_ids.txt')))
    all_training_samples = list_load(os.path.join(samples_to_choose_from_path, 'training_ids.txt'))
    validation_samples = list_load(os.path.join(samples_to_choose_from_path, 'validation_ids.txt'))
    new_samples = []
    for sample in  all_training_samples:
        if sample not in existing_examples:
            new_samples.append(sample)

    random.shuffle(new_samples)

    debug_split_path = os.path.join(cross_valid_save_path, 'debug_splits')
    relative_split_dir = debug_split_path.replace(config_dir, './config/')
    configs_path = os.path.join(cross_valid_save_path, 'configs')

    for fold in range(0, num_folds):
        additional_data = new_samples[fold*num_new_examples:fold*num_new_examples+num_new_examples]
        training_data = list(existing_examples) + additional_data
        test_data = []
        if os.path.exists(os.path.join(debug_split_path, str(fold))) is False:
            os.mkdir(os.path.join(debug_split_path, str(fold)))
        list_dump(training_data, os.path.join(debug_split_path, str(fold), 'training_ids.txt'))
        list_dump(validation_samples, os.path.join(debug_split_path, str(fold), 'validation_ids.txt'))
        list_dump(test_data, os.path.join(debug_split_path, str(fold), 'test_ids.txt'))

        detection_filename = 'config_all_small_' + str(fold) + '.json'
        segmentation_filename = 'config_roi_contour_dice_small_' + str(fold) + '.json'
        with open(os.path.join(similar_configs_path, detection_filename), 'r') as f:
            det_cfg = json.load(f)
        with open(os.path.join(similar_configs_path, segmentation_filename), 'r') as f:
            seg_cfg = json.load(f)
        det_cfg["data_dir"] = det_cfg["data_dir"][:-1] + '_partial_comparison'
        seg_cfg["data_dir"] = seg_cfg["data_dir"][:-1] + '_partial_comparison'

        det_cfg['split_dir']=relative_split_dir
        seg_cfg['split_dir']=relative_split_dir
        det_cfg['training_file']=os.path.join(relative_split_dir, str(fold), 'training_ids.txt')
        det_cfg['validation_file']=os.path.join(relative_split_dir,str(fold), 'validation_ids.txt')
        det_cfg['test_file']=os.path.join(relative_split_dir, str(fold), 'test_ids.txt')
        seg_cfg['training_file']=det_cfg['training_file']
        seg_cfg['validation_file']=det_cfg['validation_file']
        seg_cfg['test_file']=det_cfg['test_file']

        det_cfg_path = os.path.join(cross_valid_save_path, 'configs', detection_filename)
        seg_cfg_path = os.path.join(cross_valid_save_path, 'configs', segmentation_filename)
        with open(det_cfg_path, mode='w') as f:
            json.dump(det_cfg , f,  indent=2)
        with open(seg_cfg_path, mode='w') as f:
            json.dump(seg_cfg , f,  indent=2)