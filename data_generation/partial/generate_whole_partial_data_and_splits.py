import json
import os
import random
import shutil

import nibabel as nib
import numpy as np

from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from data_generation.cut_roi_data import cut_with_roi_data_np
from data_generation.partial.create_partial_annotations import generate_partial_data_rand_picking, generate_partial_data_tta_picking
from utils.read_write_data import list_load, list_dump, save_nifti


def rand_sample_whole_split(all_cases, num_cases, split_path, test_lst):
    """
    pick randomly the whole cases
    :param all_cases: cases to pick from
    :param num_cases: number of whole cases
    :param out_path: path to debug split output
    :return:
    """
    if os.path.exists(os.path.join(split_path, 'training_ids.txt')):
        print('split path already exists for: ' + split_path)
        return list_load(os.path.join(split_path, 'training_ids.txt'))

    chosen_cases = random.sample(all_cases, num_cases)
    if os.path.exists(split_path) is False:
        os.mkdir(split_path)
    list_dump(chosen_cases, os.path.join(split_path, 'training_ids.txt'))
    list_dump(test_lst, os.path.join(split_path, 'test_ids.txt'))
    shutil.copy(os.path.join(all_cases_dir, 'validation_ids.txt'), os.path.join(split_path, 'validation_ids.txt'))

    return chosen_cases



def generate_whole_partial_data(all_cases, whole_cases, data_path, out_data_path, annotations_ratio, cascade,
                                tta_est_picking = None, partial_weight = None):
    """
    create whole and partial data including corresponding masks
    :param all_cases: all training cases
    :param whole_cases: fully annotated cases out pf total cases
    :param data_path: path to whole data
    :param out_data_path: output path
    :return:
    """
    if os.path.exists(out_data_path) is False:
        os.mkdir(out_data_path)
    if cascade is True and os.path.exists(out_data_path + '_cutted') is False:
        os.mkdir(out_data_path + '_cutted')
    weights = []

    if partial_weight is None:
        partial_weight = annotations_ratio

    for case_id in all_cases:
        det_out_dir = os.path.join(out_data_path, case_id)
        if os.path.exists(det_out_dir) is False:
            os.mkdir(det_out_dir)
        if cascade is True:
            seg_out_dir = os.path.join(out_data_path + '_cutted', case_id)
            if os.path.exists(seg_out_dir) is False:
                os.mkdir(seg_out_dir)

        #copy volume for detection data
        volume_path = os.path.join(data_path, case_id, 'volume.nii')
        if os.path.exists(volume_path) is False:
            volume_path = os.path.join(data_path, case_id, 'volume.nii.gz')
            if os.path.exists(volume_path) is False:
                volume_path = os.path.join(data_path, case_id, 'data.nii.gz')
        volume = nib.load(volume_path).get_data()
        save_nifti(volume, os.path.join(det_out_dir, 'volume.nii.gz'))

        #create either partial or whole annotations based on whole_cases list
        truth_path = os.path.join(data_path, case_id, 'truth.nii')
        if os.path.exists(truth_path) is False:
            truth_path = os.path.join(data_path, case_id, 'truth.nii.gz')
        y_true = nib.load(truth_path).get_data()
        if cascade is True:
            seg_gt_save_path = os.path.join(seg_out_dir, 'truth.nii.gz')
            seg_vol_save_path = os.path.join(seg_out_dir, 'volume.nii.gz')
            seg_uncertainty_save_path = os.path.join(seg_out_dir, 'uncertainty.nii.gz')

        if case_id in whole_cases:
            weights.append(1)
            uncertainty = np.zeros_like(y_true)
            save_nifti(uncertainty, os.path.join(det_out_dir, 'uncertainty.nii.gz'))
            save_nifti(y_true, os.path.join(det_out_dir, 'truth.nii.gz'))
            if cascade is True:
                #save to segmentation dir
                cut_with_roi_data_np(y_true, volume, uncertainty, seg_gt_save_path, seg_vol_save_path, seg_uncertainty_save_path,
                                     'uncertainty.nii')
        else:
            weights.append(partial_weight)
            swapped_y_true, swap_axis = move_smallest_axis_to_z(y_true)
            if tta_est_picking is not None:
                partial_truth, uncertainty = generate_partial_data_tta_picking(swapped_y_true, case_id, tta_est_picking)
            else:
                partial_truth, uncertainty = generate_partial_data_rand_picking(swapped_y_true, annotations_ratio, case_id)
            partial_truth = swap_to_original_axis(swap_axis, partial_truth)
            uncertainty = swap_to_original_axis(swap_axis, uncertainty)

            save_nifti(uncertainty, os.path.join(det_out_dir, 'uncertainty.nii.gz'))
            save_nifti(partial_truth, os.path.join(det_out_dir, 'partial_truth.nii.gz'))

            if cascade is True:
                cut_with_roi_data_np(y_true, volume, uncertainty, seg_gt_save_path, seg_vol_save_path,
                                     seg_uncertainty_save_path, 'uncertainty.nii')
                partial_save_path = os.path.join(seg_out_dir, 'partial_truth.nii.gz')
                cut_with_roi_data_np(y_true, volume, partial_truth, seg_gt_save_path, seg_vol_save_path, partial_save_path,
                                     'partial_truth.nii')
                if os.path.exists(os.path.join(seg_out_dir, 'truth.nii.gz')):
                    os.remove(os.path.join(seg_out_dir, 'truth.nii.gz'))


def save_weights(split_path, whole_cases, partial_weight, all_cases):

    weights = []
    for case in all_cases:
        if case in whole_cases:
            weights.append(1)
        else:
            weights.append(partial_weight)

    list_dump(weights, os.path.join(split_path, 'weights.txt'))


def create_configs_from_similar(out_split_config_path, similar_configs_path, config_dir, rand_ind, out_split_path,
                                whole_cases, partial_weight, all_cases, out_data_path, annotation_ratio):
    """
    Create configuration files for each one of the whole cases randomizations for cascade framework when there is
    existing config for partial data
    :param out_split_config_path:
    :return:
    """
    num_whole = len(whole_cases)
    with open(os.path.join(similar_configs_path, 'config_all_partial.json'
            .format(annotation_ratio=annotation_ratio, num_whole=num_whole)), 'r') as f:
        det_cfg = json.load(f)
    with open(os.path.join(similar_configs_path, 'config_roi_partial.json').
                      format(annotation_ratio=annotation_ratio,num_whole=num_whole, ), 'r') as f:
        seg_cfg = json.load(f)

    partial_config_dir = os.path.dirname(similar_configs_path)
    relative_split_dir = partial_config_dir.replace(config_dir, './config/')

    det_cfg["scans_dir"] = out_data_path + "/partial{annotation_ratio}_{num_whole}whole_"\
        .format(num_whole=num_whole, annotation_ratio=annotation_ratio) + str(rand_ind)
    seg_cfg["scans_dir"] = det_cfg["scans_dir"] + '_cutted'
    det_cfg["data_dir"] = det_cfg["data_dir"][:-1] + '_' + str(rand_ind)
    seg_cfg["data_dir"] = seg_cfg["data_dir"][:-1] + '_' + str(rand_ind)
    det_cfg["samples_weights"] = relative_split_dir + "/{index}/debug_split_{num_whole}" \
                                 "/weights.txt".format(index=str(rand_ind), num_whole=num_whole)
    seg_cfg["samples_weights"] = relative_split_dir + "/{index}/debug_split_{num_whole}" \
                                 "/weights.txt".format(index=str(rand_ind), num_whole=num_whole)
    det_cfg_path = os.path.join(out_split_config_path, 'config_all_partial_{num_whole}whole_{annotation_ratio}.json'
                                .format(num_whole=num_whole, annotation_ratio=annotation_ratio))
    seg_cfg_path = os.path.join(out_split_config_path, 'config_roi_partial_{num_whole}whole_{annotation_ratio}.json'
                                .format(num_whole=num_whole, annotation_ratio=annotation_ratio))
    det_cfg['early_stop']=50
    seg_cfg['early_stop']=50
    det_cfg['batch_size']=8
    seg_cfg['batch_size']=8
    with open(det_cfg_path, mode='w') as f:
        json.dump(det_cfg , f,  indent=2)
    with open(seg_cfg_path, mode='w') as f:
        json.dump(seg_cfg , f,  indent=2)

    save_weights(out_split_path, whole_cases, partial_weight, all_cases)


def create_config_from_similar(out_split_config_path, similar_config_path, config_dir, rand_ind, split_path,
                               whole_cases, partial_weight, all_cases, out_data_path, annotation_ratio):
    """
    Create configuration files for each one of the whole cases randomizations for single segmentation network when there
    Also, save weights
    is existing config for partial data (body)
    :param out_split_config_path:
    :return:
    """

    with open(similar_config_path, 'r') as f:
        seg_cfg = json.load(f)

    relative_data_path = out_data_path.replace(config_dir, './config/')
    relative_split_path = split_path.replace(config_dir, './config/')
    num_whole = len(whole_cases)

    seg_cfg["scans_dir"] = relative_data_path + "partial{annotation_ratio}_{num_whole}whole_"\
        .format(num_whole=num_whole, annotation_ratio=annotation_ratio) + str(rand_ind)
    seg_cfg["data_dir"] = seg_cfg["data_dir"][:-1] + '_' + str(rand_ind)
    seg_cfg["samples_weights"] = "{split_path}/weights.txt".format(split_path=relative_split_path, index = str(rand_ind))
    seg_cfg['early_stop']=50
    seg_cfg['batch_size']=8

    seg_cfg_path = os.path.join(out_split_config_path, 'config_partial_whole{num_whole}_0.2.json'.format(num_whole=num_whole))
    with open(seg_cfg_path, mode='w') as f:
        json.dump(seg_cfg , f,  indent=2)

    save_weights(split_path, whole_cases, partial_weight, all_cases)


def generate_comparison_config(whole_cases_lst, all_cases, i, comparison_similar_config, num_new_examples,
                                out_config_path, validation_samples, config_dir, new_samples=None,
                               comparison_dirname='comparison'):
    """
    Generate and save comparison configs for training examples including whole_cases_lst - for cases with only
    segmentation network
    :param whole_cases_lst:
    :param all_cases:
    :param i:
    :param comparison_similar_configs:
    :return:
    """
    comparison_path = os.path.join(out_config_path, str(i), comparison_dirname)
    if os.path.exists(comparison_path) is False:
        os.mkdir(comparison_path)

    #generate training set and update debug split
    if new_samples is None:#if there is no new_samples list, generate a random one
        new_samples = []
        for sample in all_cases:
            if sample not in whole_cases_lst:
                new_samples.append(sample)
        random.shuffle(new_samples)
    additional_data = new_samples[0:num_new_examples]
    training_data = whole_cases_lst + additional_data
    test_data = []
    split_dir = os.path.join(comparison_path, 'debug_split')
    if os.path.exists(split_dir) is False:
        os.mkdir(split_dir)
    list_dump(training_data, os.path.join(comparison_path, 'debug_split', 'training_ids.txt'))
    list_dump(validation_samples, os.path.join(comparison_path, 'debug_split', 'validation_ids.txt'))
    list_dump(test_data, os.path.join(comparison_path, 'debug_split', 'test_ids.txt'))

    #create config
    with open(comparison_similar_config, 'r') as f:
        seg_cfg = json.load(f)
    dir_name = os.path.basename(os.path.dirname(comparison_similar_config))
    seg_cfg["data_dir"] = "../data/body/" + comparison_dirname + '_' + str(i)
    relative_split_dir = split_dir.replace(config_dir, './config/')
    seg_cfg['split_dir'] = relative_split_dir
    seg_cfg['training_file'] = os.path.join(relative_split_dir, 'training_ids.txt')
    seg_cfg['validation_file'] = os.path.join(relative_split_dir, 'validation_ids.txt')
    seg_cfg['test_file'] = os.path.join(relative_split_dir, 'test_ids.txt')
    seg_cfg['early_stop']=50
    seg_cfg['batch_size']=8
    seg_cfg_path = os.path.join(out_config_path, str(i), comparison_dirname, 'config_small.json')

    with open(seg_cfg_path, mode='w') as f:
        json.dump(seg_cfg , f,  indent=2)


def generate_comparison_configs(whole_cases_lst, all_cases, i, comparison_similar_configs, num_new_examples,
                                out_config_path, validation_samples, config_dir):
    """
    Generate and save comparison configs for training examples including whole_cases_lst - for Cascade framework only
    (detection and segmentation networks)
    :param whole_cases_lst:
    :param all_cases:
    :param i:
    :param comparison_similar_configs:
    :return:
    """
    comparison_path = os.path.join(out_config_path, str(i), 'comparison')
    if os.path.exists(comparison_path) is False:
        os.mkdir(comparison_path)

    #generate training set and update debug split
    new_samples = []
    for sample in all_cases:
        if sample not in whole_cases_lst:
            new_samples.append(sample)
    random.shuffle(new_samples)
    additional_data = new_samples[0:num_new_examples]
    training_data = whole_cases_lst + additional_data
    test_data = []
    split_dir = os.path.join(comparison_path, 'debug_split')
    if os.path.exists(split_dir) is False:
        os.mkdir(split_dir)
    list_dump(training_data, os.path.join(comparison_path, 'debug_split', 'training_ids.txt'))
    list_dump(validation_samples, os.path.join(comparison_path, 'debug_split', 'validation_ids.txt'))
    list_dump(test_data, os.path.join(comparison_path, 'debug_split', 'test_ids.txt'))

    #create configs
    with open(os.path.join(comparison_similar_configs, 'config_all.json'), 'r') as f:
        det_cfg = json.load(f)
    with open(os.path.join(comparison_similar_configs, 'config_roi_contour_dice.json'), 'r') as f:
        seg_cfg = json.load(f)

    det_cfg["data_dir"] = "../data/placenta/placenta_all_comparison_randomization_" + str(i)
    seg_cfg["data_dir"] = "../data/placenta/placenta_roi_comparison_randomization_" + str(i)
    relative_split_dir = split_dir.replace(config_dir, './config/')
    det_cfg['split_dir']=relative_split_dir
    seg_cfg['split_dir']=relative_split_dir
    det_cfg['training_file']=os.path.join(relative_split_dir, 'training_ids.txt')
    det_cfg['validation_file']=os.path.join(relative_split_dir, 'validation_ids.txt')
    det_cfg['test_file']=os.path.join(relative_split_dir, 'test_ids.txt')
    seg_cfg['training_file']=det_cfg['training_file']
    seg_cfg['validation_file']=det_cfg['validation_file']
    seg_cfg['test_file']=det_cfg['test_file']
    det_cfg['early_stop']=50
    seg_cfg['early_stop']=50
    det_cfg['batch_size']=8
    seg_cfg['batch_size']=8

    det_cfg_path = os.path.join(out_config_path, str(i), 'comparison', 'config_all_small.json')
    seg_cfg_path = os.path.join(out_config_path, str(i), 'comparison', 'config_roi_contour_dice_small.json')
    with open(det_cfg_path, mode='w') as f:
        json.dump(det_cfg , f,  indent=2)
    with open(seg_cfg_path, mode='w') as f:
        json.dump(seg_cfg , f,  indent=2)



if __name__ == '__main__':
    # # ##############Placenta##############
    # data_path = '/home/bella/Phd/data/placenta/ASL/placenta_training_valid_set/'
    # all_cases_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/TRUFI/ASL_data/debug_split/'
    # out_config_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/TRUFI/ASL_data/partial/'
    # out_data_path = '/home/bella/Phd/data/placenta/ASL/partial_9/'
    # config_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config/'
    # comparison_similar_configs = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/TRUFI/ASL_data/partial/'
    # similar_configs_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/TRUFI/ASL_data/partial/'
    # partial_weight = 1
    # num_new_samples = 9
    # num_cases = 0
    # cascade = True
    # start_ind = 0

    ###########Body##############
    data_path = 'D:\\users\\bella\\data\\body\\TRUFI\\TRUFI\\'
    all_cases_dir = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\config_body\\TRUFI\\debug_split_TRUFI\\debug_split_30\\'
    out_config_path = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\config_body\\TRUFI\\partial_6\\'
    out_data_path = 'D:\\users\\bella\\data\\body\\TRUFI\\partial_6\\'
    comparison_similar_config = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\config_body\\TRUFI\\partial_6\\config_TRUFI.json'
    similar_config_path = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\config_body\\TRUFI\\partial_6\\config_TRUFI_partial.json'
    config_dir = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\'
    comparison_similar_configs = None
    similar_configs_path = None
    partial_weight = 1
    num_new_samples = 6
    num_cases = 0
    cascade = False
    start_ind = 0

    #####Body and Placenta
    config_dir = 'D:\\users\\bella\\code\\code_bella\\fetal_mr\\config\\'
    annotations_ratio = 0.2
    num_random = 4

    all_cases = list_load(os.path.join(all_cases_dir, 'training_ids.txt'))
    validation_samples = list_load(os.path.join(all_cases_dir, 'validation_ids.txt'))
    test_lst = []
    for i in range(start_ind,num_random):
        if os.path.exists(os.path.join(out_config_path, str(i))) is False:
            os.mkdir(os.path.join(out_config_path, str(i)))
        split_path = os.path.join(out_config_path, str(i), 'debug_split_' + str(num_cases))
        whole_cases_lst = rand_sample_whole_split(all_cases, num_cases, split_path, test_lst)
        out_split_data_path = os.path.join(out_data_path, 'partial{annotations_ratio}_{num_cases}whole_{i}'.format(
            num_cases=num_cases, i=str(i), annotations_ratio=annotations_ratio))
        if cascade is True:
            generate_comparison_configs(whole_cases_lst, all_cases, i, comparison_similar_configs, num_new_samples,
                                    out_config_path, validation_samples, config_dir)
        else:
            generate_comparison_config(whole_cases_lst, all_cases, i, comparison_similar_config, num_new_samples,
                                    out_config_path, validation_samples, config_dir)
        generate_whole_partial_data(all_cases + validation_samples, whole_cases_lst + validation_samples, data_path, out_split_data_path, annotations_ratio,
                                    cascade)
        out_split_config_path = os.path.join(out_config_path, str(i))
        if cascade is True:
            create_configs_from_similar(out_split_config_path, similar_configs_path, config_dir, i, split_path,
                                        whole_cases_lst, partial_weight, all_cases, out_data_path, annotations_ratio)
        else:
            create_config_from_similar(out_split_config_path, similar_config_path, config_dir, i, split_path,
                                        whole_cases_lst, partial_weight, all_cases, out_data_path, annotations_ratio)

