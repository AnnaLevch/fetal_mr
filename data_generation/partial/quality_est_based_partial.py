import os
from evaluation.unsupervised_eval.quality_est_tta import QualityEstTTA
from utils.read_write_data import list_load, list_dump
from data_generation.partial.generate_whole_partial_data_and_splits import generate_comparison_config
import numpy as np
import glob
import pandas as pd


if __name__ == '__main__':
    """
    Create partial annotations based on worst quality estimation
    """
    data_path = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin'
    existing_partial_path = '/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/partial/'
    partial_config_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_body/partial'
    out_path = '/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/partial_active_selection/'
    log_dir = '/home/bella/Phd/code/code_bella/log/'
    metadata_dir = '/home/bella/Phd/data/data_description/index_all_unified.csv'
    annotations_ratio = 0.2
    partial_rand_pathes = glob.glob(os.path.join(existing_partial_path, '*'))
    segmentation_dirs = [807, 810, 813, 818]
    cascade = False
    training_cases = list_load('/home/bella/Phd/code/code_bella/fetal_mr/config/config_body/debug_split/cross_valid/2/training_ids.txt')
    comparison_similar_config = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_body/partial/config_dice_loss_2.json'
    num_new_examples = 7
    config_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config/'
    ##Estimate quality partial and whole##
    # for i in [0,1,2,3]:
    #     partial_dir = partial_rand_pathes[i]
    #     if os.path.exists(partial_dir) is False:
    #         os.mkdir(os.path.join(out_path, os.path.basename(partial_dir)))
  #       weights = np.asarray(list_load(os.path.join(partial_config_path, str(i), 'debug_split_10','weights0.2.txt')))
  #       partial_weights = np.where(weights == '0.2')
  #       partial_cases = np.array(training_cases)[partial_weights[0]]
  #       partials_lst_path = os.path.join(out_path, 'partials_list_{ind}.txt'.format(ind=str(i)))
  #       list_dump(partial_cases, partials_lst_path)
  #      tta_dir = os.path.join(log_dir, str(segmentation_dirs[i]), 'output/tta_res_' + str(i),'test')
  #      tta_quality_est = QualityEstTTA(tta_dir)
  #       tta_quality_est.run_tta_quality_est(None, segmentation_dirs[i], log_dir, partial_dir, 'tta_res_' + str(i), partials_lst_path)

  #       locations_to_annotate = tta_quality_est.estimate_partial_dir(annotations_ratio)
  #       locations_df = pd.DataFrame(locations_to_annotate).T
  #       locations_df.to_csv(os.path.join(partial_config_path, str(i), 'locations_to_annotate.csv'))
  #      tta_quality_est.estimate_whole_dir(metadata_dir,  os.path.join(partial_config_path, str(i), 'tta_whole_estimation.xlsx'))

  # #     tta_quality_est.clean_tta_results()

    # ##create partial data##
    # for i in [0,1,2,3]:
    #     data_df = pd.read_csv(os.path.join(partial_config_path, str(i), 'locations_to_annotate.csv'))
    #     whole_cases = np.asarray(list_load(os.path.join(partial_config_path, str(i), 'debug_split_10',
    #                                                     'training_ids.txt')))
    #     out_dir = os.path.join(out_path, 'partial0.2_10whole_tta_{ind}'.format(ind=i))
    #     generate_whole_partial_data(training_cases, whole_cases, data_path, out_dir, annotations_ratio, cascade,
    #                             tta_est_picking=data_df)

    ##Create whole configs##
    for i in [0,1,2,3]:
        tta_dir = os.path.join(log_dir, str(segmentation_dirs[i]), 'output/tta_res_' + str(i),'test')
        tta_quality_est = QualityEstTTA(tta_dir)
        quality_est_path = os.path.join(partial_config_path, str(i), 'tta_whole_estimation.xlsx')
        new_samples = tta_quality_est.pick_best_cases(quality_est_path, num_new_examples)
        whole_cases_lst = list_load(os.path.join(partial_config_path, str(i), 'debug_split_10','training_ids.txt'))
        validation_samples = list_load(os.path.join(partial_config_path, str(i), 'debug_split_10','validation_ids.txt'))
        all_cases = training_cases
        out_config_path = partial_config_path
        generate_comparison_config(whole_cases_lst, all_cases, i, comparison_similar_config, num_new_examples,
                                out_config_path, validation_samples, config_dir, new_samples=new_samples,
                                   comparison_dirname='comparison_active_selection')