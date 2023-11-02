import subprocess
import os
import pickle
import pandas as pd
import numpy as np


def get_scores(params_search_dir, dirs, output_postfix):
    scores_vol = {}
    mean_scores = {}
    for directory in dirs:
        pred_scores_mean = {}
        config_path = os.path.join(params_search_dir, directory, output_postfix)
        with open(config_path +'/pred_scores_vol.pkl', 'rb') as f:
                pred_scores_vol = pickle.load(f)

        pred_scores_mean['mean_dice'] = np.mean(list(pred_scores_vol['dice'].values()))
        pred_scores_mean['mean_hausdorff'] = np.mean(list(pred_scores_vol['hausdorff_lits'].values()))
        pred_scores_mean['mean_assd'] = np.mean(list(pred_scores_vol['assd_lits'].values()))
        pred_scores_mean['mean_hausdorff_robust'] = np.mean(list(pred_scores_vol['hausdorff_robust_lits'].values()))
        pred_scores_mean['mean_surface_dice'] = np.mean(list(pred_scores_vol['surface_dice'].values()))
        pred_scores_mean['mean_fnpl'] = np.mean(list(pred_scores_vol['false_negative_path_length'].values()))
        pred_scores_mean['mean_apl'] = np.mean(list(pred_scores_vol['added_path_length'].values()))
        scores_vol[directory] = pred_scores_vol
        mean_scores[directory] = pred_scores_mean
    return scores_vol, mean_scores


def summarize_params_search_to_excel(params_search_dir, dirs, output_postfix):
    scores_vol, mean_scores = get_scores(params_search_dir, dirs, output_postfix)

    excel_path = os.path.join(params_search_dir, 'params_search.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    df = pd.DataFrame.from_dict(mean_scores).T
    df = df.round(3)
    df.to_excel(writer,  sheet_name='vol_eval')
    writer.save()

if __name__ == "__main__":
    params_search_dir = '/home/bella/Phd/code/code_bella/log/params_search_FIESTA_2/'
    output_dir = 'output/FIESTA/test/'
    dir_content = os.listdir(params_search_dir)
    dirs = []
    for file in dir_content:
        path = os.path.join(params_search_dir,file)
        if((os.path.isdir(path)==True) and (file!='_sources')):
            dirs.append(file)

    for directory in dirs:
        ##run prediction on validation set
        args = "--input_path /home/bella/Phd/data/body/FIESTA/FIESTA_origin/ --output_folder {dir}{run}/output/FIESTA/" \
            " --config_dir {dir}{run}/  --preprocess window_1_99 --labeled True --ids_list" \
               " {dir}{run}/{split}/validation_ids.txt".format(dir=params_search_dir,run=directory, split=2)
        # args = "--input_path /home/bella/Phd/data/brain/FR_FSE/ --output_folder {dir}{run}/output/FR-FSE/ " \
        #    "--config_dir /home/bella/Phd/code/code_bella/log/brain21_24/22/ --config2_dir {dir}{run}/" \
        #    "  --labeled true --preprocess window_1_99 --ids_list {dir}{run}/debug_split/validation_ids.txt".format(dir=params_search_dir, run=directory)

        print('running with arguments:' + args)

        subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

        #apply evaluation
        path = os.path.join(params_search_dir,directory,output_dir)
        eval_args = '--src_dir ' + path + ' --metadata_path /home/bella/Phd/data/index_all.csv --save_to_excel false'
        print('running evaluation with arguments:')
        print(eval_args)
        subprocess.call("python -m evaluation.evaluate " + eval_args, shell=True)

    summarize_params_search_to_excel(params_search_dir, dirs, output_dir)