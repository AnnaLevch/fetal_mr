import pandas as pd
import numpy as np
from data_curation.helper_functions import resolution_from_scan_name, get_spacing_between_slices, get_FOV, get_metadata_value
from utils.read_write_data import list_load
import json
import os
import nibabel as nib
import glob
"""
This script calculates mean resolution and FOV of a training set and stores them in norm_params
Cases that don't have this information are not taken into account
"""

def get_res_fov(scan_name):
    resolution = resolution_from_scan_name(scan_name)

    if(resolution == None):
        return None
    else:
        spacing = get_spacing_between_slices(scan_name, df=df)
        if(spacing != None):
            resolution[2] = spacing
        print('spacing is: ' + str(resolution[2]))
    return resolution


def get_pathes(data_pathes, train_set_lst):
    pathes_lst = {}
    training_set_dict = {}
    splitted_pathes = data_pathes.split(';')
    for data_path in splitted_pathes:
        pathes = glob.glob(os.path.join(data_path,'*'))
        for path in pathes:
            dirname = os.path.basename(path)
            pathes_lst[dirname] = path

    for scan in train_set_lst:
        if(scan in pathes_lst):
            training_set_dict[scan] = pathes_lst[scan]

    return training_set_dict


if __name__ == "__main__":
    """
    Updating norm_params with median resolution, median fov, metadata path and training set histogram
    """
  #  metadata_path = '/home/bella/Phd/data/data_description/index_all.csv'
    metadata_path = '/home/bella/Phd/data/data_description/data_Elka/FIESTA_body.csv'
    train_set_ids = '/home/bella/Phd/code/code_bella/log/377/debug_split_small_fetuses/training_ids.txt'
    norm_params_path = '/home/bella/Phd/code/code_bella/log/377/norm_params.json'
    data_pathes = '/home/bella/Phd/data/body/FIESTA/small_fetuses/'


    df = pd.read_csv(metadata_path, encoding ="unicode_escape")
    train_set_lst = list_load(train_set_ids)
    ids_pathes_map = get_pathes(data_pathes, train_set_lst)

    res_lst = []
    fov_lst = []
    hist_list = []

    for scan_name in train_set_lst:
        if(scan_name not in ids_pathes_map):
            continue
        vol_path = os.path.join(ids_pathes_map[scan_name], "volume.nii.gz")
        if(not os.path.exists(vol_path)):
            vol_path = os.path.join(ids_pathes_map[scan_name], "volume.nii")
        volume = nib.load(vol_path).get_data()
        res = get_res_fov(scan_name)
        hist = np.histogram(volume, np.arange(0,2001))
        hist_list.append(hist[0])
        if(res!= None):
            res_lst.append(res)
        fov = get_FOV(scan_name, df=df)
        if(fov!=None):
            fov_lst.append(fov)

        print('in scan ' + scan_name +'resolution is: ' + str(res) + ', fov is: ' + str(fov))

    median_res = np.median(res_lst, 0)
    median_fov = np.median(fov_lst, 0)
    mean_histogram = np.median(np.array(hist_list))
    print('median resolution is ' + str(median_res))

    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    norm_params['xy_resolution'] = median_res[0:2].tolist()
    norm_params['mean_hist'] = mean_histogram.tolist()
    norm_params['z_scale'] = median_res[2].tolist()
    with open(norm_params_path, mode='w') as f:
        json.dump({'mean': norm_params['mean'], 'std': norm_params['std'], 'xy_resolution': norm_params['xy_resolution'], 'z_scale': norm_params['z_scale'],
                   'matadata_path': metadata_path, 'mean_hist': norm_params['mean_hist']}, f,  indent=2)

