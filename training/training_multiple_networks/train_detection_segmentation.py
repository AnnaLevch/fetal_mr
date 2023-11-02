import os
import json
import shutil as sh
import subprocess
from training.train_functions.training import get_last_model_path
"""
Training script for fetal brain segmentation, including fine tuning with more aggressive augmentations
"""

def copy_update_old_model_cfg(my_path, configs_folder, out_folder, cfgname, old_model_dir):
    detection2_cfg_path = os.path.join(my_path, configs_folder, cfgname)
    with open(detection2_cfg_path) as json_file:
        data = json.load(json_file)
    base_model_dir = os.path.join(my_path, out_folder,old_model_dir,'epoch_')
    model_path = get_last_model_path(base_model_dir)
    data['old_model_path'] = model_path
    out_config_path = os.path.join(my_path, out_folder, cfgname)
    with open(out_config_path,'w') as outfile :
            json.dump(data,outfile, indent=2)

    return out_config_path


if __name__ == "__main__":
    configs_folder = '../../config/config_brain/config_HASTE'
    detection1_cfgname = 'config_all.json'
    detection2_cfgname = 'config_all_2.json'
    segmentation1_cfgname = 'config_roi.json'
    segmentation2_cfgname = 'config_roi_2.json'
    out_folder = '../../../log/brain_HASTE/'
    my_path = os.path.abspath(os.path.dirname(__file__))

    #training detection1 network
    det1_cfg_path = os.path.join(my_path, configs_folder, detection1_cfgname)
    dest_det1_cfg_path = os.path.join(my_path, out_folder, detection1_cfgname)
    sh.copyfile(det1_cfg_path, dest_det1_cfg_path) #no changes are needed to training from scratch config
    args = "with " + dest_det1_cfg_path
    subprocess.call("python3 -m training.train_scripts.train_brain_dir2 " + args, shell=True)

    #training detection2 network
    #the base network needs to be replaced with the newly trained network
    out_config_path = copy_update_old_model_cfg(my_path, configs_folder, out_folder, detection2_cfgname, '1')
    args = "with " + out_config_path
    subprocess.call("python3 -m training.train_scripts.train_brain_dir2 " + args, shell=True)

    #training segmentation1 network
    seg1_cfg_path = os.path.join(my_path, configs_folder, segmentation1_cfgname)
    dest_seg1_cfg_path = os.path.join(my_path, out_folder, segmentation1_cfgname)
    sh.copyfile(seg1_cfg_path, dest_seg1_cfg_path) #no changes are needed to training from scratch config
    args = "with " + dest_seg1_cfg_path
    subprocess.call("python3 -m training.train_scripts.train_brain_dir2 " + args, shell=True)

    #training segmentation2 network
    out_config_path = copy_update_old_model_cfg(my_path, configs_folder, out_folder, segmentation2_cfgname, '3')
    args = "with " + out_config_path
    subprocess.call("python3 -m training.train_scripts.train_brain_dir2 " + args, shell=True)