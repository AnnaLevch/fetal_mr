import nibabel as nib
import os
from data_curation.helper_functions import resolution_from_scan_name
import numpy as np
from scipy import ndimage
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from utils.read_write_data import save_nifti
"""
This script matches resolution of dataset2 to mean resolution of dataset 1 and saves it in a new directory
"""
if __name__ == "__main__":
    subjects_folder = '/home/bella/Phd/data/brain/HASTE/HASTE/'
    matching_dataset = '/home/bella/Phd/data/brain/FR_FSE/'
    out_folder = '/home/bella/Phd/data/brain/HASTE/HASTE_res_FR-FSE/'

    ref_dirs = os.listdir(matching_dataset)
    ref_res_lst = []
    #get mean resolution of training set
    for ref_dir in ref_dirs:
        xy_resolution = resolution_from_scan_name(ref_dir)
        if(xy_resolution!=None):
            ref_res_lst.append(xy_resolution[0:2])#appending only xy resolution
    mean_ref_res = np.mean(ref_res_lst, axis=0)

    ids = os.listdir(subjects_folder)
    for id in ids:
        dir_path = os.path.join(out_folder, id)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        xy_resolution = resolution_from_scan_name(id)[0:2]
        scale = [i / j for i, j in zip(xy_resolution, mean_ref_res)]
        scale.append(1.0)
        data_path = os.path.join(subjects_folder,id, "volume.nii.gz")
        if not os.path.exists(data_path):
            data_path = os.path.join(subjects_folder,id, "volume.nii")
        volume = nib.load(data_path).get_data()

        truth_path = os.path.join(subjects_folder,id, "truth.nii.gz")
        if not os.path.exists(truth_path):
            truth_path = os.path.join(subjects_folder,id, "truth.nii")
        truth = nib.load(truth_path).get_data()
        volume, swap_axis = move_smallest_axis_to_z(volume)
        truth, swap_axis = move_smallest_axis_to_z(truth)
        volume = ndimage.zoom(volume, scale)
        truth = ndimage.zoom(truth, scale)
        volume = swap_to_original_axis(swap_axis, volume)
        truth = swap_to_original_axis(swap_axis, truth)

        save_nifti(volume, os.path.join(out_folder,id, 'volume'+'.nii.gz'))
        save_nifti(truth, os.path.join(out_folder,id, 'truth'+'.nii.gz'))