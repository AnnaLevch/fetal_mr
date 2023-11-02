import numpy as np
import nibabel as nib
import os
import glob
from evaluation.surface_distance.lits_surface import *
from data_curation.helper_functions import move_smallest_axis_to_z,swap_to_original_axis
from utils.read_write_data import save_nifti


def extract_2D_contour(mask):
    indices_truth = np.nonzero(mask>0)
    if ((len(indices_truth[0])) == 0 ):
        return None
    try:
        return Surface.compute_contour_2D(mask)
    except Exception as e:
        if(hasattr(e,'message')):
            print(e.message)
        else:
            print(e)
        return None


def extract_volume_2D_contours(mask):
    gt_contour_img_2d = np.zeros(mask.shape, dtype=np.int16)
    for i in range(0,mask.shape[2]):
        indices_truth = np.nonzero(mask[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 ):
            continue
        try:
            slice_contour = Surface.compute_contour_2D(mask[:, :, i].astype(np.int16))
        except:
            continue
        gt_contour_img_2d[:,:,i] = slice_contour

    return gt_contour_img_2d.astype(float)


def extract_and_save_contours(subject_folder, truth_filename):
    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    truth, swap_axis = move_smallest_axis_to_z(y_true)

    gt_contour_img_2d = extract_volume_2D_contours(truth)

    gt_contour_img_2d = swap_to_original_axis(swap_axis, gt_contour_img_2d)

    save_nifti(gt_contour_img_2d, os.path.join(subject_folder, 'contour.nii.gz'))


if __name__ == '__main__':
    src_dir = '/home/bella/Phd/data/brain/FR_FSE_cutted/'
    truth_filename = "truth.nii.gz"
    dirs_path =  glob.glob(os.path.join(src_dir, '*'))
    for dir in dirs_path:
       extract_and_save_contours(dir, truth_filename)
       print('processing dir ' + dir)

