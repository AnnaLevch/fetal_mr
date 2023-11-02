import numpy as np
from evaluation.surface_distance.metrics import *
import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z
from utils.read_write_data import save_nifti
from evaluation.surface_distance.lits_surface import *

scaling = (1.56,1.56,3)


def get_contour_from_pts(contour_pts):
    max_x = np.max(contour_pts[:,0])
    max_y = np.max(contour_pts[:,1])
    max_z = np.max(contour_pts[:,2])
    contour_vol = np.zeros((max_x+1,max_y+1,max_z+1), dtype=float)
    contour_vol[np.transpose(contour_pts)]=1
    return contour_vol

def get_nonzero_slices_range(vol):
    num_slices = vol.shape[2]
    non_zero_indices = []
    for i in range(0, num_slices):
        slice_labels = vol[:,:,i]
        indices = np.nonzero(slice_labels > 0)
        if (len(indices[0]) == 0):
            continue
        non_zero_indices.append(i)
    min_slice = np.min(non_zero_indices)
    max_slice = np.max(non_zero_indices)

    return min_slice, max_slice


if __name__ == "__main__":

    subject_folder = '/home/bella/Phd/code/code_bella/log/anomaly_datection/17/output/FIESTA_origin_gt_errors/test/117/'
    truth_filename = 'truth.nii.gz'
    result_filename = 'prediction.nii.gz'
    volume_filename = 'data.nii.gz'
    y_true = nib.load(os.path.join(subject_folder, truth_filename)).get_data()
    y_pred = nib.load(os.path.join(subject_folder, result_filename)).get_data()
    volume = nib.load(os.path.join(subject_folder, volume_filename)).get_data()
    truth, swap_axis = move_smallest_axis_to_z(y_true)
    pred, swap_axis = move_smallest_axis_to_z(y_pred)
    volume, _ = move_smallest_axis_to_z(volume)

    #surface calcultation current code
    borders_gt, borders_pred, _ , _ , _ = get_contour_masks(truth,pred, scaling)
    borders_gt = 1.0*borders_gt
    borders_pred = 1.0*borders_pred
    save_nifti(borders_pred, os.path.join(subject_folder, 'borders_predictions'+'.nii.gz'))
    save_nifti(borders_gt, os.path.join(subject_folder, 'borders_gt'+'.nii.gz'))

    #surface calculation lits code
    evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling)
    # pred_surface_pts = np.round(evalsurf.get_mask_edge_poVoints()).astype(int)
    # gt_surface_pts = np.round(evalsurf.get_reference_edge_points()).astype(int)
    pred_contour_img = evalsurf.compute_contour(y_pred)
    gt_contour_img = evalsurf.compute_contour(y_true)
    pred_min_slice, pred_max_slice = get_nonzero_slices_range(pred_contour_img)
    gt_min_slice, gt_max_slice = get_nonzero_slices_range(gt_contour_img)
    pred_contour_img = pred_contour_img[:,:,pred_min_slice:pred_max_slice]
    gt_contour_img_cropped = gt_contour_img[:,:,gt_min_slice:gt_max_slice]
    vol_cropped = volume[:,:,gt_min_slice:gt_max_slice]

    #test 2D lits contour extraction
    num_slices = truth.shape[2]

    gt_contour_img_2d = np.zeros(truth.shape, dtype=float)
    pred_contour_img_2d = np.zeros(truth.shape, dtype=float)
    for i in range(0,num_slices):
       # surf_2d = Surface(y_pred,truth,physical_voxel_spacing = scaling[0:1])
        #evaluate only slices that have at least one truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 ):
            continue
        gt_contour_img_2d[:,:,i] = Surface.compute_contour_2D(truth[:, :, i])

    for i in range(0,num_slices):
        indices_truth = np.nonzero(pred[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 ):
            continue
        pred_contour_img_2d[:,:,i] = Surface.compute_contour_2D(pred[:, :, i])


    save_nifti(pred_contour_img, os.path.join(subject_folder, 'lits_predictions'+'.nii.gz'))
    save_nifti(gt_contour_img, os.path.join(subject_folder, 'lits_gt_3D'+'.nii.gz'))
    save_nifti(gt_contour_img, os.path.join(subject_folder, 'lits_gt_3D_cropped'+'.nii.gz'))
    save_nifti(gt_contour_img_2d, os.path.join(subject_folder, 'lits_gt_2d_contour'+'.nii.gz'))
    save_nifti(pred_contour_img_2d, os.path.join(subject_folder, 'lits_pred_2d_contour'+'.nii.gz'))
    save_nifti(vol_cropped, os.path.join(subject_folder, 'cropped_vol'+'.nii.gz'))


