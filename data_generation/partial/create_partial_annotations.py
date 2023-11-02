import glob
import os
import nibabel as nib
import shutil
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
import numpy as np
import random
from utils.read_write_data import save_nifti


def get_partial_data_at_position(truth, slice_number, num_slices):
    """
    Create partial annotations with specified number of slices and specified location
    :param truth:
    :param slice_numbaer:
    :param num_slices:
    :return:
    """
    start_index = slice_number - int(np.round(num_slices/2))
    if start_index < 0:
        start_index = 0
    elif start_index + num_slices >= truth.shape[2]:
        start_index = truth.shape[2]-num_slices-1

    partial_truth = np.zeros_like(truth)
    uncertainty = np.ones_like(truth)
    #update uncertainty to 0 for annotated slices
    uncertainty[:,:,start_index:start_index+num_slices] = partial_truth[:,:,start_index:start_index+num_slices]
    partial_truth[:,:,start_index:start_index+num_slices] = truth[:,:,start_index:start_index+num_slices]

    return partial_truth, uncertainty


def generate_partial_data_tta_picking(y_true, case_id, tta_est_picking_df):
    """

    :param swapped_y_true: ground truth
    :param case_id: case id
    :param tta_est_picking: dataframe that includes partial annotations locations picked using TTA estimation
    :return: partially annotated mask and uncertainty mask
    """
    case_series = tta_est_picking_df[tta_est_picking_df['Subject'] == int(case_id)]
    start_index = case_series['max_start'].to_numpy()[0]
    end_index = case_series['max_end'].to_numpy()[0]
    partial_truth = np.zeros_like(y_true)
    uncertainty = np.ones_like(y_true)
    #update uncertainty to 0 for annotated slices
    uncertainty[:,:,start_index:end_index] = partial_truth[:,:,start_index:end_index]
    partial_truth[:,:,start_index:end_index] = y_true[:,:,start_index:end_index]

    return partial_truth, uncertainty


def generate_partial_data_rand_picking(y_true, annotations_ratio, subject_dir):
    """
    Randomly pick location for partial data and generate partial data block according to annotations ratio
    :param y_true: ground truth
    :param annotations_ratio: percentage of slices out of non-zero slices
    :return:
    """
    nonzero_slices = set(np.nonzero(y_true)[2])
    num_slices = int(np.round(len(nonzero_slices) * annotations_ratio))
    print('for case ' + subject_dir + ' picking ' + str(num_slices) + ' slices out of nonzero ' + str(
        len(nonzero_slices)))
    slice_number = random.sample(list(nonzero_slices), 1)[0]
    print('chosen slice is: ' + str(slice_number))

    return get_partial_data_at_position(y_true, slice_number, num_slices)


if __name__ == "__main__":
    data_path = '/home/bella/Phd/data/placenta/placenta_clean/'
    out_path = '/home/bella/Phd/data/placenta/placenta_partial_annotations0.2/'
    truth_filename = 'truth.nii'
    volume_filename = 'volume.nii'
    annotations_ratio = 0.2

    pathes = glob.glob(os.path.join(data_path, '*'))
    for subject_dir in pathes:
        case_id = os.path.basename(subject_dir)
        volume_path = os.path.join(subject_dir,volume_filename)
        if os.path.exists(os.path.join(out_path, case_id)) is False:
            os.mkdir(os.path.join(out_path, case_id))
        if os.path.exists(volume_path):
            shutil.copy(volume_path, os.path.join(out_path, case_id, volume_filename))
        else:
            shutil.copy(volume_path + '.gz', os.path.join(out_path, case_id, volume_filename + '.gz'))

        truth_path = os.path.join(subject_dir, truth_filename)
        if os.path.exists(truth_path):
            y_true = nib.load(truth_path).get_data()
            shutil.copy(truth_path, os.path.join(out_path, case_id, truth_filename))
        else:
            y_true = nib.load(truth_path + '.gz').get_data()
            shutil.copy(truth_path + '.gz', os.path.join(out_path, case_id, truth_filename + '.gz'))
        y_true, swap_axis = move_smallest_axis_to_z(y_true)

        partial_truth, uncertainty = generate_partial_data_rand_picking(y_true, annotations_ratio)
        partial_truth = swap_to_original_axis(swap_axis, partial_truth)
        uncertainty = swap_to_original_axis(swap_axis, uncertainty)
        save_nifti(partial_truth, os.path.join(out_path, case_id, 'partial_truth.nii.gz'))
        save_nifti(uncertainty, os.path.join(out_path, case_id, 'uncertainty.nii.gz'))
