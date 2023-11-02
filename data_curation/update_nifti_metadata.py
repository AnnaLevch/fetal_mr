import glob
import os
import numpy as np
import pandas as pd
from data_curation.helper_functions import patient_series_id_from_filepath, patient_underscore_series_id_from_filepath
from utils.read_write_data import read_nifti_vol_meta, save_nifti_with_metadata, read_nifti, save_nifti


def align_min_axes(vol1, vol2):
    """
    Aligning the shape of vol1 to the shape of volume 2
    Args:
        vol1: volume to switch axes in
        vol2: volume to take the target shape from
    Returns:
        aligned vol1
    """
    shape1 = vol1.shape
    min_index1 = shape1.index(min(shape1))

    shape2 = vol2.shape
    min_index2 = shape2.index(min(shape2))


    vol1 = np.swapaxes(vol1, min_index1, min_index2)

    return vol1


def check_similarity(vol_meta, vol):
    similar_dimensions = True
    similar_volumes = True

    if vol.shape != vol_meta.shape:
        print('shape was not similar, aligning volumes to check voxel similarity.')
        similar_dimensions = False
        vol = align_min_axes(vol, vol_meta)

    num_nonzero = np.nonzero(vol_meta-vol)
    if len(num_nonzero[0]) != 0:
        similar_volumes = False

    return similar_volumes, similar_dimensions

def is_trufi2020_format(case_dir):
    """
    For TRUFI 2020 data we expect 6 underscores in the format, for others 5 or less
    Args:
        case_dir: 
    Returns:
    """
    basename = os.path.basename(case_dir)
    count_underscore = basename.count('_')
    if count_underscore == 5:
        return True
    return False


def read_trufi2020_nifti(nifti_path, subject_id, series_id):
    subject_dir = glob.glob(os.path.join(nifti_path, f'*{subject_id}*'))
    nifti_with_meta_path = glob.glob(os.path.join(subject_dir[0], 'Nii', f'*{series_id}*trufi*'))
    return nifti_with_meta_path[0]

def load_volumes(case_dir, nifti_path, trufi2020_path):
    is_trufi2020 = is_trufi2020_format(case_dir)
    filename = os.path.basename(case_dir)
    if is_trufi2020 is True:
        subject_id, series_id = patient_underscore_series_id_from_filepath(case_dir)
        subject_with_metadata_path = read_trufi2020_nifti(trufi2020_path, subject_id, series_id)
    else:
        subject_id, series_id = patient_series_id_from_filepath(case_dir)
        subject_with_metadata_path = os.path.join(nifti_path, subject_id, filename + '.nii.gz')
    vol_meta, affine, header = read_nifti_vol_meta(subject_with_metadata_path)
    vol_path = os.path.join(case_dir, 'volume.nii.gz')
    if os.path.exists(vol_path) is False:
        vol_path = os.path.join(case_dir, 'data.nii.gz')
    vol, affine_dummy, header_dummy = read_nifti_vol_meta(vol_path)

    return vol, vol_meta, affine, header, vol_path, filename, subject_with_metadata_path


if __name__ == '__main__':
    """
    This script adds metadata to a given folder with cases without metadata
    It first makes sure the volumes are the same. 
    It also outputs csv indicating if the matching was exact for each case
    """
    nifti_without_meta_path = '\\\\10.101.119.14\\Dafna\\Bella\\data\\Body\\TRUFI\\'
    nifti_path = '\\\\fmri-df3\\users\\Fetal\\DataNii\\'
    trufi2020_path = '\\\\fmri-df3\\users\\Fetal\\Data-2020\\'
    log_csv_path = '\\\\10.101.119.14\\Dafna\\Bella\\data\\metadata_update\\trufi_body.csv'

    cases_pathes = glob.glob(os.path.join(nifti_without_meta_path, '*'))
    cases_info = {}
    for case_dir in cases_pathes:
        print('updating volume: ' + case_dir)

        vol, vol_meta, affine, header, vol_path, filename, subject_with_metadata_path = load_volumes(case_dir,
                                                                                                     nifti_path,
                                                                                                     trufi2020_path)
        similar_volumes, similar_dimensions = check_similarity(vol_meta, vol)

        if similar_dimensions is False:
            print('updating truth file')
            truth = read_nifti(os.path.join(case_dir, 'truth.nii.gz'))
            truth = align_min_axes(truth, vol_meta)
            save_nifti(truth, os.path.join(case_dir, 'truth.nii.gz'))
        save_nifti_with_metadata(vol_meta, affine, header, vol_path)

        cases_info[filename] = {}
        cases_info[filename]['nifti_no_metadata_path'] = os.path.join(case_dir, 'data.nii.gz')
        cases_info[filename]['nifti_with_metadata_path'] = subject_with_metadata_path
        cases_info[filename]['similar_dimensions'] = similar_dimensions
        cases_info[filename]['similar_volumes'] = similar_volumes


    df = pd.DataFrame(cases_info)
    df.to_csv(log_csv_path)