import glob
import nibabel as nib
from data_curation.helper_functions import move_smallest_axis_to_z
from shutil import copyfile
import re
import os
from data_curation.organize_and_assign_new_ids import get_case_resolution_fov, series_id_protocol_from_name
import pandas as pd


def does_contain(path, words_lst):
    path_lower = path.lower()
    for word in words_lst:
        if word not in path_lower:
            return False
    return True

def case_id_from_name(case_folder):
    splitted = case_folder.split('_')
    case_id = splitted[-2] + '_' + splitted[-1]
    return case_id


def series_id_from_name(filename):
    splitted = filename.split('_')
    return splitted[0]


def series_id_se(filename):
    try:
        basename = os.path.basename(filename)
        p = re.compile("Se(?P<series_id>[\d]+)_")
        series_id = p.findall(basename)[0]
        return series_id
    except:
        print("error in parsing resolution for file: " + filename)
        return None

    return [float(x_res), float(y_res), float(z_res)]
"""
This script extracts relevant cases based on given keywords and expected slices range
Suitable for data from Children's hospital Canada (Elka)
"""
if __name__ == "__main__":
    data_folder = '\\\\10.101.119.14\\Dafna\\CHEO-ALL\\'
    out_folder = '\\\\fmri-df3\dafna\Bella\ElkaCases2021\\FIESTA_Body\\'
    matching_file_path = 'C:\Bella\data\Elka_data2'
    out_filename = 'body_data.csv'
 #   keywords = [['fiesta', 'mom'], ['fiesta', 'mother'], ['fiesta', 'body'], ['fiesta', 'baby']]
    keywords = [['fiesta', 'mom'], ['fiesta', 'mother'], ['fiesta', 'body'],['2d_fiesta.nii'],
                ['_cor_mom.nii'],['_sag_mom.nii']]
    nifti_folder = ''
    min_slices=15
    metadata_2020 = {}

    directories = glob.glob(os.path.join(data_folder, '*'))
    trufi_body_pathes = []
    for dir in directories:
        nii_dir = os.path.join(dir, nifti_folder)
        series_pathes = glob.glob(os.path.join(nii_dir, '*.nii'))
        for path in series_pathes:
            for words in keywords:
                if does_contain(path,words) is True:
                    trufi_body_pathes.append(path)
                    break

    #copy found volumes that have more than 60 slices
    for path in trufi_body_pathes:
        volume = nib.load(path).get_data()
        volume, swap_axis = move_smallest_axis_to_z(volume)
        series_filename = os.path.basename(path)
        out_path = os.path.join(out_folder, series_filename)
        if volume.shape[2]>=min_slices:
            case_folder = os.path.dirname(path)
            case_id = str(int(os.path.basename(case_folder))+10000)
            series_id = series_id_se(series_filename)
            metadata_path = os.path.join(case_folder, 'series_data.xlsx')
            if series_id is None:
                print('series Id was not extracted correctly')
                continue
            if os.path.exists(metadata_path) is False:
                print('No metadata file for case ' + case_folder + ', skipping case!')
                continue
            print('copying case ' + os.path.join(case_folder, series_filename))
            xy_res, spacing, fov = get_case_resolution_fov(series_id, metadata_path, series_column='Sequence')
            if (xy_res == None) or (spacing == None) or (fov==None):
                print('resolution information was not extracted correctly, skipping')
                continue
            new_series_id = 'Pat{case_id}_Se{series_id}_Res{x_res}_{y_res}_Spac{spacing}'.format(case_id=case_id,
                                                                                                 series_id=series_id,
                                                                                                 x_res=xy_res[0],
                                                                                                 y_res=xy_res[1],
                                                                                              spacing=spacing)
            series_path = os.path.join(out_folder, new_series_id + '.nii')
            copyfile(path, series_path)
            metadata_2020[new_series_id]={}
            metadata_2020[new_series_id]['series_path'] = series_path
            metadata_2020[new_series_id]['Subject'] = case_id
            metadata_2020[new_series_id]['xy_res'] = '[' + str(xy_res[0]) + ',' + xy_res[1] + ']'
            metadata_2020[new_series_id]['FOV'] = fov
            metadata_2020[new_series_id]['Series'] = series_id
            metadata_2020[new_series_id]['SpacingBetweenSlices'] = spacing
            metadata_2020[new_series_id]['nSlices'] = volume.shape[2]

    FIESTA_pd = pd.DataFrame.from_dict(metadata_2020).T
    FIESTA_pd.to_csv(os.path.join(matching_file_path, out_filename))