import glob
import os
from enum import Enum
import re
import pandas as pd
import shutil


class Protocol(Enum):
    FIESTA = 1
    SS_FSE = 2
    HASTE = 3
    OTHER = 4


def get_case_id(folder_name):
    splitted_name = folder_name.split('_')
    return splitted_name[1]


def series_id_protocol_from_name(filename, search_string):
    try:
        p = re.compile("Se(?P<id>[\d]+).*" + search_string + "\.nii")
        find_res = p.findall(filename)
    except:
        print('exception in extracting series id protocol name')
        return None

    if(len(find_res) < 1):
        return None
    print('found ' + search_string + ' case in filename: ' + filename)
    return find_res[0]


def get_case_resolution_fov(series_id, metadata_path, series_column='series_id'):
    try:
        metadata_df = pd.read_excel(metadata_path, engine='openpyxl')
        new_columns = metadata_df.columns.values
        new_columns[0] = 'series_id'
        metadata_df.columns = new_columns
        row_df = metadata_df[(metadata_df[series_column] == int(series_id))]
        xy_res = row_df.iloc[0]['PixelSpacing']
        spacing = row_df.iloc[0]['SpacingBetweenSlices']
        fov = row_df.iloc[0]['FOV']
    except:
        print("no information about resolution in series " + series_id + 'in file ' + metadata_path)
        return None, None, None

    try:
        p = re.compile("\d*\.\d+")
        find_res = p.findall(xy_res)
    except:
        print('exception in extracting xy resolutions from string')
        return None, None, None

    return find_res, spacing, fov


def copy_data_with_new_ids(new_format_dir, FIESTA_pathes, SS_FSE_pathes, HASTE_pathes):
    copy_protocol_data_with_new_ids(new_format_dir, FIESTA_pathes, 'FIESTA')
    copy_protocol_data_with_new_ids(new_format_dir, SS_FSE_pathes, 'SS_FSE')
    copy_protocol_data_with_new_ids(new_format_dir, HASTE_pathes, 'HASTE')


def copy_protocol_data_with_new_ids(new_format_dir, protocol_pathes, protocol_name):
    for new_id in protocol_pathes:
        original_path = protocol_pathes[new_id]['series_path']
        new_path = os.path.join(new_format_dir, protocol_name, new_id + '.nii')
        shutil.copy(original_path, new_path)


def save_matching_files(matching_file_path, FIESTA_pathes, SS_FSE_pathes, HASTE_pathes):
    FIESTA_pd = pd.DataFrame.from_dict(FIESTA_pathes).T
    FIESTA_pd.to_csv(os.path.join(matching_file_path, 'FIESTA_matching.csv'))

    SSFSE_pd = pd.DataFrame.from_dict(SS_FSE_pathes).T
    SSFSE_pd.to_csv(os.path.join(matching_file_path, 'SSFSE_matching.csv'))

    HASTE_pd = pd.DataFrame.from_dict(HASTE_pathes).T
    HASTE_pd.to_csv(os.path.join(matching_file_path, 'HASTE_matching.csv'))


if __name__ == "__main__":
    data_dir = '/media/bella/8A1D-C0A6/Phd/data/elka_cases/original_format/'
    matching_file_path = '/home/bella/Phd/data/data_description/data_Elka/'
    new_format_path = '/media/bella/8A1D-C0A6/Phd/data/elka_cases/new_ids_brain/'

    cases_dirs = glob.glob(os.path.join(data_dir,'*'))
    FIESTA_pathes = {}
    SS_FSE_pathes = {}
    HASTE_pathes = {}

    for case_dir in cases_dirs:
        case_id = int(get_case_id(os.path.basename(case_dir)))
        case_id += 10000
        print('processing case ' + str(case_id))
        series_pathes = glob.glob(os.path.join(data_dir, case_dir, '*'))

        for series_path in series_pathes:
            series_id = series_id_protocol_from_name(os.path.basename(series_path), 'brain')
            if series_id is not None :
                protocol_type = Protocol.FIESTA
            else:
                series_id = series_id_protocol_from_name(os.path.basename(series_path), 'brain_ssfse')
                if series_id is not None :
                    protocol_type = Protocol.SS_FSE
                else:
                   series_id = series_id_protocol_from_name(os.path.basename(series_path), 'brain_haste')
                   if series_id is not None :
                       protocol_type = Protocol.HASTE
                   else:
                       protocol_type = Protocol.OTHER
            if protocol_type is Protocol.OTHER:
                continue


            metadata_path = os.path.join(data_dir, case_dir, 'series_data.xlsx')
            xy_res, spacing, fov = get_case_resolution_fov(series_id, metadata_path)
            new_series_id = 'Pat{case_id}_Se{series_id}_Res{x_res}_{y_res}_Spac{spacing}'.format(case_id=case_id, series_id=series_id,
                                                                                                 x_res=xy_res[0], y_res=xy_res[1], spacing=spacing)
            if protocol_type is Protocol.FIESTA:
                FIESTA_pathes[new_series_id] = {}
                FIESTA_pathes[new_series_id]['series_path'] = series_path
                FIESTA_pathes[new_series_id]['Subject'] = case_id
                FIESTA_pathes[new_series_id]['xy_res'] = '[' + str(xy_res[0]) + ',' + xy_res[1] + ']'
                FIESTA_pathes[new_series_id]['FOV'] = fov
                FIESTA_pathes[new_series_id]['Series'] = series_id
                FIESTA_pathes[new_series_id]['SpacingBetweenSlices'] = spacing
            elif protocol_type is Protocol.SS_FSE:
                SS_FSE_pathes[new_series_id] = {}
                SS_FSE_pathes[new_series_id]['series_path'] = series_path
                SS_FSE_pathes[new_series_id]['Subject'] = case_id
                SS_FSE_pathes[new_series_id]['xy_res'] = '[' + str(xy_res[0]) + ',' + xy_res[1] + ']'
                SS_FSE_pathes[new_series_id]['FOV'] = fov
                SS_FSE_pathes[new_series_id]['Series'] = series_id
                SS_FSE_pathes[new_series_id]['SpacingBetweenSlices'] = spacing
            else: #HASTE
                HASTE_pathes[new_series_id] = {}
                HASTE_pathes[new_series_id]['series_path'] = series_path
                HASTE_pathes[new_series_id]['Subject'] = case_id
                HASTE_pathes[new_series_id]['xy_res'] = '[' + str(xy_res[0]) + ',' + xy_res[1] + ']'
                HASTE_pathes[new_series_id]['FOV'] = fov
                HASTE_pathes[new_series_id]['Series'] = series_id
                HASTE_pathes[new_series_id]['SpacingBetweenSlices'] = spacing


    save_matching_files(matching_file_path, FIESTA_pathes, SS_FSE_pathes, HASTE_pathes)
    copy_data_with_new_ids(new_format_path, FIESTA_pathes, SS_FSE_pathes, HASTE_pathes)