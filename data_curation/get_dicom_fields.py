import pandas as pd
import pydicom
import glob
import os
import subprocess
import nibabel as nib
from data_curation.helper_functions import patient_series_id_from_filepath, patient_underscore_series_id_from_filepath


def read_dicom_file(series_path, dicom_tags):
    dicomfiles = glob.glob(os.path.join(series_path, "*.dcm"))
    if len(dicomfiles) == 0:
        print('no dicom files found!')
        return None, 0
    dicomfile = dicomfiles[0]
    dicomobj = pydicom.read_file(dicomfile)
    tags_data = {}
    for tag in dicom_tags:
        tags_data[tag] = dicomobj.get(tag)

    #update converted resolution by using dicom2niix software
    series_dir_path = os.path.dirname(series_path)
    series_basename = os.path.basename(series_dir_path)
    series_basename = series_basename.replace(' ', '_')
    # nifti_filepathes = glob.glob(os.path.join(nifti_conversion_path, series_basename + "*.nii"))
    # if len(nifti_filepathes) == 0 or os.path.exists(nifti_filepathes[0]) is False:
    #     print('converting series: ' + series_basename)
    #     subprocess.call("dcm2niix -o {out_path} \"".format(out_path=nifti_conversion_path) + series_dir_path + "\"", shell=True)
    #     nifti_filepathes = glob.glob(os.path.join(nifti_conversion_path, series_basename + "*.nii"))
    #data = nib.load(nifti_filepathes[0])
    # resolution = data.header.get_zooms()
    # tags_data['converted_PixelSpacing'] = list(resolution[:2])
    # tags_data['converted_SpacingBetweenSlices'] = resolution[2]
    tags_data['nifti_filename'] = series_basename
    tags_data['dicom_path'] = series_path
    return tags_data, len(dicomfiles)


def get_metadata(study_path, dicom_tags, series_substring, large_series_criteria=True, substring_criteria=""):
    """
    Get dicom tags from dicom data
    """
    study_series = glob.glob(os.path.join(study_path, "*" + series_substring + "*"+"/"))
    num_series = len(study_series)
    if num_series > 1: #multiple series found
        if large_series_criteria is True:
            print('for study ' + study_path + ' number of matched series: ' + str(num_series))
            print('using series with number of slices above 50')
            for i in range(num_series):
                tags_data, num_slices = read_dicom_file(study_series[i], dicom_tags)
                if num_slices > 50:
                    break
        else:
            print('using additional substring from matching')
            study_series = glob.glob(os.path.join(study_path, "*" + substring_criteria + "*"+"/"))
            print('updated number of series: ' + str(len(study_series)))
            tags_data, num_slices = read_dicom_file(study_series[0], dicom_tags)
    elif num_series == 0:
        print('no series found for path' + study_path)
        return None,0
    else:
        tags_data, num_slices = read_dicom_file(study_series[0], dicom_tags)
    return tags_data, num_series


def update_scanners_data(data_df, ichilov_data_path, old_fiesta_data_path, trufi2020_path, cheo1_path, cheo2_path,
                         ichilov_metadata_df, trufi2020_metadata_df, dicom_tags):
    FIESTA_data = data_df.loc[data_df['DataType']=='FIESTA']
    FIESTA2_data = data_df.loc[data_df['DataType'] =='FIESTA2']
    TRUFI_data = data_df.loc[data_df['DataType']=='TRUFI']
    TRUFI2020_data = data_df.loc[data_df['DataType'] =='TRUFI2020']
    CHEO1_data = data_df.loc[data_df['DataType'] =='Elka1']
    CHEO2_data = data_df.loc[(data_df['DataType'] =='CHEO2') | (data_df['DataType'] == 'Elka2_haste')]


    #FIESTA old data - The matching may not be accurate as we don't have series id (using keywords FIESTA and Vol)
    for i in FIESTA_data.index:
        subject_id = FIESTA_data['Subject'][i]
        print('updating subject ' + subject_id)
        study_path_parent = glob.glob(os.path.join(old_fiesta_data_path, subject_id + "*"))
        study_path = glob.glob(os.path.join(study_path_parent[0],  "Study*"))[0]
        series_substring = 'FIESTA*Vol'
        dicom_metadata, num_series = get_metadata(study_path, dicom_tags, series_substring)
        for tag in dicom_tags:
            data_df.loc[data_df['Subject']==subject_id, tag] = str(dicom_metadata[tag])
        data_df.loc[data_df['Subject']==subject_id, 'num_series'] = num_series
        print('for case: ' + subject_id + 'metadata is: ' + str(dicom_metadata))

    for i in FIESTA2_data.index:
        dir_name = FIESTA2_data['Subject'][i]
        print('updating subject ' + dir_name)
        if dir_name is 'Pat148_Se17_Res1.5625_1.5625_Spac4.0':
            print('stop')
        subject_id, series_id = patient_series_id_from_filepath(dir_name)
        substring = "*" + series_id + "*FIESTA"
        update_tags(data_df, dicom_tags, ichilov_data_path, ichilov_metadata_df, subject_id, series_id, substring)

    for i in TRUFI_data.index:
        dir_name = TRUFI_data['Subject'][i]
        print('updating subject ' + dir_name)
        subject_id, series_id = patient_series_id_from_filepath(dir_name)
        substring = "*" + series_id + "*trufi"
        update_tags(data_df, dicom_tags, ichilov_data_path, ichilov_metadata_df, subject_id, series_id, substring)

    for i in TRUFI2020_data.index:
        new_id = TRUFI2020_data['Subject'][i]
        dir_postfix, series_id = patient_underscore_series_id_from_filepath(new_id)
        print('updating subject ' + new_id)
        study_path= glob.glob(os.path.join(trufi2020_path, '*'+dir_postfix + '*'))
        series_substring = "Se" + series_id + "*trufi_flat_breathing"
        dicom_metadata, num_series = get_metadata(study_path[0], dicom_tags, series_substring)
        for tag in dicom_tags:
            data_df.loc[data_df['Subject']==new_id, tag] = str(dicom_metadata[tag])
        data_df.loc[data_df['Subject']==new_id, 'num_series'] = num_series
        print('for case: ' + new_id + 'metadata is: ' + str(dicom_metadata))

    for i in CHEO1_data.index:
        dir_name = CHEO1_data['Subject'][i]
        print('updating subject ' + dir_name)
        subject_id, series_id = patient_series_id_from_filepath(dir_name)
        subject_id = int(subject_id) - 10000
        study_path = glob.glob(os.path.join(cheo1_path, f"*_{subject_id}*".format(subject_id=subject_id)))
        substring = "Se" + series_id + "_*FIESTA"
        duplicate_substring = "Se" + series_id + "_BODY*FIESTA"
        dicom_metadata, num_series = get_metadata(study_path[0], dicom_tags, substring, False, duplicate_substring)
        if dicom_metadata is None:
            print('checking for HASTE series')
            substring = "Se" + series_id + "_*HASTE"
            dicom_metadata, num_series = get_metadata(study_path[0], dicom_tags, substring, False, duplicate_substring)
        for tag in dicom_tags:
            data_df.loc[data_df['Subject'] == dir_name, tag] = str(dicom_metadata[tag])
        data_df.loc[data_df['Subject'] == dir_name, 'num_series'] = num_series

    for i in CHEO2_data.index:
        dir_name = CHEO2_data['Subject'][i]
        print('updating subject ' + dir_name)
        subject_id, series_id = patient_series_id_from_filepath(dir_name)
        subject_id = int(subject_id) - 10000
        study_path = glob.glob(os.path.join(cheo2_path, str(subject_id)))
        substring = "Se" + series_id + "_*FIESTA"
        duplicate_substring = "Se" + series_id + "_BODY*FIESTA"
        dicom_metadata, num_series = get_metadata(study_path[0], dicom_tags, substring,
                                                  False, duplicate_substring)
        if dicom_metadata is None:
            print('checking for HASTE series')
            substring = "Se" + series_id + "_*HASTE"
            dicom_metadata, num_series = get_metadata(study_path[0], dicom_tags, substring,
                                                      False, duplicate_substring)
        for tag in dicom_tags:
            data_df.loc[data_df['Subject']==dir_name, tag] = str(dicom_metadata[tag])
        data_df.loc[data_df['Subject'] == dir_name, 'num_series'] = num_series

    return data_df


def update_tags(data_df, dicom_tags, data_path, metadata_df, subject_id, series_id=None, series_substring=""):
    """
    Update dicom tags in df
    """
    if series_id is not None:
        subject_metadata = metadata_df.loc[(metadata_df['Subject'] == int(subject_id)) & (metadata_df['Series'] == int(series_id))]
        if len(subject_metadata)==0:
            subject_metadata = metadata_df.loc[
                (metadata_df['Subject'] == str(subject_id)) & (metadata_df['Series'] == int(series_id))]
    else:
        subject_metadata = metadata_df.loc[metadata_df['Subject'] == int(subject_id)]
        if len(subject_metadata) == 0:
            metadata_df.loc[metadata_df['Subject'] == str(subject_id)]

    dicom_pathes = subject_metadata['path']
    if dicom_pathes.size > 0:
        dicom_path = subject_metadata['path'].iloc[0]
    else:
        print('no dicom data for subject ' + subject_id + '!')
        return None
    dicom_metadata, num_series = get_metadata(os.path.join(data_path, os.path.basename(dicom_path)), dicom_tags,
                                              series_substring)
    subject_index = data_df['Subject'].str.contains('Pat' + subject_id + '_') | data_df['Subject'].str.contains(
        'Fetus' + subject_id + '_')
    for tag in dicom_tags:
        data_df.loc[subject_index,tag] = str(dicom_metadata[tag])
    data_df.loc[subject_index, 'num_series'] = num_series
    print('for case: ' + subject_id + 'metadata is: ' + str(dicom_metadata))


if __name__ == "__main__":
    """
    Read dicom files from different sources
    """
    """
       Read dicom files from different sources
       """
    # abnormal_cases_file = "C://Bella//data//description//Body//abnormal_cases.csv"
    # normal_iugr_cases_file = "C://Bella//data//description//Body//normal_and_IUGR_cases.csv"
    normal_iugr_cases_file = "\\\\fmri-df3\\dafna\\Bella\\clinical_article_body\\cases_information\\growth_chart_cases.csv"
    ichilov_metadata_path = 'C://Bella//data//description//Index.csv'
    trufi2020_metadata_path = 'C://Bella//data//description//Body//TRUFI//PostProc.csv'
    ichilov_data_path = '\\\\fmri-df3\\users\\Fetal\\Data-All-Aggregated\\'
    old_fiesta_data_path = '\\\\fmri-df3\\users\\Fetal\\TASMC_GE\\'
    trufi2020_path = '\\\\fmri-df3\\users\\Fetal\\Data-2020\\'
    cheo1_path = '\\\\10.101.119.13\\users\\Fetal\\Fetal-Elka\\Elka\\bunch\\'
    cheo2_path = '\\\\10.101.119.14\\Dafna\\CHEO-ALL\\'
    output_path = '\\\\10.101.119.1\Dafna\\Bella\\docs\\clinical_article\\'
    dicom_tags = ['ManufacturerModelName', 'Manufacturer', 'MagneticFieldStrength', 'EchoTime', 'RepetitionTime',
                  'PixelSpacing', 'SpacingBetweenSlices', 'SeriesNumber', 'PatientID']

    #  abnormal_df = pd.read_csv(abnormal_cases_file)
    normal_iugr_df = pd.read_csv(normal_iugr_cases_file)

    for dicom_tag in dicom_tags:
        #      abnormal_df[dicom_tag] = ""
        normal_iugr_df[dicom_tag] = ""
    #   abnormal_df['num_series'] = ""
    normal_iugr_df['num_series'] = ""

    ichilov_metadata_df = pd.read_csv(ichilov_metadata_path, encoding='latin')
    trufi2020_metadata_df = pd.read_csv(trufi2020_metadata_path, encoding='latin')

    # print('*******abnormal data***********')
    # abnormal_df = update_scanners_data(abnormal_df, ichilov_data_path, old_fiesta_data_path, trufi2020_path, cheo1_path,
    #                                    cheo2_path, ichilov_metadata_df, trufi2020_metadata_df, dicom_tags)
    print('*******normal and iugr data***********')
    normal_iugr_df = update_scanners_data(normal_iugr_df, ichilov_data_path, old_fiesta_data_path, trufi2020_path,
                                          cheo1_path, cheo2_path, ichilov_metadata_df, trufi2020_metadata_df,
                                          dicom_tags)
    # abnormal_df.to_csv(os.path.join(output_path, 'abnormal_scanners_info.csv'))
    normal_iugr_df.to_csv(os.path.join(output_path, 'normal_scanners_info.csv'))

