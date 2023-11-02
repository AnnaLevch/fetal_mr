import glob
import os
import shutil


if __name__ == "__main__":
    """"
    Copy relevant series based on series name
    """

    path = '\\\\fmri-df3\\dafna\\Dafi\\Data_ASL_IVIM\\'
    body_folder = '\\\\fmri-df3\\dafna\\Bella\\results\\to_Dafi_body_placenta_brain\\body_cases'
    cases_pathes = glob.glob(os.path.join(path,'*'))
    for case_path in cases_pathes:
        print('case ' + case_path)
        study_folders = glob.glob(os.path.join(case_path, 'Study*'))
        if len(study_folders)>0:
            series_folders = glob.glob(os.path.join(study_folders[0], '*'))
            for series_folder in series_folders:
                series_name = os.path.basename(series_folder).lower()
                if 'cor_vol_body_t2_trufi_flat_breathing' in series_name:
                    relevant_series = glob.glob(os.path.join(series_folder, '*.nii'))
                    if len(relevant_series)==0:
                        continue
                    splitted_name = series_name.split('_')
                    prefix_name = splitted_name[0] + '_' + splitted_name[1]
                    for series in relevant_series:
                        series_name = prefix_name + '_' + os.path.basename(series)
                        shutil.copy(series, os.path.join(body_folder, series_name))
