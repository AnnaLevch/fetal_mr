import pandas as pd
import os
from evaluation.eval_utils.key_images import save_anomaly_images
from data_curation.helper_functions import patient_series_name_from_filepath


def get_vol_anomalies_dict(anomalies_lst):
    vol_anomalies_dict = {}

    for anomaly in anomalies_lst:
        splitted = anomaly.rsplit('_',1)
        vol_id = splitted[0]
        slice_ind = splitted[1]
        if(vol_id not in vol_anomalies_dict.keys()):
            vol_anomalies_dict[vol_id]=[]
        vol_anomalies_dict[vol_id].append(slice_ind)

    return vol_anomalies_dict


def write_image_to_excel(images_pathes, image_name, start_row, worksheet):
    worksheet.insert_image('A' + str(start_row), os.path.join(images_pathes,image_name))
    worksheet.write('B' + str(start_row-2), image_name)


def write_slice_and_context_slices(writer, sheet_name, images_path, anomaly_id):
    start_row = 7
    figure_hight = 18
    worksheet = writer.sheets[sheet_name]
    write_image_to_excel(images_path , anomaly_id + '_prev.png', start_row, worksheet)
    start_row = start_row + figure_hight + 1
    write_image_to_excel(images_path , anomaly_id + '.png', start_row, worksheet)
    start_row = start_row + figure_hight + 1
    write_image_to_excel(images_path , anomaly_id + '_next.png', start_row, worksheet)


"""
This script updates an output csv anomaly result with either the anomaly was found in ground truth anomalies
Note that ground truth anomalies might include errors by themselves (either untrue anomaly or unreported anomaly)
"""
if __name__ == "__main__":
    observed_anomalies_path = '/home/bella/Phd/code/code_bella/log/self_consistency/anomalies_true.csv'
    dir_path = '/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/unified_gt_errors_clean/'
    data_dir = dir_path
  #  data_dir = os.path.join(dir_path,'test')
    truth_filename = 'clean_truth.nii'
    result_filename = 'lerrors_truth.nii'
    volume_filename = 'volume.nii'
    predicted_anomalies_dir = os.path.join(dir_path,'anomaly_detection')

  #  predicted_anomalies_path = os.path.join(predicted_anomalies_dir,'hausdorff_per_slice.csv')
    predicted_anomalies_path = observed_anomalies_path

    writer = pd.ExcelWriter(predicted_anomalies_path + '.xlsx', engine='xlsxwriter')

    anomalies_df = pd.read_csv(observed_anomalies_path, header=None)
    anomalies_set = set(anomalies_df.iloc[:,0].tolist())
    predicted_anomalies_df = pd.read_csv(predicted_anomalies_path)
    predicted_anomalies = predicted_anomalies_df.iloc[:,0].tolist()
    #update is anomaly
    is_anomaly_lst = []
    is_anomaly_dict={}
    for anomaly in predicted_anomalies:
        if(anomaly in anomalies_set):
            is_anomaly_lst.append(1)
            is_anomaly_dict[anomaly]=True
        else:
            is_anomaly_lst.append(0)
            is_anomaly_dict[anomaly]=False
    predicted_anomalies_df['is_anomaly'] = is_anomaly_lst

    df = predicted_anomalies_df.round(2)
    df.to_excel(writer,  sheet_name='anomalies_summary')

    #write predicted anomalies images

    images_path = os.path.join(predicted_anomalies_dir,'anomalies_images')
    vol_anomalies_dict = get_vol_anomalies_dict(predicted_anomalies)
    for vol_id in vol_anomalies_dict.keys():
        images_pathes = save_anomaly_images(vol_id, vol_anomalies_dict, data_dir, truth_filename, volume_filename, result_filename, images_path)
        for anomaly_id in images_pathes.keys():
            slice_data={}
            slice_data[anomaly_id]=is_anomaly_dict[anomaly_id]
            df_2D = pd.DataFrame.from_dict(slice_data, orient='index').T
            if(len(anomaly_id)>10):
                sheet_name = patient_series_name_from_filepath(anomaly_id)
            else:
                sheet_name = anomaly_id
            df_2D.to_excel(writer, sheet_name=sheet_name)
            workbook = writer.book

            write_slice_and_context_slices(writer, sheet_name, images_path, anomaly_id)

    writer.save()







 #   predicted_anomalies_df.to_csv(predicted_anomalies_path, index=False)


