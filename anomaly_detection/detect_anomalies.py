import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import math
from anomaly_detection.evaluate_anomalies import get_vol_anomalies_dict, write_slice_and_context_slices
from evaluation.eval_utils.key_images import save_anomaly_images
from data_curation.helper_functions import patient_series_name_from_filepath


def get_object_slices(points):
    """
    Calculate first and last non-nan slices
    :param points:
    :return:
    """
    for i in range(0, len(points)):
        if(math.isnan(points[i])==False):
            break
    first_slice = i
    for i in reversed(range(0,len(points))):
        if(math.isnan(points[i])==False):
            break
    last_slice = i
    return first_slice,last_slice


def get_features_vector_cons_diff(points, i, window_size):
    features = []

    if(math.isnan(points[i])):
        features.append(-1)
        for j in range(1,window_size+1):
            features.append(-1)
            features.append(-1)
        return features

    features.append(points[i])

    for j in range(1,window_size+1):
        if(math.isnan(points[i-j]) or math.isnan(points[i-j+1])):
          features.append(-1)
        else:
            features.append(np.abs(points[i-j+1]-points[i-j]))
        if(math.isnan(points[i+j]) or math.isnan(points[i+j-1])):
            features.append(-1)
        else:
            features.append(np.abs(points[i+j-1]-points[i+j]))

    return features


def get_features_vector(points, i, window_size):
    features = []

    if(math.isnan(points[i])):
        features.append(-1)
        for j in range(1,window_size+1):
            features.append(-1)
            features.append(-1)
        return features

    features.append(points[i])

    for j in range(1,window_size+1):
        if(math.isnan(points[i-j])):
          features.append(-1)
        else:
            features.append(np.abs(points[i]-points[i-j]))
        if(math.isnan(points[i+j])):
            features.append(-1)
        else:
            features.append(np.abs(points[i]-points[i+j]))

    return features


def get_features_vector_square_diff(points, i, window_size):
    features = []

    if(math.isnan(points[i])):
        features.append(-1)
        for j in range(1,window_size+1):
            features.append(-1)
            features.append(-1)
        return features

    features.append(points[i])

    for j in range(1,window_size+1):
        if(math.isnan(points[i-j])):
          features.append(-1)
        else:
            features.append(np.square(points[i]-points[i-j]))
        if(math.isnan(points[i+j])):
            features.append(-1)
        else:
            features.append(np.square(points[i]-points[i+j]))

    return features


def metric_features(metric_per_slice, window_size, border_distance):
    pts = {}
    for key in metric_per_slice:
        points = metric_per_slice[key]
        start_slice, end_slice = get_object_slices(points)
        for i in range(start_slice+border_distance, end_slice-border_distance):
            # if(math.isnan(points[i]) or math.isnan(points[i-1]) or math.isnan(points[i+1]) or math.isnan(points[i-2]) or math.isnan(points[i+2]) or  math.isnan(points[i-3]) or math.isnan(points[i+3]) or math.isnan(points[i-4]) or math.isnan(points[i+4])):
            #     continue

            pts[key + '_' + str(i)] = get_features_vector(points, i,window_size=window_size)

    return pts


def unify_features(eval_features_list, num_metrics, eval_features_size):
    num_features = num_metrics*eval_features_size
    features = {}
    eval_ind = 0
    for eval_features in eval_features_list.values():
        for ind in eval_features:
            if(ind not in features):
                features[ind] = np.empty(num_features)
                features[ind][:] = -1
            start_ind = eval_ind*eval_features_size
            features[ind][start_ind:start_ind+eval_features_size]=(eval_features[ind])
        eval_ind+=1

    return features


def calc_metrics_features(path, metrics, window_size, border_distance):
    """
    This function loads metrics data and calculates feature vector from them
    Features from different metrics are unified to a single vector
    :param path:
    :param metrics:
    :return: dictionary with metrics features
    """
    metrics_features = {}
    for metric in metrics:
        metric_data = pd.read_csv(os.path.join(path, 'test',metric + '.csv'), index_col=0).to_dict()
        metrics_features[metric] = metric_features(metric_data, window_size, border_distance)

    features = unify_features(metrics_features, len(metrics),1+window_size*2)

    return features


def write_anomalies_to_excel(anomaly_df, predicted_anomalies_path, data_dir, anomalies_filename, truth_filename="truth.nii.gz", volume_filename = 'data.nii.gz'):

    writer = pd.ExcelWriter(os.path.join(predicted_anomalies_path, anomalies_filename + '.xlsx'), engine='xlsxwriter')
    df = anomaly_df.round(2)
    df.to_excel(writer,  sheet_name='anomalies_summary')
    images_path = os.path.join(predicted_anomalies_path,'anomalies_images')

    predicted_anomalies = anomaly_df.index.values.tolist()
    vol_anomalies_dict = get_vol_anomalies_dict(predicted_anomalies)

    for vol_id in vol_anomalies_dict.keys():
        images_pathes = save_anomaly_images(vol_id, vol_anomalies_dict, data_dir, truth_filename,volume_filename , None, images_path)
        for anomaly_id in images_pathes.keys():
            if(len(anomaly_id)>10):
                sheet_name = patient_series_name_from_filepath(anomaly_id)
            else:
                sheet_name = anomaly_id

            slice_data={}
            slice_data[anomaly_id]='anomaly'
            df_2D = pd.DataFrame.from_dict(slice_data, orient='index').T
            df_2D.to_excel(writer, sheet_name=sheet_name)

            write_slice_and_context_slices(writer, sheet_name, images_path, anomaly_id)

    writer.save()




if __name__ == "__main__":
    path = '/home/bella/Phd/code/code_bella/log/self_consistency/24/output/FR-FSE/'
#    metrics = ['dice_zero_per_slice', 'hausdorff_per_slice', 'assd_per_slice','num_componnents', 'componnents_size_std', 'componnents_size_mean', 'componnents_hausdorff']
#    metrics = ['num_componnents', 'componnents_size_mean','componnents_size_std','componnents_hausdorff']
 #   metrics = ['pixel_difference']
    truth_filename = 'before_reconstruction.nii.gz'
    volume_filename = 'volume.nii.gz'

    metrics = ['pixel_difference']
    anomalies_filename = 'pixel_difference'
    window_size = 2
    border_distance = 2
    write_to_excel = True #add tabs with anomaly slices

    predicted_anomalies_path = os.path.join(path, 'anomaly_detection')

    features = calc_metrics_features(path, metrics, window_size,border_distance)

    df_data = pd.DataFrame.from_dict(features).T

    model = IsolationForest(n_estimators = 1000,contamination=0.005, bootstrap=True, random_state=43435)
    model.fit(df_data)

    predictions = model.predict(df_data)

    anomaly_indices = np.where(predictions==-1)
    anomaly_df = df_data.iloc[anomaly_indices]
    anomaly_df.to_csv(os.path.join(predicted_anomalies_path, anomalies_filename + '.csv'))

    if write_to_excel is True:
        data_dir = os.path.join(path, 'test')
        write_anomalies_to_excel(anomaly_df, predicted_anomalies_path, data_dir, anomalies_filename, truth_filename, volume_filename)
