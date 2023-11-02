import pandas as pd
import os
"""
Assuming that we have a csv with case id and slice number of anomalies, this script outputs an excel file
with found slices and their consequitove slices for debugging purposes
"""




if __name__ == "__main__":
    anomalies_dir_path = '/home/bella/Phd/code/code_bella/log/anomaly_datection/'
    filename = 'anomalies_true.csv'
    images_dir = 'anomaly_images'
    data_dir = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin_gt_errors/'

    anomalies_df = pd.read_csv(os.path.join(anomalies_dir_path, filename), header=None)
    anomalies_lst = anomalies_df.iloc[:,0].tolist()
    vol_anomalies = get_vol_anomalies_dict(anomalies_lst)

    images_dict = {}
    for vol_id in vol_anomalies.keys():
        truth_path = os.path.join(data_dir, vol_id, "truth.nii.gz")
        vol_path = os.path.join(data_dir, vol_id, "volume.nii.gz")
        #saving anomaly images to a directory and saving relevant pathes in a dictionary
    #    for slice in vol_anomalies[vol_id]:

