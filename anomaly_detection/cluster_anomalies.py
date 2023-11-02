import pandas as pd
import os
from sklearn.cluster import SpectralClustering, KMeans


if __name__ == "__main__":
    anomalies_path = '/home/bella/Phd/code/code_bella/log/27/output/FIESTA_origin_gt_errors/anomaly_detection'
    filename = 'anomaly_dice_hausdorff.csv'
    anomaly_data = pd.read_csv(os.path.join(anomalies_path, filename ), index_col=0)
  #  anomaly_data = list(anomaly_data.values())

    clustering = KMeans(n_clusters=4).fit(anomaly_data.values)
 #   clustering = SpectralClustering(n_clusters=4,affinity='nearest_neighbors').fit(anomaly_data.values)
    labels_series = clustering.labels_
    anomaly_data['label'] = labels_series
    anomaly_data.to_csv(os.path.join(anomalies_path,'anomaly_dice_hausdorff.csv'))
    print('clustering')