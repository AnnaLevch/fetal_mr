import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z
from evaluation.eval_utils.feature_extraction import connected_components_data_features
import numpy as np
from evaluation.evaluate import write_eval_per_slice
from evaluation.eval_utils.eval_functions import min_surface_distance
from utils.visualization import visualize_slice
from data_generation.extract_contours import extract_2D_contour


def initialize_features_dict(features):
    measure_per_slice_dict = {}

    for feature in features:
        measure_per_slice_dict[feature] = {}

    return measure_per_slice_dict


def initialize_volume_info(measure_per_slice_dict, features, vol_id):
    for feature in features:
        measure_per_slice_dict[feature][vol_id]={}
    return measure_per_slice_dict


def calculate_connected_component_max_min(components_cnt, values_cnt):
    if(len(values_cnt) == 2):#if there is only one connected component
        return 0
    max_min = 0
    for label in values_cnt[1:]:
        label_slice = np.zeros_like(components_cnt)
        label_indices = np.where(components_cnt==label)
        label_slice[label_indices] = 1

        other_labels = np.copy(components_cnt)
        other_labels[label_indices] = 0
        other_labels[np.where(other_labels>0)]=1

        min_distance = min_surface_distance(label_slice,other_labels,(1,1))

        if(min_distance>max_min):
            max_min=min_distance

    return max_min


def update_connected_componenet_data(measure_per_slice_dict, y_true, i):
    components, values, counts = connected_components_data_features(y_true[:,:,i])
    measure_per_slice_dict['num_componnents'][dir][i+1] =len(values)-1
    if(len(counts)==2): #if there is only one componenet, no meaning to std
        measure_per_slice_dict['componnents_size_std'][dir][i+1] = 0
    else:
        measure_per_slice_dict['componnents_size_std'][dir][i+1] = np.std(counts[1:])
    measure_per_slice_dict['componnents_size_mean'][dir][i+1] = np.mean(counts[1:])
    measure_per_slice_dict['componnents_hausdorff'][dir][i+1] = calculate_connected_component_max_min(components, values)

    return measure_per_slice_dict


def update_zero_slice_data(measure_per_slice_dict, i):
    measure_per_slice_dict['num_componnents'][dir][i+1] = 0
    measure_per_slice_dict['componnents_size_std'][dir][i+1] = None
    measure_per_slice_dict['componnents_size_mean'][dir][i+1] = None
    measure_per_slice_dict['componnents_hausdorff'][dir][i+1] = None
    return measure_per_slice_dict


def write_measures_per_slice(measure_per_slice_dict, features):
    for feature in features:
        write_eval_per_slice(measure_per_slice_dict[feature], os.path.join(src_dir,feature + '.csv'))



if __name__ == '__main__':
    src_dir = '/home/bella/Phd/code/code_bella/log/brain21_24/23/output/FR-FSE_anomalies/test/'
    features = ['num_componnents', 'componnents_size_std', 'componnents_size_mean', 'componnents_hausdorff']
    dirs = next(os.walk(os.path.join(src_dir,'.')))[1]

    measure_per_slice_dict = initialize_features_dict(features)

    for dir in dirs:
        print('processing case: ' + dir)
        truth_path = os.path.join(src_dir, dir, "truth.nii.gz")
   #     result_path = os.path.join(src_dir, dir, "prediction.nii.gz")

        y_true = nib.load(truth_path).get_data()
        y_true, swap_axis = move_smallest_axis_to_z(y_true)

        measure_per_slice_dict = initialize_volume_info(measure_per_slice_dict, features, dir)

        for i in range(0,y_true.shape[2]):
            indices_truth = np.nonzero(y_true[:,:,i]>0)
            if ((len(indices_truth[0])) == 0 ):
                measure_per_slice_dict = update_zero_slice_data(measure_per_slice_dict, i)
                continue

            measure_per_slice_dict = update_connected_componenet_data(measure_per_slice_dict, y_true, i)

    write_measures_per_slice(measure_per_slice_dict, features)

