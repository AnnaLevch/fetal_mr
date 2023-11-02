import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z,swap_to_original_axis
from evaluation.features_extraction.feature_extractor import FeatureExtractor
from scipy.spatial.distance import mahalanobis, euclidean
from active_learning.active_learning import ActiveLearning
import numpy as np
from scipy import ndimage


class ActiveLearningUniqueness(ActiveLearning):
    def __init__(self, model_path):
        ActiveLearning.__init__(model_path)


    def calc_distance_from_training(self, data, training_examples, n_training = 2):
        """
        Calculate mean distance between the scan and training examples with munimum distance
        :param sample:
        :param training_examples:
        :param n_training: number of training examples with minimum distance to use for distance calculation
        :return:
        """
        distances = []
        for training_data in training_examples:
            euc_dist = euclidean(data, training_data)
            distances.append(euc_dist)
        euc_dis_smallest_dist = sorted(distances)[0:n_training]

        # training_inv_conv = np.linalg.inv(np.cov(np.transpose(training_examples)))
        # training_mean = np.average(np.array(training_examples), 0)
        # mahalanobis_dist = mahalanobis(data, training_mean, training_inv_conv)

        return np.average(euc_dis_smallest_dist)


    def load_extract_features(self, data_pathes, layer_name, scale, patch_size):
        """
        Load and extract features from a given layer
        :param data_pathes: path to data
        :param layer_name: name of the layer for feature extraction
        :return:
        """
        data_features = {}
        feature_extractor = FeatureExtractor()
        for data_path in data_pathes:
            print('training data path: ' + data_path)
            if os.path.exists(os.path.join(data_path, 'volume.nii')) is True:
                data = nib.load(os.path.join(data_path, 'volume.nii')).get_data()
            else:
                data = nib.load(os.path.join(data_path, 'volume.nii.gz')).get_data()
            data, swap_axis = move_smallest_axis_to_z(data)
            if scale is not None:
                data = ndimage.zoom(data, scale)
            features = feature_extractor.get_layer_features(self._model, layer_name, data, patch_size)
            data_features[os.path.basename(data_path)] = features.flatten()

        return data_features


    def calc_samples_uniqueness(self, training_data_pathes, data_pathes, config=None, patch_size= [128,128,48], layer_name='concatenate_1'):
        """
        Calculate distance score from the training examples for each one of the examples
        :param training_examples:
        :param current_examples:
        :return:
        """
        #Load and extract feature space from all training examples
        scale = config.get('scale', None)
        training_data_features = self.load_extract_features(training_data_pathes, layer_name, scale, patch_size).values()
        unsupervised_data_features = self.load_extract_features(data_pathes, layer_name, scale, patch_size)

        samples_uniqueness = {}
        for path in data_pathes:
            data_features = unsupervised_data_features[os.path.basename(path)]
            distance_from_training = ActiveLearningUniqueness.calc_distance_from_training(self, data_features.flatten(), training_data_features)
            print('distance from training is: ' + str(distance_from_training))
            samples_uniqueness[os.path.basename(path)] = distance_from_training

        return samples_uniqueness
