import keras
from keras.layers import Input
import numpy as np


class FeatureExtractor:

    def pad_to_fit(self, data, patch_size):
        """
        Pad data in case patch size is larger than data
        :param data:
        :param patch_size:
        :return:
        """
        if data.shape[0] >= patch_size[0] and data.shape[1] >= patch_size[1] and data.shape[2] >= patch_size[2]:
            return data

        new_shape = list(data.shape)
        new_shape[0] = max(data.shape[0], patch_size[0])
        new_shape[1] = max(data.shape[1], patch_size[1])
        new_shape[2] = max(data.shape[2], patch_size[2])

        padded_data = np.zeros(new_shape)

        if data.shape[0] < patch_size[0]:
            x_start = int((patch_size[0]-data.shape[0])/2)
            x_end = x_start + data.shape[0]
        else:
            x_start = 0
            x_end = data.shape[0]
        if data.shape[1] < patch_size[1]:
            y_start = int((patch_size[1]-data.shape[1])/2)
            y_end = y_start + data.shape[1]
        else:
            y_start = 0
            y_end = data.shape[1]
        if data.shape[2] < patch_size[2]:
            z_start = int((patch_size[2]-data.shape[2])/2)
            z_end = z_start + data.shape[2]
        else:
            z_start = 0
            z_end = data.shape[2]

        padded_data[x_start:x_end, y_start:y_end, z_start:z_end] = data

        return padded_data


    def extract_center_patch(self, data, patch_size):
        """
        Extract patch in the center fo the data
        :param data:
        :param patch_size:
        :return:
        """

        data = self.pad_to_fit(data, patch_size)

        data_center_index = [data.shape[0]/2, data.shape[1]/2, data.shape[2]/2]
        x_start = data_center_index[0] - (patch_size[0]/2)
        x_end = x_start + patch_size[0]
        y_start = data_center_index[1] - (patch_size[1]/2)
        y_end = y_start + patch_size[1]
        z_start = data_center_index[2] - (patch_size[2]/2)
        z_end = z_start + patch_size[2]
        patch_data = data[int(x_start):int(x_end), int(y_start):int(y_end), int(z_start):int(z_end)]

        return patch_data

    def get_layer_features(self, model, layer_name, data, patch_size):
        """
        Extracts features from a given layer for a patch centered on the data
        :param model: model
        :param layer_name: the name of the features layer
        :return: feature vector of the data from the given layer
        """
        ##TODO: potentially use additional patches beside center
        patch_data = self.extract_center_patch(data, patch_size)
        patch_data = np.expand_dims(patch_data.squeeze(), 0)
        patch_data = np.expand_dims(patch_data, 1)
        intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(patch_data)
