import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from anomaly_detection.autoencoder.autoencoder_data_trasform import pad_to_size, unpad_to_original_size
from anomaly_detection.anomaly_utils.utils import get_object_slices
import numpy as np
from training.train_functions.training import load_old_model, get_last_model_path
from utils.read_write_data import save_nifti, pickle_dump
import shutil
from keras import Model, Input


def autoencoder_predict(scans_path, dir, truth_filename, out_path, volume_filename, slice_size, layer_name, context_window, vol_truth_pairs):
    """
    This function saves reconstruction of autoencoder and intermediate layer representation
    :param scans_path:
    :param dir:
    :param truth_filename:
    :param out_path:
    :param volume_filename:
    :param slice_size:
    :return:
    """
    data_dir = os.path.join(scans_path, dir)
    print('processing directory: ' + data_dir)
    truth_path = os.path.join(data_dir, truth_filename)
    if(not os.path.exists(truth_path)):
        truth_filename = truth_filename + '.gz'
        truth_path = os.path.join(data_dir, truth_filename)
    y_true = nib.load(truth_path).get_data()
    y_true, swap_axis = move_smallest_axis_to_z(y_true)

    volume_path = os.path.join(data_dir, volume_filename)
    if(not os.path.exists(volume_path)):
        volume_filename = volume_filename + '.gz'
        volume_path = os.path.join(data_dir, volume_filename)
    if vol_truth_pairs is True:
        volume = nib.load(volume_path).get_data()
        volume, swap_axis = move_smallest_axis_to_z(volume)

    origin_size = y_true.shape
    [first_slice, last_slice] = get_object_slices(y_true)
    y_true, delta = pad_to_size(y_true,slice_size)
    if vol_truth_pairs is True:
        volume, delta = pad_to_size(volume,slice_size)

#    encoder_res = np.zeros_like(y_true)
    encoder_res = y_true
    nn_representations = {}
    for i in range(first_slice+context_window,last_slice-context_window):
        if(context_window!=0):
            truth_context = np.copy(y_true[:,:,i-context_window:i+context_window+1])
            if vol_truth_pairs is True:
                volume_context = np.copy(volume[:,:,i-context_window:i+context_window+1])
                slice_representation = np.concatenate((volume_context, truth_context),axis=2)
            else:
                slice_representation = truth_context
        else:
            slice_representation = np.copy(y_true[:,:,i])
            slice_representation = np.expand_dims(slice_representation,-1)

        slice_representation = np.expand_dims(slice_representation,0)
        decoded_img =_model.predict(slice_representation)
        if(context_window!=0):
            if vol_truth_pairs is True:
                encoder_res[:,:,i] = decoded_img[0,:,:,3*context_window + 1] > 0.5
            else: #only mask was reconstructed
                encoder_res[:,:,i] = decoded_img[0,:,:,context_window] > 0.5
        else:
            encoder_res[:,:,i] = decoded_img[0,:,:,0] > 0.5

    #    inter_layer_model = Model(inputs=_model.input, outputs=_model.get_layer(layer_name).output)
    #    nn_representation = inter_layer_model.predict(slice_representation)
    #    nn_representations[i] = np.float16(np.squeeze(nn_representation))

    encoder_res = unpad_to_original_size(encoder_res,delta, origin_size)
    encoder_res = swap_to_original_axis(swap_axis, np.int16(encoder_res))


    out_dir = os.path.join(out_path, dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #save data and autoencoder result
    save_nifti(encoder_res, os.path.join(out_dir, 'prediction.nii.gz'))
#    pickle_dump(nn_representations, os.path.join(out_dir, 'representation.pickle'))
    shutil.copyfile(os.path.join(data_dir, truth_filename), os.path.join(out_dir, "before_reconstruction.nii.gz"))
    shutil.copyfile(os.path.join(data_dir, volume_filename), os.path.join(out_dir, volume_filename))


if __name__ == "__main__":
    scans_path = '/home/bella/Phd/data/brain/FR_FSE_cutted/'
    model_path = '/home/bella/Phd/code/code_bella/log/self_consistency/24/'
    out_path = os.path.join(model_path, 'output','FR-FSE')
    truth_filename = 'truth.nii'
    volume_filename = 'volume.nii'
    slice_size = [512,512]
    layer_name = 'max_pooling2d_5'
    context_window = 2
    vol_truth_pairs = False
    last_model_path = get_last_model_path(model_path)
    print('First:' + last_model_path)
    _model = load_old_model(last_model_path, build_manually=False)
    print(_model.summary())
    dirs_path = next(os.walk(scans_path))[1]
    for dir in dirs_path:
        autoencoder_predict(scans_path, dir, truth_filename, out_path, volume_filename, slice_size, layer_name, context_window, vol_truth_pairs)
