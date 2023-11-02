import logging
from anomaly_detection.autoencoder.autoencoder_data_trasform import extract_context_rep, pad_to_size, \
    extract_single_slice_rep, extract_context_rep_pairs
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from sacred import Experiment
import os
from anomaly_detection.autoencoder.model import context_autoencoder, single_slice_autoencoder,\
    context_autoencoder_simple, single_slice_autoencoder_simple, single_slice_autoencoder_simple_skip,\
    single_slice_autoencoder_3pool, context_autoencoder_depth4, context_autoencoder_depth3

from data_curation.helper_functions import move_smallest_axis_to_z
import training.train_functions.metrics
import nibabel as nib
import numpy as np
"""
Training with sacred framework. By defult the hardcoded configuration is used. If you want to use existing configuration just add as argument
with [config_path]
"""
#initializing sacred
my_path = os.path.abspath(os.path.dirname(__file__))
sacred_path = os.path.join(my_path, '../runs/')
ex = Experiment(sacred_path)

from sacred.observers import FileStorageObserver


@ex.config
def config():
    my_path = os.path.abspath(os.path.dirname(__file__))

    scans_dir = os.path.join(my_path, '../../../../../data/brain/FR_FSE_cutted') #directory of the raw scans
    slice_size = [512,512]
    batch_size = 16
    model_type = {
        0: 'context_autoencoder',
        1: 'single_slice_autoencoder',
        2: 'context_autoencoder_simple',
        3: 'single_slice_autoencoder_simple',
        4: 'single_slice_autoencoder_simple_skip',
        5: 'single_slice_autoencoder_3pool',
        6: 'context_autoencoder_depth4',
        7: 'context_autoencoder_depth3'
    }[7]

    loss = {
        0: 'binary_crossentropy_loss',
        1: 'dice_coefficient_loss',
        2: 'l1_dice_loss' #use for pairs data only! applying on half of the data l1 loss and half of it dice loss (l1 for volume reconstruction, dice for masks)
    }[1]
    pairs_input = False
    input_shape = [512,512,5]
    epochs = 200


def is_context_autoencoder(model_type):
    if(model_type == 'context_autoencoder' or model_type=='context_autoencoder_simple' or
               model_type=='context_autoencoder_depth4' or 'context_autoencoder_depth3'):
        return True
    else:
        return False


@ex.main
def my_main():

    data = prepare_data()

    train_data(data)


def get_callbacks(ex, learning_rate_drop=0.5, learning_rate_patience=50, verbosity=1,early_stopping_patience=25, save_best_only=True):
    callbacks = list()

    callbacks.append(ModelCheckpoint(filepath=os.path.join(ex.observers[0].dir, 'epoch_{epoch:03d}-loss{loss:.3f}_model.hdf5'),
                                     save_best_only=save_best_only, verbose=verbosity, monitor='loss'))
    callbacks.append(CSVLogger(os.path.join(ex.observers[0].dir, 'metrics.csv')))


    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def get_model(model_type, input_shape, loss):
    if(model_type=='context_autoencoder'):
        autoencoder = context_autoencoder(input_shape=input_shape,loss=loss)
    elif(model_type=='single_slice_autoencoder'):
        autoencoder = single_slice_autoencoder(loss=loss)
    elif(model_type=='context_autoencoder_simple'):#model_type == context_autoencoder_simple
        autoencoder = context_autoencoder_simple(loss=loss)
    elif(model_type=='single_slice_autoencoder_simple'):
        autoencoder = single_slice_autoencoder_simple(loss=loss)
    elif(model_type=='single_slice_autoencoder_simple_skip'):
        autoencoder = single_slice_autoencoder_simple_skip(loss=loss)
    elif(model_type=='single_slice_autoencoder_3pool'):
        autoencoder = single_slice_autoencoder_3pool(loss=loss)
    elif(model_type=='context_autoencoder_depth4'):
        autoencoder = context_autoencoder_depth4(loss=loss),
    elif(model_type=='context_autoencoder_depth3'):
        autoencoder = context_autoencoder_depth3(input_shape= input_shape,loss=loss)

    return autoencoder


@ex.capture
def train_data(train_data, batch_size, model_type, input_shape, loss, epochs):
    loss_func = getattr(training.train_functions.metrics, loss)
    autoencoder = get_model(model_type, input_shape, loss_func)

    autoencoder.fit(train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=get_callbacks(ex=ex))


def load_and_preprocess_data(scans_dir, dir, filename, slice_size):
    data_path = os.path.join(scans_dir, dir, filename)
    if(not os.path.exists(data_path)):
            data_path = os.path.join(scans_dir, dir, filename + '.gz')
    data = nib.load(data_path).get_data()
    data, swap_axis = move_smallest_axis_to_z(data)
    data, _ = pad_to_size(data, slice_size)
    return data


@ex.capture
def prepare_data(scans_dir, slice_size, model_type, pairs_input, input_shape):
    """
    Using both volume data and mask data for representation
    For each one we take 2 consecutive slices, resulting in data with depth 10
    :param scans_dir:
    :param slice_size:
    :param model_type:
    :return:
    """
    dirs = next(os.walk(os.path.join(scans_dir,'.')))[1]
    slices_data = []
    for dir in dirs:
        print('processing case: ' + dir)

        mask = load_and_preprocess_data(scans_dir,dir,'truth.nii',slice_size)
        if(pairs_input == True):
            volume = load_and_preprocess_data(scans_dir,dir,'volume.nii',slice_size)
            scan_slices_data = extract_context_rep_pairs(volume, mask, input_shape)
        else:
            if(is_context_autoencoder(model_type)):
                scan_slices_data = extract_context_rep(mask)
            else:
                scan_slices_data = extract_single_slice_rep(mask)

        slices_data.extend(scan_slices_data)

    if(pairs_input == True):
        train_data = np.reshape(np.concatenate(slices_data, axis=0), (len(slices_data), input_shape[0], input_shape[1], 10))
    else:
        if(is_context_autoencoder(model_type)):
            train_data = np.reshape(np.concatenate(slices_data, axis=0), (len(slices_data), input_shape[0], input_shape[1], 5))
        else:
            train_data = np.reshape(np.concatenate(slices_data, axis=0), (len(slices_data), input_shape[0], input_shape[1], 1))

    return train_data


if __name__ == '__main__':

    log_dir = '../../../log/self_consistency/'
    log_level = logging.INFO
    my_path = os.path.abspath(os.path.dirname(__file__))
    log_path = os.path.join(my_path, log_dir)

    # uid = uuid.uuid4().hex
    # fs_observer = FileStorageObserver.create(os.path.join(log_path, uid))
    fs_observer = FileStorageObserver.create(log_path)

    ex.observers.append(fs_observer)

    # initialize logger
    logger = logging.getLogger()
    hdlr = logging.FileHandler(os.path.join(ex.observers[0].basedir, 'messages.log'))
    FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(filename)20s:%(lineno)-5s %(funcName)-25s %(message)s"
    formatter = logging.Formatter(FORMAT)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    # logger.removeHandler(lhStdout)
    logger.setLevel(log_level)
    ex.logger = logger
    logging.info('Experiment {}, run {} initialized'.format(ex.path, ex.current_run))

    ex.run_commandline()