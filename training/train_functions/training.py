import glob
import math
import os
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import load_model, Model
import training.train_functions.metrics
from training.train_functions.metrics import *
from keras.optimizers import Adam
from training.train_functions.ReduceLROnPlateauWithRestarts import ReduceLROnPlateauWithRestarts
from training.train_functions.dynamic_param import DynamicParam


class LearningRateLogger(Callback):
    """
    Logging of learning rate for debugging purposes
    """
    def __init__(self):
        super().__init__()
      #  self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


def get_last_model_path(model_file_path):
    model_names = glob.glob(model_file_path + '*.h5')
    if not model_names:
      model_names = glob.glob(model_file_path + '*.hdf5')
    if(len(model_names) == 0):
        print('error loading any model with prefix ' + model_file_path)

    return sorted(model_names, key=os.path.getmtime)[-1]

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


def get_callbacks(ex, steps_per_epoch, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, verbosity=1, early_stopping_patience=None, save_best_only=True,
                  reduce_plateau_with_restarts=False, num_cycle_epochs=60, gamma_dyn=None):

    callbacks = list()

    callbacks.append(ModelCheckpoint(filepath=os.path.join(ex.observers[0].dir, 'epoch_{epoch:03d}-loss{val_loss:.3f}_model.hdf5'),
                                     save_best_only=save_best_only, verbose=verbosity, monitor='val_loss'))
 #   callbacks.append(LearningRateLogger())
    callbacks.append(CSVLogger(os.path.join(ex.observers[0].dir, 'metrics.csv')))
    if gamma_dyn is not None:
        callbacks.append(gamma_dyn)

    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))

    if reduce_plateau_with_restarts:
        # callbacks.append(SGDRScheduler(min_lr=0.000001, max_lr=0.0001, steps_per_epoch=steps_per_epoch,
        #                                lr_decay=learning_rate_drop, cycle_length=40))
        callbacks.append(ReduceLROnPlateauWithRestarts(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity, num_cycle_epochs=num_cycle_epochs,
                                                       train_dir=ex.observers[0].dir))
        return callbacks

    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                            drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))

    return callbacks



def update_training_params(model, loss, initial_learning_rate):

    metrics = ['binary_accuracy', vod_coefficient]
    if loss != dice_coefficient_loss:
        metrics += [dice_coefficient]
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss, metrics=metrics)

    return model


def build_model_manually(model_file, loss, initial_learning_rate, dropout_rate, input_shape, chosen_model, lambda_ac,
                         beta_ac, gamma_ac, theta_ac, band, alpha_q, beta_q, u_th, denoising, drop_xy_levels,
                         mask_shape, gamma_dyn):
    print('Trying to build model manually...')
    loss_func = getattr(training.train_functions.metrics, loss)
    model_func = getattr(training.model, chosen_model)
    model = model_func(input_shape=input_shape,
                       initial_learning_rate=initial_learning_rate,
                       **{'dropout_rate': dropout_rate,
                          'loss_function': loss_func,
                          'lambda_ac' : lambda_ac,
                          'beta' : beta_ac,
                          'gamma' : gamma_ac,
                          'theta' : theta_ac,
                          'band': band,
                          'alpha_q': alpha_q,
                          'beta_q': beta_q,
                          'u_th': u_th,
                          'gamma_dyn': gamma_dyn,
                          'drop_xy_levels':drop_xy_levels,
                          'denoising' : denoising,
                          'mask_shape':mask_shape})
                    #      'old_model_path': model_file})
    model.load_weights(model_file)

    return model


def load_old_model(model_file, loss=None, initial_learning_rate=None, dropout_rate=None, input_shape=[0,0,0],
                   chosen_model=None, build_manually=True,verbose=True, lambda_ac=1, beta_ac=1, gamma_ac=1, theta_ac=0,
                   band=0, alpha_q=0, beta_q=0, u_th=0.5, denoising=False, drop_xy_levels=2, mask_shape=None,
                   gamma_dyn=None, th_binarization=1) -> Model:
    """
    By default, model is built manually to allow different loss functions of the pretrained model and the new model
    """
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'vod_coefficient': vod_coefficient,
                      'vod_coefficient_loss': vod_coefficient_loss,
                      'focal_loss': focal_loss,
                      'focal_loss_fixed': focal_loss,
                      'dice_and_xent': dice_and_xent,
                      'double_dice_loss': double_dice_loss,
                      'dice_distance_weighted_loss': dice_distance_weighted_loss(tf.keras.backend.zeros(1)),
                      'Active_Contour_Loss': active_contour_loss(input_shape[1:], lambda_ac),
                      'active_contour_loss': active_contour_loss(input_shape[1:], lambda_ac),
                      'active_contour_assym_loss': active_contour_assym_loss(input_shape[1:], lambda_ac, beta_ac),
                      'loss': dice_coef_loss,
                      'active_contour_contour_emphesize_loss':active_contour_contour_emphesize_loss(input_shape[1:], lambda_ac,
                                                                                                    beta_ac, gamma_ac, theta_ac),
                      'contour_losses_and_dice_loss': contour_losses_and_dice_loss(input_shape[1:], beta_ac, gamma_ac, lambda_ac, band, th_binarization),
                      'contour_losses_and_dice_loss_dyn_gamma': contour_losses_and_dice_loss_dyn_gamma(input_shape[1:],beta_ac, DynamicParam(gamma_ac), lambda_ac, band, th_binarization),
                      'contour_dice_tolerance_and_dice_loss': contour_dice_tolerance_and_dice_loss(input_shape[1:], beta_ac, gamma_ac, lambda_ac, band),
                      'contour_dice_tolerance_and_dice_uncertainty_loss': contour_dice_tolerance_and_dice_uncertainty_loss(tf.keras.backend.zeros(1),
                                                                                                                           input_shape[1:], beta_ac, gamma_ac, lambda_ac, band, u_th),
                      'uncertainty_contour_dice_tolerance_and_dice_loss': uncertainty_contour_dice_tolerance_and_dice_loss(tf.keras.backend.zeros(1),
                                                                                                                           input_shape[1:], beta_ac, gamma_ac, lambda_ac, band, u_th),
                      'dice_with_uncertainty_loss':dice_with_uncertainty_loss(tf.keras.backend.zeros(1), u_th),
                      'boundary_loss_dice_loss': boundary_loss_dice_loss(beta_ac, DynamicParam(gamma_ac)),
                      'hausdorff_loss_dice_loss': hausdorff_loss_dice_loss(beta_ac, DynamicParam(gamma_ac)),
                      'perimeter_loss_dice_loss': perimeter_loss_dice_loss(input_shape[1:], beta_ac, DynamicParam(gamma_ac)),
                      'dice_cross_entropy_loss':dice_cross_entropy_loss,
                      'dice_cross_entropy_normalized_length_loss':dice_cross_entropy_normalized_length_loss(beta_ac, gamma_ac, t=1.1),
                      'dice_focal_normalized_length_loss':dice_focal_normalized_length_loss(beta_ac, gamma_ac, t=1.1),
                      'length_ratio': length_ratio,
                      'length_loss': length_loss,
                      'tversky_loss': tversky_loss,
                      'quality_est_loss': quality_est_loss(tf.keras.backend.zeros(1), alpha_q, beta_q),
                      'error_emphesize_loss': error_emphesize_loss(tf.keras.backend.zeros(1), alpha_q)
                      }
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass

    if(build_manually):
        model = build_model_manually(model_file, loss, initial_learning_rate, dropout_rate, input_shape, chosen_model, lambda_ac, beta_ac, gamma_ac,
                                     theta_ac, band, alpha_q, beta_q, u_th, denoising, drop_xy_levels, mask_shape=mask_shape, gamma_dyn=gamma_dyn)
        return model

    try:
        if verbose:
            print('Loading model from {}...'.format(model_file))
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        print(error)
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            print(error)



def train_model(model, ex, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500, gamma_dyn=None,
                learning_rate_patience=20, early_stopping_patience=None, save_best_only=True, reduce_plateau_with_restarts=False):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return:
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        max_queue_size=20,
                        workers=1,
                        use_multiprocessing=False,
                        callbacks=get_callbacks(ex=ex,
                                                steps_per_epoch=steps_per_epoch,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience,
                                                reduce_plateau_with_restarts=reduce_plateau_with_restarts,
                                                save_best_only=save_best_only,
                                                gamma_dyn=gamma_dyn))
