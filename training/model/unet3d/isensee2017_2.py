from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, concatenate
from keras.engine import Model
from keras.optimizers import Adam

from training.train_functions.metrics import *
from training.train_functions.dynamic_param import DynamicParam
from .unet import create_convolution_block, concatenate

import numpy as np

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

"""
isensee2017_2 performs pooling only in x,y dimensions as z dimension is usually relatively small
"""
def isensee2017_model_3d(input_shape=(1, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                         n_segmentation_levels=1, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                         loss_function=dice_coefficient_loss, activation_name="sigmoid", mask_shape=None,
                         concat_mask=False, mask_output=False, drop_xy_levels=2, lambda_ac=0, beta=0, gamma=0, theta=0,
                         alpha_q=0, beta_q=0, denoising = False, t=1.1, band=0, u_th=0.5, gamma_dyn=None, th_binarization=1):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)
    if mask_shape is not None and concat_mask:
        mask_input = Input(shape=mask_shape)
        inputs = [inputs, mask_input]

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    #Encoder layers
    for level_number in range(depth):
        n_level_filters = min(128, (2 ** level_number) * n_base_filters)
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            if mask_shape is not None and concat_mask:
                current_layer = concatenate(inputs, axis=-4)
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            if level_number <= drop_xy_levels:
                print(level_number)
                in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 1))
            else:
                in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    #decoder and contractive unet connections
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        if level_number <= (drop_xy_levels - 1):
            print(level_number)
            up_sampling = create_up_sampling_module(current_layer, level_filters[level_number], size=(2,2,1))
        else:
            up_sampling = create_up_sampling_module(current_layer, level_filters[level_number], size=(2, 2, 2))

        if denoising is True and level_number < (drop_xy_levels - 1):
            localization_output = create_localization_module(up_sampling, level_filters[level_number])
        else:
            concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
            localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    #Adding decoder skip connections
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
        print(level_number)

    activation_block = Activation(activation_name)(output_layer)

    metrics = ['binary_accuracy', vod_coefficient]
    if loss_function != dice_coefficient_loss:
        metrics += [dice_coefficient]

    if mask_shape is not None and concat_mask is False:
        mask_input = Input(shape=mask_shape)
        inputs = [inputs, mask_input]

    if(loss_function == dice_distance_weighted_loss): #if we are using distance mask we expect it to be the same
        loss_function = loss_function(mask_input)

    if(loss_function == quality_est_loss): #if we are using distance mask we expect it to be the same
        loss_function = loss_function(mask_input, alpha_q, beta_q)

    if(loss_function == error_emphesize_loss): #if we are using distance mask we expect it to be the same
        loss_function = loss_function(mask_input, alpha_q)

    if(loss_function == contour_losses_and_dice_loss or loss_function == contour_dice_tolerance_and_dice_loss):
        loss_function = loss_function(input_shape[1:], beta, gamma, lambda_ac, band, th_binarization)

    if(loss_function == dice_with_uncertainty_loss):
        loss_function = loss_function(mask_input, u_th)

    if(loss_function == contour_dice_tolerance_and_dice_uncertainty_loss or loss_function == uncertainty_contour_dice_tolerance_and_dice_loss):
        loss_function = loss_function(mask_input, input_shape[1:], beta, gamma, lambda_ac, band, u_th)

    if loss_function == boundary_loss_dice_loss:
        loss_function = loss_function(beta, gamma_dyn)

    if loss_function == hausdorff_loss_dice_loss:
        loss_function = loss_function(beta, gamma_dyn)

    if loss_function == contour_losses_and_dice_loss_dyn_gamma:
        loss_function = loss_function(input_shape[1:], beta, gamma_dyn, lambda_ac, band, th_binarization)

    if loss_function == perimeter_loss_dice_loss:
        loss_function = loss_function(input_shape[1:], beta, gamma_dyn)


    if(loss_function == dice_cross_entropy_normalized_length_loss):
        loss_function = loss_function(beta, gamma, t)
        metrics.append(length_ratio)
        metrics.append(length_loss)

    if(loss_function == dice_focal_normalized_length_loss):
        loss_function = loss_function(beta, gamma)
        metrics.append(length_ratio)
        metrics.append(length_loss)

    if(loss_function == active_contour_loss):
        loss_function = loss_function(input_shape[1:], lambda_ac)

    if(loss_function == active_contour_assym_loss):
        loss_function = loss_function(input_shape[1:], lambda_ac, beta)

    if(loss_function == active_contour_contour_emphesize_loss):
        loss_function = loss_function(input_shape[1:], lambda_ac, beta, gamma, theta)

    model = Model(inputs=inputs, outputs=activation_block, name='isensee2017_3d_Model_'+str(np.random.random()))
    model.summary()
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2