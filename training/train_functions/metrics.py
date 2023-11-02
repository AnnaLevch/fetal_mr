from functools import partial
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def extract_volume_2D_contours_tf(mask, input_shape, th=1):
    kernel = tf.zeros((3,3,input_shape[2]), dtype=tf.int32)
    if th == 0.5:
        int_mask = tf.cast(tf.round(mask), dtype=tf.int32)
    elif th == 1:
        int_mask = tf.cast(mask, dtype=tf.int32)
    eroded_mask = tf.nn.erosion2d(int_mask, kernel=kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    contour = tf.cast(tf.bitwise.bitwise_xor(int_mask,eroded_mask), dtype=tf.float32)
    return contour


def extract_volume_2D_bands_tf(mask, input_shape, band_size=3, th=1):
    kernel_erosion = tf.zeros((band_size+2,band_size+2,input_shape[2]), dtype=tf.int32)
    kernel_dilation = tf.zeros((band_size,band_size,input_shape[2]), dtype=tf.int32)
    if th == 0.5:
        int_mask = tf.cast(tf.round(mask), dtype=tf.int32)
    elif th == 1:
        int_mask = tf.cast(mask, dtype=tf.int32)
    dilated_mask = tf.nn.dilation2d(int_mask, filter=kernel_dilation, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    eroded_mask = tf.nn.erosion2d(int_mask, kernel=kernel_erosion, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    contour_dilated = tf.bitwise.bitwise_xor(dilated_mask, int_mask)
    contour_eroded = tf.bitwise.bitwise_xor(eroded_mask,int_mask)
    unified_contour = tf.bitwise.bitwise_or(contour_dilated, contour_eroded)
    band = tf.cast(unified_contour, dtype=tf.float32)
    return band


def extract_volume_2D_contours_tf_test(mask, input_shape):
    kernel = tf.zeros((3,3,input_shape[2]), dtype=tf.int32)
    int_mask = tf.cast(tf.round(mask), dtype=tf.int32)
 #   int_mask = tf.cast(mask, dtype=tf.int32)
    eroded_mask = tf.nn.erosion2d(int_mask, kernel=kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    contour = tf.cast(tf.bitwise.bitwise_xor(int_mask,eroded_mask), dtype=tf.float32)
    return contour, eroded_mask


def extract_volume_2D_bands_tf_test(mask, input_shape, band_size=3):
    kernel_erosion = tf.zeros((band_size+2,band_size+2,input_shape[2]), dtype=tf.int32)
    kernel_dilation = tf.zeros((band_size,band_size,input_shape[2]), dtype=tf.int32)
    int_mask = tf.cast(tf.round(mask), dtype=tf.int32)
 #   int_mask = tf.cast(mask, dtype=tf.int32)
    dilated_mask = tf.nn.dilation2d(int_mask, filter=kernel_dilation, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    eroded_mask = tf.nn.erosion2d(int_mask, kernel=kernel_erosion, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    band_binary = tf.bitwise.bitwise_xor(dilated_mask, eroded_mask)
    band = tf.cast(band_binary, dtype=tf.float32)
    return band, dilated_mask, eroded_mask


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def contour_vol_dice(true_contours, pred_contours, y_pred, smooth=1.):
    """
    This function extracts and normalizes the opposite of fnpl
    :param true_contours:
    :param pred_contours:
    :param y_pred:
    :return:
    """
    y_true_contours_f = K.flatten(true_contours)
    y_pred_f = K.flatten(y_pred)
    y_pred_contours_f = K.flatten(pred_contours)

    contour_vol_intersection = K.sum(y_true_contours_f * y_pred_f)
    return (2. * contour_vol_intersection + smooth) / (K.sum(y_true_contours_f) + K.sum(y_pred_contours_f) + smooth)


def l1_loss_func(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.abs(y_true_f - y_pred_f))


def l1_dice_loss_func(y_true, y_pred, z_size):
    y_true_l1 = y_true[:,:,:,0:z_size]
    y_pred_l1 = y_pred[:,:,:,0:z_size]
    y_true_dice = y_true[:,:,:,z_size:]
    y_pred_dice = y_pred[:,:,:,z_size:]

    return l1_loss_func(y_true_l1, y_pred_l1) - (dice_coefficient(y_true_dice, y_pred_dice)*100)
 #   return l1_loss_func(y_true_l1, y_pred_l1) # + binary_crossentropy(y_true_dice, y_pred_dice)


def contour_length_term(data):
    x = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
    y = data[:, :, :, 1:, :] - data[:, :, :, :-1, :]
    z = data[:, :, :, :, 1:] - data[:, :, :, :, :-1]

    delta_x = x[:,:,1:,:-2,:-2]**2
    delta_y = y[:,:,:-2,1:,:-2]**2
    delta_z = z[:,:,:-2,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001

    return K.sum(K.sqrt(delta_u + epsilon))

def dice_cross_entropy_normalized_length(y_true, y_pred, beta, gamma, t_max):
    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), t_max)

    return dice_coefficient_loss(y_true, y_pred) + (beta*binary_crossentropy(y_true, y_pred)) + (gamma*length_term)


def dice_focal_normalized_length(y_true, y_pred, beta, gamma, t_max):
    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), t_max)

    return dice_coefficient_loss(y_true, y_pred) + (beta*focal_loss(y_true, y_pred)) + (gamma*length_term)


def dice_cross_entropy_normalized_length_min_max(y_true, y_pred, beta, gamma, t_max, t_min):
    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), t_max)
    length_term = K.maximum(length_term, t_min)
    return dice_coefficient_loss(y_true, y_pred) + (beta*binary_crossentropy(y_true, y_pred)) + (gamma*length_term)


def contour_dice(y_true, y_pred, input_shape, smooth, th):
    true_contours = extract_volume_2D_contours_tf(y_true[:,0,:,:,:], input_shape, th)
    pred_contours = extract_volume_2D_contours_tf(y_pred[:,0,:,:,:], input_shape, th)

    return dice_coefficient(true_contours, pred_contours, smooth)


def contour_dice_uncertainty(y_true, y_pred, input_shape, smooth, uncertainty_map, u_th):
    true_contours = extract_volume_2D_contours_tf(y_true[:,0,:,:,:], input_shape)
    pred_contours = extract_volume_2D_contours_tf(y_pred[:,0,:,:,:], input_shape)

    indices = tf.where(uncertainty_map[:,0,:,:,:] <= u_th)
    true_contours = tf.gather_nd(true_contours, indices)
    pred_contours = tf.gather_nd(pred_contours, indices)

    return dice_coefficient(true_contours, pred_contours, smooth)


def contour_dice_with_tolerance(y_true, y_pred, input_shape, band_size, smooth):
    true_contours = extract_volume_2D_contours_tf(y_true[:,0,:,:,:], input_shape)
    pred_contours = extract_volume_2D_contours_tf(y_pred[:,0,:,:,:], input_shape)
    true_band = extract_volume_2D_bands_tf(y_true[:,0,:,:,:], input_shape, band_size)
    pred_band = extract_volume_2D_bands_tf(y_pred[:,0,:,:,:], input_shape, band_size)

    true_contour_f = K.flatten(true_contours)
    pred_contour_f = K.flatten(pred_contours)
    true_band_f = K.flatten(true_band)
    pred_band_f = K.flatten(pred_band)
    intersections = K.sum(true_contour_f * pred_band_f) + K.sum(true_band_f * pred_contour_f)
    return (intersections + smooth) / (K.sum(true_contour_f) + K.sum(pred_contour_f) + smooth)


def contour_dice_with_band(y_true, y_pred, input_shape, band_size, smooth, th=1):
    if band_size == 0:
        return contour_dice(y_true, y_pred, input_shape, smooth, th)

    true_band = extract_volume_2D_bands_tf(y_true[:,0,:,:,:], input_shape, band_size, th)
    pred_band = extract_volume_2D_bands_tf(y_pred[:,0,:,:,:], input_shape, band_size, th)

    true_band_f = K.flatten(true_band)
    pred_band_f = K.flatten(pred_band)
    intersection = K.sum(true_band_f * pred_band_f)
    return (2*intersection + smooth) / (K.sum(true_band_f) + K.sum(pred_band_f) + smooth)


def contour_dice_with_band_uncertainty(y_true, y_pred, input_shape, band_size, smooth, u_th, uncertainty_map):


    if band_size == 0:
        return contour_dice_uncertainty(y_true, y_pred, input_shape, smooth, uncertainty_map, u_th)

    true_band = extract_volume_2D_bands_tf(y_true[:,0,:,:,:], input_shape, band_size)
    pred_band = extract_volume_2D_bands_tf(y_pred[:,0,:,:,:], input_shape, band_size)

    indices = tf.where(uncertainty_map[:,0,:,:,:] <= u_th)
    true_band = tf.gather_nd(true_band, indices)
    pred_band = tf.gather_nd(pred_band, indices)

    return dice_coefficient(true_band, pred_band, smooth)


def contour_losses_and_dice_loss_func(y_true, y_pred, input_shape, beta, gamma, band=0, smooth=1., labmda=0, th=1):

    dice = dice_coefficient(y_true, y_pred, smooth)

    contour_dice_loss = contour_dice_with_band(y_true, y_pred, input_shape, band, smooth, th)

    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), 1.1)

    return -dice + (beta*binary_crossentropy(y_true, y_pred)) - (gamma*contour_dice_loss) + (labmda*length_term)


def dice_with_uncertainty(y_true, y_pred, uncertainty_map, smooth, u_th = 0.5):
    """
    Apply dice score for indices with confidence above 0.5
    :param y_true: truth
    :param y_pred:prediction
    :param uncertainty_map:uncertainty map
    :param smooth:smoothing parameter
    :return:
    """
    indices = tf.where(uncertainty_map <= u_th)
    y_true = tf.gather_nd(y_true, indices)
    y_pred = tf.gather_nd(y_pred, indices)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def uncetainty_contour_losses_and_dice_loss_func(y_true, y_pred, uncertainty_map, input_shape, beta, gamma, band=0,
                                                 smooth=1., labmda=0, u_th=0.5):

    dice_uncertainty = dice_with_uncertainty(y_true, y_pred, uncertainty_map, smooth, u_th)

    contour_dice_loss = contour_dice_with_band_uncertainty(y_true, y_pred, input_shape, band, smooth, u_th,
                                                           uncertainty_map)

    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), 1.1)

    return -dice_uncertainty + (beta*binary_crossentropy(y_true, y_pred)) - (gamma*contour_dice_loss) + (labmda*length_term)


def contour_losses_and_dice_uncetainty_loss_func(y_true, y_pred, uncertainty_map, input_shape, beta, gamma, band=0,
                                                 smooth=1., labmda=0, u_th=0.5):

    dice_uncertainty = dice_with_uncertainty(y_true, y_pred, uncertainty_map, smooth, u_th)

    contour_dice_loss = contour_dice_with_band_uncertainty(y_true, y_pred, input_shape, band, smooth, u_th,
                                                            uncertainty_map)
   # contour_dice_loss = contour_dice_with_band(y_true, y_pred, input_shape, band, smooth)

    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), 1.1)

    return -dice_uncertainty + (beta*binary_crossentropy(y_true, y_pred)) - (gamma*contour_dice_loss) + (labmda*length_term)


def contour_dice_tolerance_and_dice_loss_func(y_true, y_pred, input_shape, beta, gamma, band=3, smooth=1., labmda=0):

    dice = dice_coefficient(y_true, y_pred, smooth)

    if band == 0:
        contour_dice_loss = contour_dice(y_true, y_pred, input_shape, smooth)
    else:
        contour_dice_loss = contour_dice_with_tolerance(y_true, y_pred, input_shape, band, smooth)

    length_term = K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), 1.1)

    return -dice + (beta*binary_crossentropy(y_true, y_pred)) - (gamma*contour_dice_loss) + (labmda*length_term)


def hausdorff_loss_dice_loss_func(y_true, y_pred, beta, gamma_dyn):
    dice = dice_coefficient(y_true, y_pred)

    return -dice + (beta*binary_crossentropy(y_true, y_pred)) + (gamma_dyn.get_param()*hausdorff_loss_keras(y_true, y_pred))


def boundary_loss_dice_loss_func(y_true, y_pred, beta, gamma_dyn):
    dice = dice_coefficient(y_true, y_pred)

    return -dice + (beta*binary_crossentropy(y_true, y_pred)) + (gamma_dyn.get_param()*boundary_loss_keras(y_true, y_pred))


def perimeter_loss_dice_loss_func(y_true, y_pred, input_shape, beta, gamma_dyn):
    """
    Combination of perimeter loss with regional losses
    :param y_true:
    :param y_pred:
    :param input_shape:
    :param beta:
    :param gamma:
    :return:
    """
    return -dice_coefficient(y_true, y_pred) + (beta*binary_crossentropy(y_true, y_pred)) \
           +(gamma_dyn.get_param()*perimeter_loss(y_true, y_pred, input_shape))


def length_ratio(y_true, y_pred):
    return contour_length_term(y_pred)/contour_length_term(y_true)


def length_loss(y_true, y_pred):
    return K.minimum(contour_length_term(y_pred)/contour_length_term(y_true), 1.1)


def active_contour_contour_emphesize(y_true, y_pred, input_shape, lambdaP, beta, gamma, theta = 0):

    """
    :param y_true: truth volume
    :param y_pred: prediction volume
    :param input_shape: s
    :param lambdaP:
    :param beta:
    :param true_contour: Contour of truth volume (to avoid recalculation)
    :return:
    """

    x = y_pred[:,:,1:,:,:] - y_pred[:,:,:-1,:,:]
    y = y_pred[:,:,:,1:,:] - y_pred[:,:,:,:-1,:]
    z = y_pred[:,:,:,:,1:] - y_pred[:,:,:,:,:-1]

    delta_x = x[:,:,1:,:-2,:-2]**2
    delta_y = y[:,:,:-2,1:,:-2]**2
    delta_z = z[:,:,:-2,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones(input_shape)
    C_2 = np.zeros(input_shape)

    #calculate added path lenght
    true_contours_tf = extract_volume_2D_contours_tf(y_true[:,0,:,:,:], input_shape)
    pred_contours_tf = extract_volume_2D_contours_tf(y_pred[:,0,:,:,:], input_shape)

    region_in = K.abs(K.sum( y_pred[:,0, :,:,:] * ((y_true[:,0,:,:,:] - C_1)**2) ) ) # equ.(12) in the paper, region in of prediction
    region_out = K.abs(K.sum( (1-y_pred[:,0,:,:,:]) * ((y_true[:,0,:,:,:] - C_2)**2) )) # equ.(12) in the paper, region out of prediction

    contour_out = K.abs(K.sum( (1-y_pred[:,0,:,:,:]) * ((true_contours_tf - C_2)**2) )) # equ.(12) in the paper, region out of prediction

    apl = K.abs(K.sum( (1-pred_contours_tf) * ((true_contours_tf - C_2)**2) ))

    loss = lenth + (lambdaP * (region_in + beta*region_out)) + (gamma * contour_out) + (theta*apl)

    return loss


def active_contour_assym_loss_func(y_true, y_pred, input_shape, lambdaP, beta):

    """
    lenth term
    """

    x = y_pred[:,:,1:,:,:] - y_pred[:,:,:-1,:,:]
    y = y_pred[:,:,:,1:,:] - y_pred[:,:,:,:-1,:]
    z = y_pred[:,:,:,:,1:] - y_pred[:,:,:,:,:-1]

    delta_x = x[:,:,1:,:-2,:-2]**2
    delta_y = y[:,:,:-2,1:,:-2]**2
    delta_z = z[:,:,:-2,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones(input_shape)
    C_2 = np.zeros(input_shape)

    region_in = K.abs(K.sum( y_pred[:,0, :,:,:] * ((y_true[:,0,:,:,:] - C_1)**2) ) ) # equ.(12) in the paper
    region_out = K.abs(K.sum( (1-y_pred[:,0,:,:,:]) * ((y_true[:,0,:,:,:] - C_2)**2) )) # equ.(12) in the paper

    loss =  lenth + lambdaP * (region_in + beta*region_out)

    return loss


def active_contour_loss_func(y_true, y_pred, input_shape, lambdaP):

    """
    lenth term
    """

    x = y_pred[:,:,1:,:,:] - y_pred[:,:,:-1,:,:]
    y = y_pred[:,:,:,1:,:] - y_pred[:,:,:,:-1,:]
    z = y_pred[:,:,:,:,1:] - y_pred[:,:,:,:,:-1]

    delta_x = x[:,:,1:,:-2,:-2]**2
    delta_y = y[:,:,:-2,1:,:-2]**2
    delta_z = z[:,:,:-2,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y + delta_z)

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    """
    region_in = K.abs(K.sum( y_pred[:,0, :,:,:] * ((y_true[:,0,:,:,:] - C_1)**2) ) ) # equ.(12) in the paper
    region_out = K.abs(K.sum( (1-y_pred[:,0,:,:,:]) * ((y_true[:,0,:,:,:] - C_2)**2) )) # equ.(12) in the paper

    region term
    """

    C_1 = np.ones(input_shape)
    C_2 = np.zeros(input_shape)

    loss =  lenth + lambdaP * (region_in + region_out)

    return loss


def dice_distance_weighted(y_true, y_pred, distance_map, smooth=1., beta=0.2):
    """
    Weight dice by distance map in order to give more weight to voxels in the contour
    :param y_true:
    :param y_pred:
    :param distance_map: Distance map from graound truth contour. Minimum value is 1
    :param smooth:
    :return:
    """
    #compute distance weighted dice
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    distance_map = K.flatten(distance_map)
    y_true_f = y_true_f/(K.log(distance_map) + smooth)
    y_pred_f = y_pred_f/(K.log(distance_map) + smooth)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_distance_weighted = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    dice = dice_coefficient(y_true, y_pred)

    return (beta * dice_distance_weighted) + ((1-beta) * dice)


def double_dice_loss(y_true, y_pred, ratio=10.0):
    return -dice_coefficient(y_true, y_pred) + ratio*dice_coefficient(1-y_true, y_pred)


def vod_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true),
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def quailty_est(true_diff, pred_diff, seg, alpha_q):
    """
    Quality estimation loss
    Contains both pixelwise overlap loss and MSE of predicted dice vs. truth dice
    :param y_true: truth difference from ground truth
    :param y_pred: predicted difference from ground truth
    :param seg: evaluated segmentation
    :param alpha_q: weight parameter
    :return:
    """
    #unify diff with mask
    eval_seg = tf.cast(seg, dtype=tf.int32)
    rounded_diff = tf.cast(tf.round(pred_diff), dtype=tf.int32)
    int_pred_diff = tf.cast(rounded_diff, dtype=tf.int32)
    updated_seg = tf.cast(tf.bitwise.bitwise_xor(eval_seg,int_pred_diff),dtype=tf.float32)
    rounded_diff = tf.cast(true_diff, dtype=tf.int32)
    int_true_diff = tf.cast(rounded_diff, dtype=tf.int32)
    truth_seg = tf.cast(tf.bitwise.bitwise_xor(eval_seg,int_true_diff), dtype=tf.float32)

    est_dice = dice_coefficient(updated_seg, seg)
    truth_dice = dice_coefficient(truth_seg, seg)

    se = (truth_dice - est_dice)*(truth_dice - est_dice)
    return -dice_coefficient(true_diff, pred_diff)+alpha_q*se


def error_ephesize_loss(truth, pred, error_diff, alpha=1):
    truth_diff_locations = truth * error_diff
    pred_diff_locations = pred * error_diff
    dice_error_locations = dice_coefficient(truth_diff_locations, pred_diff_locations)
    dice = dice_coefficient(truth, pred)

    return -dice - (alpha*dice_error_locations)


def dice_distance_weighted_loss(distance_map):
    def loss(y_true, y_pred):
        return -dice_distance_weighted(y_true, y_pred, distance_map)
    return loss


def quality_est_loss(seg, alpha_q, beta_q):
    def loss(y_true, y_pred):
        return quailty_est(y_true, y_pred, seg, alpha_q)
    return loss


def error_emphesize_loss(error_locations_mask, alpha_q):
    def loss(y_true, y_pred):
        return error_ephesize_loss(y_true, y_pred, error_locations_mask, alpha_q)
    return loss


def contour_losses_and_dice_loss_dyn_gamma(input_shape, beta, gamma_dyn, labmda=0, band=0, th=1):
    def loss(y_true, y_pred):
        return contour_losses_and_dice_loss_func(y_true, y_pred, input_shape, beta=beta, gamma=gamma_dyn.get_param(), labmda=labmda, band=band)
    return loss


def contour_losses_and_dice_loss(input_shape, beta, gamma, labmda=0, band=0, th=1):
    def loss(y_true, y_pred):
        return contour_losses_and_dice_loss_func(y_true, y_pred, input_shape, beta=beta, gamma=gamma, labmda=labmda, band=band)
    return loss


def dice_with_uncertainty_loss(uncertainty_map, u_th):
    def loss(y_true, y_pred):
        return -dice_with_uncertainty(y_true, y_pred, uncertainty_map, u_th)
    return loss


def contour_dice_tolerance_and_dice_uncertainty_loss(uncertainty_map, input_shape, beta, gamma, labmda=0, band=0,
                                                     u_th=0.5):
    """
    Loss function that performs selection of only certain voxels for dice loss only
    :param uncertainty_map: map with zeros for certain voxels
    :param input_shape:
    :param beta:
    :param gamma:
    :param labmda:
    :param band:
    :param u_th: uncertainty threshold
    :return:
    """
    def loss(y_true, y_pred):
        return contour_losses_and_dice_uncetainty_loss_func(y_true, y_pred, uncertainty_map, input_shape, beta=beta,
                                                            gamma=gamma, labmda=labmda, band=band, u_th=u_th)
    return loss


def uncertainty_contour_dice_tolerance_and_dice_loss(uncertainty_map, input_shape, beta, gamma, labmda=0, band=0,
                                                     u_th=0.5):
    """
    Loss function that performs selection of only certain voxels for both dice and conotur dice losses
    :param uncertainty_map: map with zeros for certain voxels
    :param input_shape:
    :param beta:
    :param gamma:
    :param labmda:
    :param band:
    :param u_th: uncertainty threshold
    :return:
    """
    def loss(y_true, y_pred):
        return uncetainty_contour_losses_and_dice_loss_func(y_true, y_pred, uncertainty_map, input_shape, beta=beta,
                                                    gamma=gamma, labmda=labmda, band=band, u_th=u_th)
    return loss


def contour_dice_tolerance_and_dice_loss(input_shape, beta, gamma, labmda=0, band=0):
    def loss(y_true, y_pred):
        return contour_dice_tolerance_and_dice_loss_func(y_true, y_pred, input_shape, beta=beta, gamma=gamma, labmda=labmda, band=band)
    return loss


def hausdorff_loss_dice_loss(beta, gamma_dyn):
    def loss(y_true, y_pred):
        return hausdorff_loss_dice_loss_func(y_true, y_pred, beta, gamma_dyn)
    return loss


def boundary_loss_dice_loss(beta, gamma_dyn):
    def loss(y_true, y_pred):
        return boundary_loss_dice_loss_func(y_true, y_pred, beta, gamma_dyn)
    return loss

def perimeter_loss_dice_loss(input_shape, beta, gamma_dyn):
    def loss(y_true, y_pred):
        return perimeter_loss_dice_loss_func(y_true, y_pred, input_shape, beta, gamma_dyn)
    return loss


def active_contour_loss(input_shape, lambda_ac):
    def loss(y_true, y_pred):
        return active_contour_loss_func(y_true, y_pred, input_shape, lambda_ac)
    return loss


def active_contour_assym_loss(input_shape, lambda_ac, beta):
    def loss(y_true, y_pred):
        return active_contour_assym_loss_func(y_true, y_pred, input_shape, lambda_ac, beta)
    return loss


def active_contour_contour_emphesize_loss(input_shape, lambda_ac, beta, gamma, theta=0):
    def loss(y_true, y_pred):
        return active_contour_contour_emphesize(y_true, y_pred, input_shape, lambda_ac, beta, gamma, theta)
    return loss


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def dice_cross_entropy_loss(y_true, y_pred):
    return dice_coefficient_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)


def dice_cross_entropy_normalized_length_loss(beta, gamma,t):
    def loss(y_true, y_pred):
        return dice_cross_entropy_normalized_length(y_true, y_pred, beta, gamma, t_max=t)
    return loss


def dice_cross_entropy_normalized_length_loss_max_min(beta, gamma, t_max=1.1, t_min=0.8):
    def loss(y_true, y_pred):
        return dice_cross_entropy_normalized_length_min_max(y_true, y_pred, beta, gamma, t_max=t_max, t_min=t_min)
    return loss


def dice_focal_normalized_length_loss(beta, gamma, t):
    def loss(y_true, y_pred):
        return dice_focal_normalized_length(y_true, y_pred, beta, gamma, t_max=t)
    return loss


def l1_dice_loss(z_size):
    def loss(y_true, y_pred):
        return l1_dice_loss_func(y_true, y_pred, z_size)
    return loss


def vod_coefficient_loss(y_true, y_pred):
    return -vod_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def dice_and_xent(y_true, y_pred, xent_weight=1.0, weight_mask=None):
    return dice_coef_loss(y_true, y_pred) + \
           xent_weight * weighted_cross_entropy_loss(y_true, y_pred, weight_mask)


def weighted_cross_entropy_loss(y_true, y_pred, weight_mask=None):
    xent = K.binary_crossentropy(y_true, y_pred)
    if weight_mask is not None:
        xent = K.prod(weight_mask, xent)
    return K.mean(xent)


def _focal_loss(gamma=2., alpha=.5):
    #Not sure it is working properly
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def tversky_loss(y_true, y_pred):
    """
    # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18
    :param y_true:
    :param y_pred:
    :return:
    """
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):

    return np.array([calc_dist_map(y)
                     for y in y_true]).astype(np.float32)


def perimeter_loss(y_true, y_pred, input_shape):
    """
    Perimeter loss implementation from the article: https://openreview.net/pdf?id=NDEmtyb4cXu
    Normalization is similar to Pytorch implementation: https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation/blob/master/losses.py
    :param y_true:
    :param y_pred:
    :return:
    """
    true_contours = extract_volume_2D_contours_tf(y_true[:,0,:,:,:], input_shape)
    pred_contours = extract_volume_2D_contours_tf(y_pred[:,0,:,:,:], input_shape)
    per_true = K.sum(true_contours, axis=[1,2,3])
    per_pred = K.sum(pred_contours, axis=[1,2,3])

    perimeter_loss = (per_true-per_pred)**2
    perimeter_loss = perimeter_loss/(input_shape[0]*input_shape[1]*input_shape[2])
    return K.mean(perimeter_loss, axis=0)


def boundary_loss_keras(y_true, y_pred):
    """
    Boundary loss implementation based on distance from boundary
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_dist_map = tf.py_func(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


def hausdorff_loss_keras(y_true, y_pred):
    """
    Hausdorff loss implementation based on the paper and the pytorch implementation
    :param y_true:
    :param y_pred:
    :return:
    """
    delta_s = (y_true-y_pred)**2
    y_true_dist_map = tf.py_func(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    y_pred_dist_map = tf.py_func(func=calc_dist_map_batch,
                                     inp=[y_pred],
                                     Tout=tf.float32)
    true_dtm = y_true_dist_map**2
    pred_dtm = y_pred_dist_map**2
    dtm = true_dtm + pred_dtm
    multiplied = tf.math.multiply(delta_s, dtm)
    hd_loss = K.mean(multiplied)

    return hd_loss


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
binary_crossentropy_loss = binary_crossentropy
#focal_loss = _focal_loss()