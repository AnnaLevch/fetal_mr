from collections import namedtuple
import nibabel as nib
import os
import numpy as np
from PIL import Image
from data_curation.helper_functions import move_smallest_axis_to_z
#from utils.visualization import prepare_mask_for_plotting,prepare_vol_for_plotting, prepare_for_plotting
from data_generation.preprocess import window_1_99
import matplotlib.pyplot as plt
from evaluation.eval_utils.utils import *


def is_local_minimum(vol_slice_eval, index):
    """
    Find local minimum.
    :param vol_slice_eval:
    :param index:
    :param reverse_value:
    :return:
    """
    prev_index = index - 1
    next_index = index + 1
    curr_val = vol_slice_eval[index]

    if((prev_index not in vol_slice_eval) and (next_index in vol_slice_eval)):#if it is the first slice, check that next slice has higher dice
        if(vol_slice_eval[next_index] > curr_val):
            return True
        else:
            return False

    if((next_index not in vol_slice_eval) and (prev_index in vol_slice_eval)):#if it is the last slice, check that previous slice has higher dice
        if(vol_slice_eval[prev_index] > curr_val):
            return True
        else:
            return False

    #otherwise check that both previous and next slice have higher dice
    if((next_index in vol_slice_eval) and (prev_index in vol_slice_eval) and (vol_slice_eval[next_index]>curr_val) and (vol_slice_eval[prev_index]>curr_val)):
        return True
    else:
        return False


def get_key_slices_indexes(vol_slice_eval, num_key_images, thresh_value):
    """
    This function gets key images indices based on dice local minimas
    1. A maximum of num_key_images "bad" images
	2. [One median image for sanity check from the middle] - not implemented
	3. "Bad Image" is defined as local minima (slice after is higher) and below 92 dice
    If there are no "bad images", write 1 worst image
    """
    key_indexes_dict = {}
    sorted_dict = sorted(vol_slice_eval.items(), key=lambda item: item[1])
    curr_iter = iter(sorted_dict)

    num_chosen_images = 0
    while(num_chosen_images< num_key_images):
        try:
            item = curr_iter.__next__()
        except StopIteration:
            break

        index = item[0]
        value = vol_slice_eval[index]
        if(value > thresh_value and key_indexes_dict):#if the smallest value is larger than threshold, return index of the smallest value only
            break
        if(is_local_minimum(vol_slice_eval, index)):
            key_indexes_dict[index] = value
            num_chosen_images = num_chosen_images + 1

    return key_indexes_dict


def get_key_slices_indexes_largest(vol_slice_eval, num_key_images):
    """
    This function gets key images indices based on dice local minimas
    1. A maximum of num_key_images "bad" images
	2. [One median image for sanity check from the middle] - not implemented
	3. "Bad Image" is defined as local minima (slice after is higher) and below 92 dice
    If there are no "bad images", write 1 worst image
    """
    num_chosen_images = 0

    max_val = list_max(list(vol_slice_eval.values()))
    vol_slice_eval_reversed={}
    for key in vol_slice_eval:
        vol_slice_eval_reversed[key] = max_val - vol_slice_eval[key]

    key_indexes_dict = {}
    sorted_dict = sorted(vol_slice_eval_reversed.items(), key=lambda item: item[1])
    curr_iter = iter(sorted_dict)

    while(num_chosen_images< num_key_images):
        try:
            item = curr_iter.__next__()
        except StopIteration:
            break

        index = item[0]
        value = vol_slice_eval[index]

        if(is_local_minimum(vol_slice_eval_reversed, index)):
            key_indexes_dict[index] = value
            num_chosen_images = num_chosen_images + 1

    return key_indexes_dict


def overlay_masks(mask1_origin, mask2_origin):
    mask1 = mask1_origin.astype(np.uint8)
    mask2 = mask2_origin.astype(np.uint8)
    gt_res = np.zeros([mask1.shape[0], mask1.shape[1],3], dtype=np.uint8)
    gt_res[:,:,1] = mask1*mask2*255
    gt_res[:,:,0] = ((1-mask1)*mask2 + (1-mask2)*mask1)*255
    return Image.fromarray(gt_res).convert("RGBA")


def overlay_image_mask(img, mask, blend_factor=0.4):
    img = img*(255.0/img.max())

    img = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    gt_np = np.zeros([mask.shape[0], mask.shape[1],3], dtype=np.uint8)
    gt_np[:,:,0] = (mask.astype(np.uint8))*255
    gt = Image.fromarray(gt_np).convert("RGBA")
    return Image.blend(img, gt, blend_factor)

def convert_to_rgba(image):
    return Image.fromarray(image).convert("RGBA")


def resize_image(img,size):

    return img.resize(size, Image.ANTIALIAS)


def get_slice_truth_img(slice_img, truth_img):
    overlay_truth = overlay_image_mask(slice_img, truth_img, blend_factor=0.8)
    imgs_comb = np.hstack((np.array(convert_to_rgba(slice_img)), np.array(overlay_truth)))
    slice_truth = Image.fromarray(imgs_comb)
    new_size = (512, 256) #specify fixed size for key images
    slice_truth = resize_image(slice_truth, new_size)

    return slice_truth

def get_truth_res_img(slice_img, truth_img,res_img):
    overlay_res = overlay_image_mask(slice_img, res_img, blend_factor=0.8)
    overlay_truth_res = overlay_masks(truth_img, res_img)
    imgs_comb = np.hstack((np.array(overlay_res), np.array(overlay_truth_res)))
    slice_truth = Image.fromarray(imgs_comb)
    new_size = (512, 256) #specify fixed size for key images
    slice_truth = resize_image(slice_truth, new_size)

    return slice_truth

def prepare_for_plotting(volume, truth, pred):
    #swap x and y axes
    volume = np.swapaxes(volume,0,1)
    truth = np.swapaxes(truth,0,1)
    pred = np.swapaxes(pred,0,1)

    #transpose x axes
    volume = np.flip(volume, 1)
    truth = np.flip(truth, 1)
    pred = np.flip(pred, 1)

    #transpose y axes
    volume = np.flip(volume, 0)
    truth = np.flip(truth, 0)
    pred = np.flip(pred, 0)

    volume = window_1_99(volume, 0, 99)

    return volume, truth, pred


def prepare_vol_for_plotting(volume):
    volume = np.swapaxes(volume,0,1)
    volume = np.flip(volume, 1)
    volume = np.flip(volume, 0)

    volume = window_1_99(volume, 0, 99)

    return volume


def prepare_mask_for_plotting(mask):
    mask = np.swapaxes(mask,0,1)
    mask = np.flip(mask, 1)
    mask = np.flip(mask, 0)

    return mask


def get_and_save_anomaly_img(vol, truth, slice_num, out_folder, vol_id, postfix, images_pathes):
    slice = get_slice_truth_img(vol[:, :, slice_num], truth[:, :, slice_num])
    path = os.path.join(out_folder, vol_id + postfix)
    slice.save(path)
    images_pathes[vol_id].append(path)
    return images_pathes


def get_and_save_anomaly_img_with_result(vol, truth, result, slice_num, out_folder, vol_id, postfix, images_pathes):
    vol_truth = get_slice_truth_img(vol[:, :, slice_num], truth[:, :, slice_num])
    res_truth = get_truth_res_img(vol[:, :, slice_num], truth[:, :, slice_num], result[:, :, slice_num])
    imgs_comb = Image.fromarray(np.hstack((np.array(vol_truth), np.array(res_truth))))

    path = os.path.join(out_folder, vol_id + postfix)
    imgs_comb.save(path)
    images_pathes[vol_id].append(path)
    return images_pathes


def load_and_prepare_for_plotting(data_dir, vol_id, filename, is_mask):
    vol = nib.load(os.path.join(data_dir, vol_id, filename)).get_data()
    vol, swap_axis = move_smallest_axis_to_z(vol)
    if(is_mask == True):
        vol = prepare_mask_for_plotting(vol)
    else:#volume
       vol = prepare_vol_for_plotting(vol)
    return vol


def get_save_anomaly_images(vol, truth, slice_num, out_folder, vol_id, images_pathes):
    images_pathes = get_and_save_anomaly_img(vol, truth, slice_num-2, out_folder, vol_id, '_prev.png', images_pathes)
    images_pathes = get_and_save_anomaly_img(vol, truth, slice_num-1, out_folder, vol_id, '.png', images_pathes)
    images_pathes = get_and_save_anomaly_img(vol, truth, slice_num, out_folder, vol_id, '_next.png', images_pathes)
    return images_pathes


def get_save_anomaly_images_with_result(vol, truth, result, slice_num, out_folder, vol_id, images_pathes):
    images_pathes = get_and_save_anomaly_img_with_result(vol, truth, result, slice_num-2, out_folder, vol_id, '_prev.png', images_pathes)
    images_pathes = get_and_save_anomaly_img_with_result(vol, truth, result, slice_num-1, out_folder, vol_id,  '.png', images_pathes)
    images_pathes = get_and_save_anomaly_img_with_result(vol, truth, result, slice_num, out_folder, vol_id, '_next.png', images_pathes)
    return images_pathes


def save_anomaly_images(vol_id, vol_anomalies_dict, data_dir, truth_filename, vol_filename, res_filename, out_folder):
    """
    saves png anomaly slices with their truth overlay and consecutive slices to a folder
    Can be with or result image (res_filename can be None). outputing result image can be useful for cases like autoencoder where we want to compare
    autoencoder results with original mask

    """
    images_pathes = {}

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(os.path.join(data_dir, vol_id, truth_filename)):
        truth_filename += '.gz'
    if not os.path.exists(os.path.join(data_dir, vol_id, vol_filename)):
        vol_filename += '.gz'
    truth = load_and_prepare_for_plotting(data_dir, vol_id, truth_filename, is_mask=True)
    vol = load_and_prepare_for_plotting(data_dir, vol_id, vol_filename, is_mask=False)

    if(res_filename != None):
        if not os.path.exists(os.path.join(data_dir, vol_id, res_filename)):
            res_filename += '.gz'
        result = load_and_prepare_for_plotting(data_dir, vol_id, res_filename, is_mask=True)

    slices_list = vol_anomalies_dict[vol_id]
    for slice in slices_list:
        slice_num = int(slice)
        id = vol_id + '_' + slice
        images_pathes[id] = []
        if(res_filename != None):
            images_pathes = get_save_anomaly_images_with_result(vol, truth, result, slice_num, out_folder,id, images_pathes)
        else:
            images_pathes = get_save_anomaly_images(vol, truth, slice_num, out_folder, id, images_pathes)

    return images_pathes


def save_key_images(key_images_indices, eval_folder, out_folder, vol_id, volume_filename, truth_filename, pred_filename, eval_func):
    """
    saves png key images in the evaluation folder
    """
    images_pathes = {}
    print('saving key images for vol: ' + str(vol_id))
    folder_path = os.path.join(eval_folder, str(vol_id))
    truth = nib.load(os.path.join(folder_path, truth_filename)).get_data()
    pred = nib.load(os.path.join(folder_path, pred_filename)).get_data()
    volume = nib.load(os.path.join(folder_path, volume_filename)).get_data()

    truth, swap_axis = move_smallest_axis_to_z(truth)
    pred, swap_axis = move_smallest_axis_to_z(pred)
    volume, swap_axis = move_smallest_axis_to_z(volume)

    key_images_folder = out_folder + '_key_images/'
    if not os.path.exists(key_images_folder):
        os.makedirs(key_images_folder)

    volume, truth, pred = prepare_for_plotting(volume, truth, pred)
    for key in key_images_indices:
        func_val = "{0:.2f}".format(key_images_indices[key])
        slice_img = volume[:, :, key - 1]
        truth_img = truth[:, :, key - 1]
        pred_img = pred[:, :, key - 1]
        overlay_truth = overlay_image_mask(slice_img, truth_img)
        overlay_pred = overlay_image_mask(slice_img, pred_img)
        imgs_comb = np.hstack((np.array(overlay_truth), np.array(overlay_pred)))
        res_gt = Image.fromarray(imgs_comb)
        new_size = (512, 256) #specify fixed size for key images
        res_gt = resize_image(res_gt, new_size)

        image_path = key_images_folder + "image_{0}_{1}_{2}_{3}.png".format(vol_id, key, eval_func, func_val)
        res_gt.save(image_path)
        images_pathes[key] = image_path

    return images_pathes

