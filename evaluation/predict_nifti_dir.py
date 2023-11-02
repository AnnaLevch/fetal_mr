import argparse
import glob
import json
import shutil
from pathlib import Path
import numpy as np
from scipy import ndimage
from data_curation.helper_functions import *
from data_generation.cut_roi_data import find_bounding_box, check_bounding_box
from data_generation.preprocess import window_1_99, normalize_data, window_1_99_2D, adapt_hist
from evaluation.eval_utils.postprocess import postprocess_prediction
from evaluation.eval_utils.prediction import patch_wise_prediction, predict_augment, predict_flips
from training.train_functions.training import load_old_model, get_last_model_path
from utils.read_write_data import list_load
from utils.read_write_data import save_nifti, save_nifti_with_metadata,  read_img, read_nifti_vol_meta
import data_generation.preprocess
from utils.arguments import str2bool
from evaluation.uncertainty.entropy import entropy


def round_up(num, divisor):
    if divisor is 1:
        return num
    return num + (num%divisor)


def get_padding(vol_shape, padding, division):
    padding[0] = round_up(vol_shape[0] + padding[0], division[0]) - vol_shape[0]
    padding[1] = round_up(vol_shape[1] + padding[1], division[1]) - vol_shape[1]
    padding[2] = round_up(vol_shape[2] + padding[2], division[2]) - vol_shape[2]
    return padding


def secondary_prediction_iterate(mask, vol, config2, model2, preprocess_method2=None, norm_params2=None,
                                 scale=None, overlap_factor=0.9, augment2=None, num_augment=10,
                                 return_all_preds=False, input_mask=None, iterations = 2):
    """
    This function supports multiple detection iterations to refine detection results
    :param mask:
    :param vol:
    :param config2:
    :param model2:
    :param preprocess_method2:
    :param norm_params2:
    :param scale:
    :param overlap_factor:
    :param augment2:
    :param num_augment:
    :param return_all_preds:
    :param input_mask:
    :param iterations: iterations-1 detection iterations, the last iteration is segmentation
    :return:
    """
    if return_all_preds is True:
        raise Exception('iterative detection is not supported for TTA return all preds')

    num_iterations = iterations

    while num_iterations>0:

        bbox_end, bbox_start, data, data_size = extract_volume_box(mask, scale, vol, padding = [16, 16, 4])
        print('box start = ' + str(bbox_start) + ', box end is: ' + str(bbox_end))
        data = preproc_and_norm(data, preprocess_method2, norm_params2, scale=scale,preproc=config2.get('preprocess', None))
        prediction, uncertainty = get_prediction(data, model2, augment=augment2, num_augments=num_augment, return_all_preds=return_all_preds,
                                    overlap_factor=overlap_factor, config=config2, input_mask=input_mask)
        if scale is not None and (scale[0] != 1.0 or scale[1] != 1.0 or scale[2] != 1.0):#revert to original size
            prediction = ndimage.zoom(prediction.squeeze(), [data_size[0]/prediction.shape[0], data_size[1]/prediction.shape[1], data_size[2]/prediction.shape[2]], order=1)
            if uncertainty is not None:
                uncertainty = ndimage.zoom(uncertainty.squeeze(), [data_size[0]/uncertainty.shape[0], data_size[1]/uncertainty.shape[1], data_size[2]/uncertainty.shape[2]], order=1)
        #pad to volume size
        padding2 = list(zip(bbox_start, np.array(vol.shape) - bbox_end))
        prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)
        uncertainty = np.pad(uncertainty, padding2, mode='constant', constant_values=0)
        mask = np.int16(postprocess_prediction(prediction,  threshold=0.5, fill_holes=fill_holes))

        num_iterations-=1

    return prediction, uncertainty


def secondary_prediction(mask, vol, config2, model2,
                         preprocess_method2=None, norm_params2=None, scale=None,
                         overlap_factor=0.9, augment2=None, num_augment=10, return_all_preds=False, input_mask=None):

    bbox_end, bbox_start, data, data_size = extract_volume_box(mask, scale, vol)


    data = preproc_and_norm(data, preprocess_method2, norm_params2, scale=scale,preproc=config2.get('preprocess', None))

    prediction, uncertainty, _ = get_prediction(data, model2, augment=augment2, num_augments=num_augment, return_all_preds=return_all_preds,
                                overlap_factor=overlap_factor, config=config2, input_mask=input_mask)

    if scale is not None:#revert to original size
        if return_all_preds is False:
            prediction = ndimage.zoom(prediction.squeeze(), [data_size[0]/prediction.shape[0], data_size[1]/prediction.shape[1], data_size[2]/prediction.shape[2]], order=1)
        else:
            zoomed_predictions = []
            for i in range(len(prediction)):
                zoomed_prediction = ndimage.zoom(prediction[i], [data_size[0] / prediction[i].shape[0],
                                                             data_size[1] / prediction[i].shape[1],
                                                             data_size[2] / prediction[i].shape[2]], order=1)
                zoomed_predictions.append(zoomed_prediction)
            prediction = zoomed_predictions
        if uncertainty is not None:
                uncertainty = ndimage.zoom(uncertainty.squeeze(), [data_size[0]/uncertainty.shape[0], data_size[1]/uncertainty.shape[1], data_size[2]/uncertainty.shape[2]], order=1)

    #pad to volume size
    padding2 = list(zip(bbox_start, np.array(vol.shape) - bbox_end))
    if uncertainty is not None:
        uncertainty = np.pad(uncertainty, padding2, mode='constant', constant_values=0)

    if return_all_preds:
        padding2 = [(0, 0)] + padding2
    else:
        print(padding2)
        print(prediction.shape)
    prediction = np.pad(prediction, padding2, mode='constant', constant_values=0)

    return prediction, uncertainty


def extract_volume_box(mask, scale, vol, padding = [16, 16, 8]):
    pred = mask
    bbox_start, bbox_end = find_bounding_box(pred)
    check_bounding_box(pred, bbox_start, bbox_end)
    box_size = bbox_end - bbox_start
    bbox_start = np.maximum(bbox_start - padding, 0).astype(int)
    if (scale is not None):
        padding = get_padding(box_size, padding, [1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2]])
    bbox_end = np.minimum(bbox_end + padding, mask.shape).astype(int)
    data = vol.astype(np.float)[
           bbox_start[0]:bbox_end[0],
           bbox_start[1]:bbox_end[1],
           bbox_start[2]:bbox_end[2]
           ]
    data_size = data.shape
    return bbox_end, bbox_start, data, data_size


def is_in_preprocess(preprocesses, method):
    preprocess_methods = set(preprocesses.split(';'))
    if(method in preprocess_methods):
        return True
    return False


def preproc_and_norm(data, preprocess_methods=None, norm_params=None, scale=None, preproc=None):
    if preprocess_methods is not None:
        print('Applying preprocess by {}...'.format(preprocess_methods))
        try:
            if is_in_preprocess(preprocess_methods, 'window_1_99'):
                data = window_1_99(data)
            if is_in_preprocess(preprocess_methods, 'window_1_99_2D'):
                data = window_1_99_2D(data)
        except Exception as e:
            print('preprocessing exception: ' + str(e))

    if scale is not None:
        data = ndimage.zoom(data, scale)
    if preproc is not None:
        preproc_func = getattr(data_generation.preprocess, preproc)
        data = preproc_func(data)

    # data = normalize_data(data, mean=data.mean(), std=data.std())
    if norm_params is not None and any(norm_params.values()):
        data = normalize_data(data, mean=norm_params['mean'], std=norm_params['std'])
    try:
        if is_in_preprocess(preprocess_methods, 'adapt_hist'):
                data = adapt_hist(data)
    except Exception as e:
            print('preprocessing exception: ' + str(e))

    return data


def get_prediction(data, model, augment, num_augments, return_all_preds, overlap_factor, config, input_mask, visualize_tta=False):
    uncertainty = None
    transformed_data_pred = None

    if augment is not None:
        patch_shape = config["patch_shape"] + [config["patch_depth"]]
        if augment == 'all':
            predictions, transformed_data_pred = predict_augment(data, model=model, overlap_factor=overlap_factor, num_augments=num_augments,
                                         patch_shape=patch_shape, input_mask=input_mask, visualize=visualize_tta)
        elif augment == 'flip':
            predictions = predict_flips(data, model=model, overlap_factor=overlap_factor, patch_shape=patch_shape, config=config)
        else:
            raise ("Unknown augmentation {}".format(augment))
        if not return_all_preds:
            uncertainty = entropy(predictions)
            prediction = np.median(predictions, axis=0)
        else:
            uncertainty = entropy(predictions)
            median_prediction = np.median(predictions, axis=0)
            prediction = np.append(predictions,np.expand_dims(median_prediction,0), axis=0)
    else:
        prediction = \
            patch_wise_prediction(model=model,
                                  data=np.expand_dims(data, 0),
                                  overlap_factor=overlap_factor,
                                  patch_shape=config["patch_shape"] + [config["patch_depth"]], input_mask=input_mask)
        prediction = prediction.squeeze()

    return prediction, uncertainty, transformed_data_pred


def delete_nii_gz(s):
    if s[-3:] == '.gz':
        s = s[:-3]
    if s[-4:] == '.nii':
        s = s[:-4]
    return s


def get_scan_res_fov(scan_id, metadata_path):
    curr_fov = get_FOV(scan_id, metadata_path=metadata_path)
    curr_resolution = resolution_from_scan_name(scan_id)[0:2]
    return curr_fov, curr_resolution


def scale_data(data, xy_scale, z_scale, xy_autoscale=None, z_autoscale = None, norm_params=None, scan_id=None, metadata_path=None,scan_series_id_from_name=True):

    if(xy_autoscale is True):
        # if metadata_path is not None and 'xy_fov' in norm_params:
        #     res_train = np.array(norm_params['xy_resolution'])/np.array(norm_params['xy_fov'])
        #     curr_fov, curr_resolution = get_scan_res_fov(scan_id, metadata_path)
        #     res_scan = np.array(curr_resolution)/np.array(curr_fov)
        # else: #use only xy resolution from name
        res_train = np.array(norm_params['xy_resolution'])
        resolution = resolution_from_scan_name(scan_id)
        if resolution is not None:
            xy_curr_resolution = resolution[0:2]
        else:
            xy_curr_resolution = get_resolution(scan_id, metadata_path=metadata_path, extract_scan_series_id=scan_series_id_from_name)
        res_scan = np.array(xy_curr_resolution)
        scale = [i / j for i, j in zip(res_scan, res_train)]
        x_scale = scale[0]
        y_scale= scale[1]

    elif (xy_scale is None):
        x_scale = y_scale = 1.0
    else:
        x_scale = xy_scale
        y_scale = xy_scale

    if z_autoscale is True:
        z_scale_train = np.array(norm_params['z_scale'])
        spacing = get_spacing_between_slices(scan_id, metadata_path=metadata_path, extract_scan_series_id=scan_series_id_from_name)
        z_scale = spacing/z_scale_train
    elif (z_scale is None):
        z_scale = 1.0

    if(x_scale!=1 or y_scale!=1 or z_scale!=1):
        data = ndimage.zoom(data, [x_scale, y_scale, z_scale])

    print('scaled data by:' + str([x_scale, y_scale, z_scale]))
    return data, [x_scale, y_scale, z_scale]


def unify_scale(scale, cfg):
    cfg_scale = cfg.get('scale', None)
    if (cfg_scale is not None):
        if(scale[0] != 1.0 or scale[1] != 1.0 or scale[2] != 1.0):
            return [scale[0]*cfg_scale[0], scale[1]*cfg_scale[1], scale[2]*cfg_scale[2]]
        return cfg_scale
    return scale


def run_inference(input_path, output_path, scan_id,
                  config, model, overlap_factor=0.7, has_gt=False, preprocess_methods=None, norm_params=None, augment=None, num_augment=0,
                  config2=None, model2=None, preprocess_method2=None, norm_params2=None, augment2=None, num_augment2=0,
                  z_scale=None, xy_scale=None, return_all_preds=False, xy_autoscale=None, z_autoscale=None, fill_holes=True, connected_component=True, input_mask_path=None,
                  metadata_path=None, truth_filename='truth.nii', detection_iter=1, scan_series_id_from_name=True, save_soft_labels=False, visualize_tta=False):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + 'test/' + str(scan_id) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Loading nifti from {}...'.format(input_path))
    nifti_data, affine, header = read_nifti_vol_meta(input_path)
    print('Predicting mask...')
    save_nifti_with_metadata(nifti_data, affine, header, os.path.join(output_path, 'data.nii.gz'))
    nifti_data, swap_axis = move_smallest_axis_to_z(nifti_data)
    data_size = nifti_data.shape
    data = nifti_data.astype(np.float).squeeze()
    print('original_shape: ' + str(data.shape))

    data, scale = scale_data(data, xy_scale, z_scale, xy_autoscale, z_autoscale, norm_params, scan_id, metadata_path,scan_series_id_from_name)

    net_input_scale = config.get('scale', None)
    if net_input_scale is not None:
        cfg1_scale = np.array(net_input_scale)*np.array(scale)
    else:
        cfg1_scale = np.array(scale)
    data = preproc_and_norm(data, preprocess_methods, norm_params,
                            scale=net_input_scale,
                            preproc=config.get('preprocess', None))
    mask_data = None
    if input_mask_path is not None:
        input_mask = read_img(input_mask_path).get_fdata()
        input_mask, swap_axis = move_smallest_axis_to_z(input_mask)
        mask_data = input_mask.astype(np.float).squeeze()
        mask_data = ndimage.zoom(mask_data,cfg1_scale, order=0)

 #   data = np.pad(data, 3, 'constant', constant_values=data.min())
    print("case: " + str(scan_id) + '\n' +'Shape: ' + str(data.shape) )

    #returns single prediction in the regular case, multiple predictions in case of test time augmentations and return_all_preds=True
    prediction, uncertainty, transformed_data_pred = get_prediction(data=data, model=model, augment=augment,
                                num_augments=num_augment, return_all_preds=return_all_preds,
                                overlap_factor=overlap_factor, config=config, input_mask=mask_data,
                                                                    visualize_tta=visualize_tta)

    #unify scaling
    if config.get('scale', None) is not None:
        cfg_scale = config.get('scale', None)
        cfg1_scale = [a*b for a,b in zip(scale,cfg_scale)]
    else:
        cfg1_scale = scale

    mask, prediction, uncertainty = scale_and_postprocess(augment, cfg1_scale, connected_component, data_size, fill_holes,
                                             prediction, return_all_preds, uncertainty)

    if config2 is not None:
        swapped_mask = swap_to_original_axis(swap_axis, mask)
        save_nifti(np.int16(swapped_mask), os.path.join(output_path, 'prediction_all.nii.gz'))
        scale = unify_scale(scale, config2)
        print('segmentation scale: ' + str(scale))
        if detection_iter>1:
            prediction, uncertainty = secondary_prediction_iterate(mask, vol=nifti_data.astype(np.float),
                                              config2=config2, model2=model2,
                                              preprocess_method2=preprocess_method2, norm_params2=norm_params2,
                                              overlap_factor=overlap_factor, augment2=augment2, num_augment=num_augment2,
                                              return_all_preds=return_all_preds, scale=scale, input_mask=mask_data, iterations = detection_iter)
        else:
            prediction, uncertainty = secondary_prediction(mask, vol=nifti_data.astype(np.float),
                                              config2=config2, model2=model2,
                                              preprocess_method2=preprocess_method2, norm_params2=norm_params2,
                                              overlap_factor=overlap_factor, augment2=augment2, num_augment=num_augment2,
                                              return_all_preds=return_all_preds, scale=scale, input_mask=mask_data)

        if return_all_preds is False:
            prediction_binarized = np.int16(postprocess_prediction(prediction,  threshold=0.5, fill_holes=fill_holes))
        else:
            prediction_binarized = np.zeros(np.array(prediction).shape, dtype=np.int16)
            for i in range(len(prediction)):
                prediction_binarized[i] = postprocess_prediction(prediction[i], threshold=0.5, fill_holes=fill_holes,
                                                 connected_component=connected_component)

        save_prediction(prediction_binarized, output_path, return_all_preds,swap_axis, save_soft_labels, prediction,
                        uncertainty)

    else: #if there is no secondary prediction, save the first network prediction or predictions as the final ones
        save_prediction(mask, output_path, return_all_preds, swap_axis, save_soft_labels, prediction, uncertainty,
                        transformed_data_pred)

    if(has_gt):
        save_truth(input_path, output_path, truth_filename)
    if(input_mask_path is not None):
        shutil.copy(input_mask_path, os.path.join(output_path, 'mask.nii.gz'))

    print('Saving to {}'.format(output_path))
    print('Finished.')


def save_truth(input_path, output_path, truth_filename):
    """
    Save ground truth data
    :param input_path: ground truth directory path
    :param output_path: output directory path
    :param truth_filename: filename of ground truth to be copied
    :return:
    """
    volume_dir = os.path.dirname(input_path)
    gt_path = os.path.join(volume_dir, truth_filename)
    if (not os.path.exists(gt_path)):
        gt_path = os.path.join(volume_dir, 'truth.nii.gz')
    if (not os.path.exists(gt_path)):
        gt_path = os.path.join(volume_dir, 'prediction_dafi.nii')
    if (not os.path.exists(gt_path)):
        gt_path = os.path.join(volume_dir, 'prediction_dafi.nii.gz')
    truth = read_img(gt_path).get_fdata()
    save_nifti(np.int16(truth), os.path.join(output_path, 'truth.nii.gz'))


def save_prediction(mask, output_path, return_all_preds, swap_axis, save_soft_labels=False, prediction=None,
                    uncertainty=None, transformed_data_pred=None):
    """
    Save all outputs
    :param mask: output after postprocessing
    :param output_path: path to output folder
    :param augment: TTA augmentations (None if were not used)
    :param return_all_preds: should return all predictions in case of TTA?
    :param swap_axis: axis for swapping pack (performed swapping in te beginning for aligning z axis)
    :param save_soft_labels: should we save soft labels before binarization?
    :param prediction: soft labels before binarization
    :return:
    """
    if uncertainty is not None:
        uncertainty = swap_to_original_axis(swap_axis, uncertainty)
        save_nifti(uncertainty, os.path.join(output_path, 'uncertainty.nii.gz'))

    if return_all_preds is True:
        for i in range(len(mask)-1):
            tta_mask = swap_to_original_axis(swap_axis, mask[i])
            save_nifti(tta_mask, os.path.join(output_path, 'tta' + str(i) + '_prediction.nii.gz'))
            if transformed_data_pred is not None:
                transformed_data_pred[i*2+1]= np.int16(postprocess_prediction(transformed_data_pred[i*2+1]))#postprocess mask
                save_nifti(transformed_data_pred[i*2], os.path.join(output_path, 'transformed_data' + str(i) + '.nii.gz'))
                save_nifti(transformed_data_pred[i*2+1], os.path.join(output_path, 'transformed_pred' + str(i) + '.nii.gz'))

        save_nifti(mask[-1], os.path.join(output_path, 'prediction.nii.gz'))
        if save_soft_labels is True:
            mean_prediction = prediction[-1]
            mean_prediction = swap_to_original_axis(swap_axis, mean_prediction)
            save_nifti(mean_prediction, os.path.join(output_path, 'prediction_soft.nii.gz'))
    else:
        mask = swap_to_original_axis(swap_axis, mask)
        save_nifti(mask, os.path.join(output_path, 'prediction.nii.gz'))
        if save_soft_labels is True:
            prediction = swap_to_original_axis(swap_axis, prediction)
            save_nifti(prediction, os.path.join(output_path, 'prediction_soft.nii.gz'))


def scale_and_postprocess(augment, cfg1_scale, connected_component, data_size, fill_holes, prediction, return_all_preds, uncertainty):
    if (augment is None) or (return_all_preds is False):
        # revert to original size
        if cfg1_scale is not None and (cfg1_scale[0] != 1.0 or cfg1_scale[1] != 1.0 or cfg1_scale[2] != 1.0):
            prediction = ndimage.zoom(prediction,
                                      [data_size[0] / prediction.shape[0], data_size[1] / prediction.shape[1],
                                       data_size[2] / prediction.shape[2]], order=1)
            if uncertainty is not None:
                uncertainty = ndimage.zoom(uncertainty,
                                      [data_size[0] / uncertainty.shape[0], data_size[1] / uncertainty.shape[1],
                                       data_size[2] / uncertainty.shape[2]], order=1)
        mask = postprocess_prediction(prediction, threshold=0.5, fill_holes=fill_holes,
                                      connected_component=connected_component)
        mask = np.int16(mask)
    else:
        if cfg1_scale is not None and (cfg1_scale[0] != 1.0 or cfg1_scale[1] != 1.0 or cfg1_scale[2] != 1.0):
            zoomed_predictions = []
            for i in range(len(prediction)):
                zoomed_prediction = ndimage.zoom(prediction[i], [data_size[0] / prediction[i].shape[0],
                                                             data_size[1] / prediction[i].shape[1],
                                                             data_size[2] / prediction[i].shape[2]], order=1)
                zoomed_predictions.append(zoomed_prediction)
            prediction = zoomed_predictions
            if uncertainty is not None:
                uncertainty = ndimage.zoom(uncertainty,
                                      [data_size[0] / uncertainty.shape[0], data_size[1] / uncertainty.shape[1],
                                       data_size[2] / uncertainty.shape[2]], order=1)
        mask = np.zeros(np.array(prediction).shape, dtype=np.int16)
        for i in range(len(prediction)):
            mask[i] = postprocess_prediction(prediction[i], threshold=0.5, fill_holes=fill_holes,
                                             connected_component=connected_component)
    return mask, prediction, uncertainty


def get_params(config_dir):
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        __config = json.load(f)
    does_exist = os.path.exists(os.path.join(config_dir, 'norm_params.json'))
    with open(os.path.join(config_dir, 'norm_params.json'), 'r') as f:
        __norm_params = json.load(f)
    __model_path = os.path.join(config_dir, 'epoch_')

    return __config, __norm_params, __model_path


def predict_single_case(volume_dir, opts, is_labeled, _config, _config2, _model, _model2):
    scan_id = os.path.basename(volume_dir)
    volume_path = get_volume_path(volume_dir)
    if(opts.mask_filename is not None):
       input_mask_path = os.path.join(opts.input_path, input_dir, opts.mask_filename)
    else:
        input_mask_path = None
    print('input path is: ' + volume_path)
    run_inference(volume_path, opts.output_folder, scan_id, has_gt=is_labeled, overlap_factor=opts.overlap_factor,
                          config=_config, model=_model, preprocess_methods=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
                          num_augment=opts.num_augment,
                          config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
                          num_augment2=opts.num_augment2, z_scale=opts.z_scale, xy_scale=opts.xy_scale, z_autoscale=opts.z_autoscale, return_all_preds=opts.return_all_preds,
                          xy_autoscale=opts.xy_autoscale, fill_holes=opts.fill_holes, input_mask_path=input_mask_path, metadata_path=opts.metadata_path,
                          truth_filename=opts.truth_filename, connected_component=opts.connected_component, detection_iter=opts.detection_iter,
                          scan_series_id_from_name=opts.scan_series_id_from_name, save_soft_labels=opts.save_soft_labels, visualize_tta=opts.visualize_tta)


def get_volume_path(path):
    """
    Loading volume based on predefined names. If they don't exist, using the first nifti file encountered.
    """
    volume_path = os.path.join(path,'volume.nii')
    if (not os.path.exists(volume_path)):
        volume_path = os.path.join(path,'volume.nii.gz')
    if (not os.path.exists(volume_path)):
        volume_path = os.path.join(path,'data.nii')
    if (not os.path.exists(volume_path)):
        volume_path = os.path.join(path,'data.nii.gz')
    if (not os.path.exists(volume_path)):
        volume_path = os.path.join(path,'cropped.nii.gz')
    if (not os.path.exists(volume_path)):
        filenames = glob.glob(os.path.join(path,'Pat*.nii*'))
        if (len(filenames) != 0):
            volume_path = filenames[0]
        else:
            print('Notice: Using the first nifti file as volume')
            filenames = glob.glob(os.path.join(path,'*.nii*'))
            if (len(filenames) != 0):
                volume_path = filenames[0]

    return volume_path


def get_pathes(pathes):
    if '%' not in pathes:#only one data path
        return [pathes]
    else:
        return pathes.split('%')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="specifies nifti file dir path or filename in case of single prediction",
                        type=str, required=True)
    parser.add_argument("--ids_list", help="specifies which scans from the directory to use for inference. "
                                           "By default, all scans from the directory are used. Expected to be in config_dir",
                        type=str, required=False)
    parser.add_argument("--output_folder", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.7)
    parser.add_argument("--z_scale", help="scling of z axis",
                        type=float, default=1)
    parser.add_argument("--z_autoscale", help="automatic z scaling. expects z_scale to be updated in norm_params and metadata path of test scans",
                        type=str2bool, default=False)
    parser.add_argument("--xy_scale", help="specifies xy scaing to perform",
                        type=float, default=None)
    parser.add_argument("--xy_autoscale", help="automatic scaling according to resolution. Expects to have resolution data in norm params",
                        type=str2bool, default=False)
    parser.add_argument("--metadata_path", help="path of metadata for FOV and resolution extraction. If not specified, resolution will be taken from name",
                        type=str, required=False)
    parser.add_argument("--labeled", help="in case of labeled data, copy ground truth for convenience",
                        type=str2bool, default=False)
    parser.add_argument("--all_in_one_dir", help="in case of unlabeled data, this option allows to have all the volumes in one directory without directory hierarchy",
                        type=str2bool, default=False)
    parser.add_argument("--save_soft_labels", help="save labels before binarization",
                        type=str2bool, default=False)
    parser.add_argument("--mask_filename", help="filename of mask if mask is given as input",
                        type=str, required=False, default=None)
    parser.add_argument("--truth_filename", help="filename of truth file if not default",
                        type=str, required=False, default="truth.nii")
    parser.add_argument("--predict_single", help="predict a single case",
                        type=str2bool, default=False)
    # Params for primary prediction
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--preprocess", help="which preprocess to do",   #currently, 'window_1_99',  'window_1_99_2D' and 'adapt_hist" preprocessings are supported
                                                                        #for multiple preprocessings, cancatenate them with ';' delimiter
                        type=str, required=False, default=None)
    #Augmentations
    parser.add_argument("--augment", help="what augment to do",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment", help="what augment to do",
                        type=int, required=False, default=0)  # one of 'flip, all'
    parser.add_argument("--return_all_preds", help="return all predictions or mean result for prediction?",
                        type=str2bool, default=False)
    parser.add_argument("--visualize_tta", help="should we save transformed data and results of TTA. Use only for visualization",
                        type=str2bool, default=False)

    # Params for secondary prediction
    parser.add_argument("--config2_dir", help="specifies config dir path",
                        type=str, required=False, default=None)
    parser.add_argument("--preprocess2", help="what preprocess to do",
                        type=str, required=False, default=None)
    parser.add_argument("--augment2", help="Should we do TTA",
                        type=str, required=False, default=None)  # one of 'flip, all'
    parser.add_argument("--num_augment2", help="number of TTA",
                        type=int, required=False, default=0)  # one of 'flip, all'
    parser.add_argument("--fill_holes", help="what augment to do",
                        type=str2bool, required=False, default=True)  # one of 'flip, all'
    parser.add_argument("--connected_component", help="should we extract one 3D connected component",
                        type=str2bool, required=False, default=True)  # one of 'flip, all'
    parser.add_argument("--detection_iter", help="number of detection refinement iterations",
                        type=int, required=False, default=1)
    parser.add_argument("--scan_series_id_from_name", help="should we extract scan and series id from name to get metadata. Set to True if metadata has this values",
                        type=str2bool, required=False, default=True)
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':

    opts = get_arguments()

    print(opts.input_path)
    is_labeled = opts.labeled
    path = Path(opts.output_folder)
    if(os.path.exists(path.parent)==False):
        path.parent.mkdir()
    path.mkdir(exist_ok=True)

    with open(os.path.join(opts.output_folder, 'inference_args.json'), 'wt') as f:
        json.dump(vars(opts), f, indent=4)
    _config, _norm_params, _model_path = get_params(opts.config_dir)
    if opts.config2_dir is not None:
        _config2, _norm_params2, _model2_path = get_params(opts.config2_dir)
    else:
        _config2, _norm_params2, _model2_path = None, None, None
    if(opts.ids_list != None):
        scans_list = list_load(os.path.join(opts.config_dir, opts.ids_list))

    last_model_path = get_last_model_path(_model_path)
    print('First:' + last_model_path)
    _model = load_old_model(last_model_path, build_manually=False)

    if(_model2_path is not None):
        last_model2_path = get_last_model_path(_model2_path)
        print('Second:' + last_model2_path)
        _model2 = load_old_model(last_model2_path, build_manually=False)
    else: _model2 = None

    all_in_one_dir = opts.all_in_one_dir #is the data arranged in directory hierarchy or all volumes in one dir?
    predict_single = opts.predict_single

    if(all_in_one_dir):
        for volume_path in glob.glob(os.path.join(opts.input_path, '*')):
            scan_id = origin_id_from_filepath(volume_path)
            if opts.ids_list != None:
                patient_id = patient_id_from_filepath(volume_path)
                if patient_id in scans_list:#use only scans that are specified in ids_list
                    continue

            print('input path is: ' + volume_path)
            run_inference(volume_path, opts.output_folder, scan_id, has_gt=is_labeled, overlap_factor=opts.overlap_factor,
                          config=_config, model=_model, preprocess_methods=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
                          num_augment=opts.num_augment, config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2,
                          augment2=opts.augment2,num_augment2=opts.num_augment2, z_scale=opts.z_scale, xy_scale=opts.xy_scale,
                          return_all_preds=opts.return_all_preds, xy_autoscale=opts.xy_autoscale, z_autoscale=opts.z_autoscale, fill_holes=opts.fill_holes,
                          input_mask_path=None, metadata_path=opts.metadata_path, truth_filename=opts.truth_filename, connected_component=opts.connected_component,
                          detection_iter=opts.detection_iter, save_soft_labels=opts.save_soft_labels, visualize_tta=opts.visualize_tta)
    elif(predict_single):
        filepath = opts.input_path
        predict_single_case(filepath, opts, is_labeled, _config, _config2, _model, _model2)
    else:
        input_pathes = get_pathes(opts.input_path)
        for input_path in input_pathes:
            scans_dirs = os.listdir(input_path)
            for input_dir in scans_dirs:
                scan_id = input_dir
                if (opts.ids_list != None) and (scan_id not in scans_list):#use only scans that are specified in ids_list
                    continue
                volume_path = get_volume_path(os.path.join(input_path, scan_id))
                if(opts.mask_filename is not None):
                    input_mask_path = os.path.join(opts.input_path, input_dir, opts.mask_filename)
                else:
                    input_mask_path = None

                print('input path is: ' + volume_path)
                run_inference(volume_path, opts.output_folder, scan_id, has_gt=is_labeled, overlap_factor=opts.overlap_factor,
                              config=_config, model=_model, preprocess_methods=opts.preprocess, norm_params=_norm_params, augment=opts.augment,
                              num_augment=opts.num_augment,
                              config2=_config2, model2=_model2, preprocess_method2=opts.preprocess2, norm_params2=_norm_params2, augment2=opts.augment2,
                              num_augment2=opts.num_augment2, z_scale=opts.z_scale, xy_scale=opts.xy_scale, z_autoscale=opts.z_autoscale, return_all_preds=opts.return_all_preds,
                              xy_autoscale=opts.xy_autoscale, fill_holes=opts.fill_holes, input_mask_path=input_mask_path, metadata_path=opts.metadata_path,
                              truth_filename=opts.truth_filename, connected_component=opts.connected_component, detection_iter=opts.detection_iter,
                              scan_series_id_from_name=opts.scan_series_id_from_name, save_soft_labels=opts.save_soft_labels, visualize_tta=opts.visualize_tta)

