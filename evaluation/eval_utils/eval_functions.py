import random
import copy
from evaluation.surface_distance.metrics import *
from evaluation.surface_distance.lits_surface import *
from data_generation.extract_contours import extract_volume_2D_contours, extract_2D_contour


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def calc_dice_per_slice(test_truth, prediction_filled):
    dice_per_slice_dict = {}
    num_slices = test_truth.shape[2]

    for i in range(0,num_slices):
        indices_truth = np.nonzero(test_truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction_filled[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 & (len(indices_pred[0]) == 0)):
            continue
        dice_per_slice = dice(test_truth[:,:,i], prediction_filled[:,:,i])
        dice_per_slice_dict[i+1] = dice_per_slice

    return dice_per_slice_dict

def calc_overlap_measure_per_slice(truth, prediction, eval_function):
    """
    Calculate overlap measure. Make sure that either in result or ground truth there are some segmentation pixels
    :param truth:
    :param prediction:
    :param eval_function:
    :return:
    """
    eval_per_slice_dict = {}
    num_slices = truth.shape[2]

    for i in range(0,num_slices):
        #evaluate only slices that have at least one truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 and (len(indices_pred[0]) == 0)):
            continue

        eval_per_slice = eval_function(truth[:, :, i], prediction[:, :, i])
        eval_per_slice_dict[i+1] = eval_per_slice

    return eval_per_slice_dict


def calc_overlap_measure_per_slice_no_zero_pixels(truth, prediction, eval_function):
    """
    Calculate overlap measure. Make sure that BOTH in result and ground truth there are some segmentation pixels
    :param truth:
    :param prediction:
    :param eval_function:
    :return:
    """
    eval_per_slice_dict = {}
    num_slices = truth.shape[2]

    for i in range(0,num_slices):
        #evaluate only slices that have both truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 or (len(indices_pred[0]) == 0)):
            continue

        eval_per_slice = eval_function(truth[:, :, i], prediction[:, :, i])
        eval_per_slice_dict[i+1] = eval_per_slice

    return eval_per_slice_dict


def calc_distance_measure_per_slice(truth, prediction, resolution, eval_function):
    """
    Calculate overlap measure. Make sure that BOTH in result and ground truth there are some segmentation pixels
    """
    eval_per_slice_dict = {}
    num_slices = truth.shape[2]

    for i in range(0,num_slices):
        #evaluate only slices that have at least one truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 or (len(indices_pred[0]) == 0)):
            continue

        eval_per_slice = eval_function(truth[:, :, i], prediction[:, :, i], resolution[0:1])
        eval_per_slice_dict[i+1] = eval_per_slice

    return eval_per_slice_dict


def IoU(gt_seg, estimated_seg):
    """
    compute Intersection over Union
    :param gt_seg:
    :param estimated_seg:
    :return:
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(estimated_seg).astype(np.bool)

    # Compute IOU
    intersection = np.logical_and(seg1, seg2)
    union = np.logical_or(seg1, seg2)

    return intersection.sum() / union.sum()


def VOE(gt_seg, pred_seg):
    """
    compute volumetric overlap error (in percent) = 1 - intersection/union
    :param gt_seg:
    :param pred_seg:
    :return:
    """
    return 1 - IoU(gt_seg, pred_seg)


def seg_ROI_overlap(gt_seg, roi_pred):
    """
    compare ground truth segmentation to predicted ROI, return number of voxels from gt seg that aren't contained in
    the predicted ROI
    :param gt_seg: segmentation
    :param roi_pred: ROI represented as a binary segmentation
    :return:
    """
    seg = np.asarray(gt_seg).astype(np.bool)
    seg_roi = np.asarray(roi_pred).astype(np.bool)

    # if segmentation is bigger than intersection seg_roi, we are out of bounds
    intersection = np.logical_and(seg, seg_roi)
    return np.sum(seg ^ intersection)


def vod(mask1, mask2, verbose=False):
    mask1, mask2 = mask1.flatten(), mask2.flatten()
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if verbose:
        print('intersection\t', intersection)
        print('union\t\t', union)
    return 1 - (intersection + 1) / (union + 1)


def dice(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten() > 0
    y_pred_f = y_pred.flatten() > 0
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def pixel_difference(y_true, y_pred):
    return np.sum((1-y_pred)*y_true) + np.sum(y_pred*(1-y_true))


def nvd(mask1, mask2):
    mask1, mask2 = mask1.flatten(), mask2.flatten()
    sum_ = (mask1.sum() + mask2.sum())
    diff = abs(mask1.sum()-mask2.sum())
    return 2 * diff / sum_


def hosdorf_and_assd(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    assd = np.mean(compute_average_surface_distance(surface_distances))
    hausdorff = compute_robust_hausdorff(surface_distances, 100)
    return hausdorff, assd

def hausdorff2(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    hausdorff = compute_max_surface_distance(surface_distances)
    return hausdorff

def hausdorff(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    hausdorff = compute_max_surface_distance(surface_distances)
    return hausdorff


def hausdorff_robust(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    hausdorff = compute_robust_hausdorff(surface_distances, 95)
    return hausdorff

def assd(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    assd = np.mean(compute_average_surface_distance(surface_distances))
    return assd


def assd_lits(y_true, y_pred, scaling):
    if(len(y_true.shape)==3):
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling)
    else:#2D measure
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling, is_3D=False)
    assd = evalsurf.get_average_symmetric_surface_distance()
    return assd


def min_surface_distance(y_true, y_pred, scaling):
    if(len(y_true.shape)==3):
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling)
    else:#2D measure
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling, is_3D=False)
    min_distance = evalsurf.get_minimum_symmetric_surface_distance()
    return min_distance


def hausdorff_lits(y_true, y_pred, scaling):
    if(len(y_true.shape)==3):
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling)
    else:#2D measure
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling, is_3D=False)
    Hausdorff = evalsurf.get_maximum_symmetric_surface_distance()
    return Hausdorff


def volume_difference(y_true, y_pred, scaling):
    """
    Calculate the difference between volume calculation of result mask and ground truth mask in mm
    :param y_true:
    :param y_pred:
    :param scaling:
    :return:
    """
    return (volume(y_pred, scaling) - volume(y_true, scaling))/1000


def volume_difference_ratio(y_true, y_pred, scaling=None):
    """
    Calculate the difference between volume calculation of result mask and ground truth mask in mm and normalize by y_true volume
    :param y_true: reference volume
    :param y_pred: result volume
    :param scaling: scaling factor (in-plane and spacing)
    :return: volume difference ratio
    """
    # ref_volume = volume(y_true, scaling)
    # est_volume = volume(y_pred, scaling)
    # return (est_volume-ref_volume)/ref_volume

    num_nonzero_truth = len(np.nonzero(y_true)[0])
    num_nonzero_est = len(np.nonzero(y_pred)[0])
    return (num_nonzero_est-num_nonzero_truth)/num_nonzero_truth



def volume(mask, scaling):
    """
    Volume calculation
    :param mask: segmentation mask
    :param scaling: resolution- in-plane and pixel spacing in [x,y,z] format
    :return: volume calculation
    """
    num_nonzero = len(np.nonzero(mask)[0])
    return num_nonzero*scaling[0]*scaling[1]*scaling[2]


def num_nonzero(mask):
    """
    Number of voxels in the mask
    :param mask: mask
    :return: number of voxels
    """
    return len(np.nonzero(mask)[0])


def num_nonzero_2D(mask):
    """
    Number of nonzero voxels in 2D
    :param mask:
    :return:
    """
    non_zero={}
    for slice in range(0, mask.shape[2]):
        non_zero[slice]=len(np.nonzero(mask[:,:,slice])[0])

    return non_zero


def hausdorff_robust_lits(y_true, y_pred, scaling):
    if(len(y_true.shape)==3):
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling)
    else:#2D measure
        evalsurf = Surface(y_pred,y_true,physical_voxel_spacing = scaling, is_3D=False)
    Hausdorff = evalsurf.get_percentile_surface_distance(95)
    return Hausdorff


def surface_intersection_contour(y_true, y_pred):
    true_contour = extract_volume_2D_contours(y_true)
    pred_contour = extract_volume_2D_contours(y_pred)
    y_true_f = true_contour > 0
    y_pred_f = pred_contour > 0
    return np.int16(y_true_f * y_pred_f)


def surface_dice(y_true, y_pred):
    if(len(y_true.shape)==3):
        true_contour = extract_volume_2D_contours(y_true)
        pred_contour = extract_volume_2D_contours(y_pred)
    else:#2D measure
        true_contour = extract_2D_contour(y_true)
        pred_contour = extract_2D_contour(y_pred)
    surface_dice = dice(true_contour, pred_contour)
    return surface_dice


def false_negative_path_length_contour(y_true, y_pred):
    if(len(y_true.shape)==3):
        true_contour = extract_volume_2D_contours(y_true)
    else:#2D measure
        true_contour = extract_2D_contour(y_true)
    return (1-y_pred) * (true_contour)


def false_negative_path_length(y_true, y_pred):
    fnpl_contour = false_negative_path_length_contour(y_true, y_pred)
    fnpl = np.sum(fnpl_contour)
    return fnpl


def added_path_length_contour(y_true, y_pred):
    if(len(y_true.shape)==3):
        true_contour = extract_volume_2D_contours(y_true)
        pred_contour = extract_volume_2D_contours(y_pred)
    else:#2D measure
        true_contour = extract_2D_contour(y_true)
        pred_contour = extract_2D_contour(y_pred)
    return (1-pred_contour) * (true_contour)


def added_path_length(y_true, y_pred):
    apl_contour = added_path_length_contour(y_true, y_pred)
    apl = np.sum(apl_contour)
    return apl


class ErrorDetMetrics:

    def error_slices_det_precision(self, true_error_mask, pred_error_mask, num_pixels_th):
        """
        Precision of error slices detection with relaxation of prediction with est_band number of slices.
        "ground truth" error slice is defined as a slice with error larger than threshold
        :param true_error_mask: mask of truth error
        :param pred_error_mask: mask of predicted error
        :return: precision calculation
        """
        true_pos_cnt = 0
        false_pos_cnt = 0

        for slice in range(0, pred_error_mask.shape[2]):
            if len(np.nonzero(pred_error_mask[:,:,slice])[0]) > 0: #if the slice contains predicted errors
                if len(np.nonzero(true_error_mask[:,:,slice])[0]) > num_pixels_th:
                    true_pos_cnt += 1
                else:
                    false_pos_cnt += 1

        return true_pos_cnt/(true_pos_cnt+false_pos_cnt)


    def error_slices_det_recall(self, true_error_mask, pred_error_mask, num_pixels_th, relaxation=False):
        """
        Recall of error slices detection with relaxation of prediction with 1 slice.
        "ground truth" error slice is defined as a slice with error larger than threshold
        :param true_error_mask: mask of truth error
        :param pred_error_mask: mask of predicted error
        :return: recall calculation
        """
        true_pos_cnt = 0
        false_negative_cnt = 0
        est_slices = set()

        for slice in range(0, pred_error_mask.shape[2]):
            if len(np.nonzero(pred_error_mask[:,:,slice])[0]) > 0:
                est_slices.add(slice)

        for slice in range(0, true_error_mask.shape[2]):
            if len(np.nonzero(true_error_mask[:,:,slice])[0]) > num_pixels_th: #if the slice contains predicted errors
                if slice in est_slices:
                    true_pos_cnt += 1
                elif relaxation is True and ((slice-1) in est_slices) or ((slice+1) in est_slices):
                    true_pos_cnt += 1
                else:
                    false_negative_cnt += 1

        if (true_pos_cnt+false_negative_cnt) == 0:
            return -1
        return true_pos_cnt/(true_pos_cnt+false_negative_cnt)


    def eval_after_slice_correction(self, mask, pred_error_mask, truth_mask):
        """
        # A metric that calculates c
        :param mask:
        :param prediction:
        :param truth_unified:
        :return: Result metrics after correction of predicted slices
        """
        correct_slices = []
        for s in range(0, pred_error_mask.shape[2]):
            if len(np.nonzero(pred_error_mask[:,:,s])[0]) > 0:
                correct_slices.append(s)
        copied_mask = copy.deepcopy(mask)
        for s in correct_slices:
            copied_mask[:,:, s] = truth_mask[:,:,s]

        dice_score = dice(truth_mask, copied_mask)
        vdr = volume_difference_ratio(truth_mask, copied_mask)

        return dice_score, vdr, len(correct_slices)


    def calc_metrics_percent_correction(self, truth_mask, mask, correction_slices):
        """
        Given a dictionary of sorted slices, calculate metrics for different percentages
        :param truth_mask: truth segmentation
        :param mask: network result segmentation
        :param correction_slices: sorted slices to correct (from high score error to low score error)
        :return:
        """
        num_slices = truth_mask.shape[2]
        percentages = np.linspace(0, 100, num=11)

        dice_scores = {}
        vdr_scores = {}
        dice_scores[0] = dice(truth_mask, mask)
        vdr_scores[0] = volume_difference_ratio(truth_mask, mask)
        dice_scores[100] = 1
        vdr_scores[100] = 0

        for ind in range(1, len(percentages)-1):
            copied_mask = copy.deepcopy(mask)
            percentage = percentages[ind]
            num_slices_to_correct = np.int(np.round(percentage*num_slices/100))

            for slice in correction_slices:
                if num_slices_to_correct <= 0:
                    break
                copied_mask[:, :, slice] = truth_mask[:, :, slice]
                num_slices_to_correct -= 1
            dice_scores[percentage] = dice(truth_mask, copied_mask)
            vdr_scores[percentage] = volume_difference_ratio(truth_mask, copied_mask)

        return dice_scores, vdr_scores


    def eval_after_slice_correct_different_percentages(self, mask, pred_error_soft, truth_mask):
        """
        Evaluation after correction of different percentages of slices
        :param mask: segmentation mask to evaluate
        :param pred_error_soft: error prediction with soft labels before binarization
        :param truth_mask: truth segmentation
        :return: Result metrics after correction for different percentages
        """
        correct_slices = {}
        #calculate predicted error score by maximum score
        for s in range(0, pred_error_soft.shape[2]):
            nonzero = np.nonzero(pred_error_soft[:,:,s])
            if len(nonzero[0]) > 0:
                correct_slices[s] = np.max(pred_error_soft[nonzero[0], nonzero[1],s])
        correct_slices = dict(sorted(correct_slices.items(), key=lambda x:x[1], reverse=True))

        return self.calc_metrics_percent_correction(truth_mask, mask, correct_slices)


    def eval_after_sequential_correction_different_percentages(self, mask, truth_mask):
        """
        :param mask:
        :param truth_mask:
        :return:
        """
        slices = list(set(np.nonzero(mask)[2]))#should we use also 2 slices before?

        return self.calc_metrics_percent_correction(truth_mask, mask, slices)


    def eval_after_random_correction(self, mask, truth_mask, num_slices, rand_repeat=10):
        """
        A metric that calculates dice after correction of random slices
        random slices are picked rans_repeat number of times and the final dice is the average between rounds
        :param mask: prediction to correct
        :param truth_mask: truth segmentation
        :param num_slices: number of slices to randomly pick
        :return:
        """
        dice_rounds = []
        vdr_rounds = []
        for round in range(0,rand_repeat):
            seq = range(0,truth_mask.shape[2])
            slices_list = random.sample(seq, num_slices)
            round_mask = copy.deepcopy(mask)

            for s in slices_list:
                round_mask[:,:, s] = truth_mask[:,:,s]

            dice_rounds.append(dice(truth_mask, round_mask))
            vdr_rounds.append(volume_difference_ratio(truth_mask, round_mask))

        evg_dice = np.average(dice_rounds)
        evg_vdr = np.average(vdr_rounds)

        return evg_dice, evg_vdr


    def get_nonzero_consec_slices(self, mask):
        """
        Get nonzero slices in mask and 2 consecutive slices
        :param mask: segmentation mask
        :return:
        """
        num_mask_slices = mask.shape[2]
        nonzero_slices = list(set(np.nonzero(mask)[2]))
        min_nonzero = np.min(nonzero_slices)
        if min_nonzero - 1 >=0:
            nonzero_slices.append(min_nonzero - 1)
            if min_nonzero - 2 >= 0:
                nonzero_slices.append(min_nonzero - 2)
        max_nonzero = np.max(nonzero_slices)
        if max_nonzero + 1 < num_mask_slices:
            nonzero_slices.append(max_nonzero + 1)
            if max_nonzero + 2 < num_mask_slices:
                nonzero_slices.append(max_nonzero + 2)

        return nonzero_slices


    def calc_dicts_avg(self, dices, vdrs):
        percentages = dices[0].keys()
        dices_avg = {}
        vdrs_avg = {}
        for percentage in percentages:
            dices_sum=0
            vdrs_sum=0
            for dices_lst in dices:
                dices_sum += dices_lst[percentage]
            dices_avg[percentage] = dices_sum/len(dices)
            for vdrs_lst in vdrs:
                vdrs_sum += vdrs_lst[percentage]
            vdrs_avg[percentage] = vdrs_sum/(len(vdrs))

        return dices_avg, vdrs_avg


    def eval_after_rand_correct_percentages(self, mask, truth_mask, rand_repeat=10):
        """
        Calculate metrics after correcting random slices
        :param mask: segmentation mask to evaluate
        :param truth_mask: truth segmentation
        :return: Result metrics after correction for different percentages
        """
        slices = list(range(0,mask.shape[2]))
        dices = []
        vdrs = []
        for round in range(0,rand_repeat):
            random.shuffle(slices)
            dice_scores, vdr_scores = self.calc_metrics_percent_correction(truth_mask, mask, slices)
            dices.append(dice_scores)
            vdrs.append(vdr_scores)

        return self.calc_dicts_avg(dices, vdrs)


    def eval_after_rand_nonzero_correct_percentages(self, mask, truth_mask, rand_repeat=10):
        """
        Calculate metrics after correcting random slices with heuristics
        :param mask: segmentation mask to evaluate
        :param truth_mask: truth segmentation
        :return: Result metrics after correction for different percentages
        """
        nonzero_slices = self.get_nonzero_consec_slices(mask)
        dices = []
        vdrs = []
        for round in range(0,rand_repeat):
            random.shuffle(nonzero_slices)
            dice_scores, vdr_scores = self.calc_metrics_percent_correction(truth_mask, mask, nonzero_slices)
            dices.append(dice_scores)
            vdrs.append(vdr_scores)

        return self.calc_dicts_avg(dices, vdrs)


    def metrics_after_random_nonzero_correction(self, mask, truth_mask, num_slices, rand_repeat=10):
        """
        A metric that calculates dice after correction of random slices
        random slices are picked rand_repeat number of times and the final dice is the average between rounds
        Similar to dice_after_random_correction with the difference that only nonzero slices or consecutive 2 slices
        can be picked
        :param mask: prediction to correct
        :param truth_mask: truth segmentation
        :param num_slices: number of slices to randomly pick
        :return:
        """
        nonzero_slices = self.get_nonzero_consec_slices(mask)

        dice_rounds = []
        vdr_rounds = []
        for round in range(0,rand_repeat):
            slices_list = random.sample(nonzero_slices, num_slices)
            round_mask = copy.copy(mask)

            for s in slices_list:
                round_mask[:,:, s] = truth_mask[:,:,s]

            dice_rounds.append(dice(truth_mask, round_mask))
            vdr_rounds.append(volume_difference_ratio(truth_mask, round_mask))

        dice_avg = np.average(dice_rounds)
        vdr_avg = np.average(vdr_rounds)

        return dice_avg, vdr_avg