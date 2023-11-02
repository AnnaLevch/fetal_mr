import argparse
import cv2
import numpy as np
import nibabel as nib
import shutil
from scipy import ndimage
import os
from math import *
from PIL import Image
from skimage.filters import sobel_h, sobel_v
from data_generation.extract_contours import extract_volume_2D_contours
from utils.read_write_data import save_nifti
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis


def intensity_prior(img, gt, contour, w, k):
    """
     Calculate the intensity prior for a given image and a segmentation
    :param k: Kernel size k = 3,5,7
    :param w: arbitrary wight for the intensity prior
    :param contour: numpy binary array of the contour
    :param gt: ground truth
    :param img: single slice of MRI scan
    :return: intensity prior as a numpy array
    """

    kernel_size_config = {'3': (1, 2), '5': (2, 3), '7': (3, 4)}
    m, n = kernel_size_config[str(k)]
    sp_intensity = np.zeros(img.shape)
    if len(img.shape) > 2:
        for j in range(img.shape[2]):
            if gt[ :, :, j].sum() == 0:
                continue
            else:
                sx = cv2.Sobel(img[ :, :, j], cv2.CV_64F,1,0,ksize=3)
                sy = cv2.Sobel(img[ :, :, j], cv2.CV_64F,0,1,ksize=3)

                sp_intensity[ :, :, j][contour[ :, :, j]==1]=1
                sigma = np.std(img[ :, :, j][gt[ :, :, j]==1])
                c_points = np.argwhere(contour[ :, :, j] == 1)
                # get gradients:
                for i in range(len(c_points)):
                    x, y = c_points[i]
                    grad_x = np.average(sx[x-m:x+n,y-m:y+n])
                    grad_y = np.average(sy[x-m:x+n,y-m:y+n])
                    grad_tot = (grad_x**2 + grad_y**2)**0.5
                    if abs(grad_tot)<= sigma*w:
                        sp_intensity[ :, :, j][x, y] = abs(grad_tot)/(sigma*w)
                    else:
                        sp_intensity[ :, :, j][x, y] = 1
    else:
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sp_intensity = np.zeros(img.shape)
        sp_intensity[contour == 1] = 1
        sigma = np.std(img[gt == 1])
        c_points = np.argwhere(contour == 1)
        # get gradients:
        for i in range(len(c_points)):
            x, y = c_points[i]
            grad_x = np.average(sx[x - m:x + n, y - m:y + n])
            grad_y = np.average(sy[x - m:x + n, y - m:y + n])
            grad_tot = (grad_x ** 2 + grad_y ** 2) ** 0.5
            if abs(grad_tot) <= sigma * w:
                sp_intensity[x, y] = abs(grad_tot) / (sigma * w)
            else:
                sp_intensity[x, y] = 1
        return sp_intensity

    return sp_intensity


def get_im_data(data_path, w, k, vol_filename='volume.nii', truth_filename='truth.nii'):
    """
    :param k: kernel size
    :param w: arbitrary wight for the intensity prior
    :param data_path: Path to a folder with scan and ground truth data
    :return: A dictionary with the scans, ground truth and contour arrays
    """

    im_data = {}
    img = nib.load(os.path.join(data_path, vol_filename))
    im_data['img_mat'] = img.get_fdata()

    truth_img = nib.load(os.path.join(data_path, truth_filename))
    im_data['gt_mat'] = truth_img.get_fdata()

    im_data['img_mat'], swap_axis = move_smallest_axis_to_z(im_data['img_mat'])
    im_data['gt_mat'], swap_axis = move_smallest_axis_to_z(im_data['gt_mat'])

    im_data['contour'] = extract_volume_2D_contours(im_data['gt_mat'])
    im_data['intensity_pr'] = intensity_prior(im_data['img_mat'], im_data['gt_mat'], im_data['contour'], w, k)

    return im_data, swap_axis


def overlay_image_mask(img, mask, mask2=None):
    """
        :param img: numpy array of a scan
        :param mask: binary numpy array
        :param mask2: binary numpy array
        :return: blended image of img, mask and mask2
        """

    img *= 255.0/img.max()
    img = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    gt_np = np.zeros([mask.shape[0], mask.shape[1],3], dtype=np.uint8)
    gt_np[:, :, 0] = (mask.astype(np.uint8))*255
    if mask2 is not None:
        gt_np[:,:,1] = (mask2.astype(np.uint8))*255
    gt = Image.fromarray(gt_np).convert("RGBA")
    return Image.blend(img, gt, 0.4)


def get_surrounding_contour_pts(current_point, low_quality_pts, img_shape):
    """
    Checking adjecent low quality pixels 1 pixel away (3*3 neigborhood)
    :param current_point: point od reference
    :param low_quality_pts: points that are quality pixels
    :param img_shape: dimensions of the image
    :return: surrounding contour points
    """
    [x, y] = current_point
    surrounding_pts = []
    if(x-1>=0):
        if([x-1,y] in low_quality_pts):
            surrounding_pts.append([x-1,y])
        if((y-1>=0) and [x-1,y-1] in low_quality_pts):
            surrounding_pts.append([x-1,y-1])
        if(y+1<=img_shape[1] and [x-1, y+1] in low_quality_pts):
            surrounding_pts.append([x-1, y+1])
    if(y-1>=0 and [x, y-1] in low_quality_pts):
        surrounding_pts.append([x,y-1])
    if(y+1<=img_shape[1] and [x, y+1] in low_quality_pts):
        surrounding_pts.append([x,y+1])

    if(x+1<=img_shape[0]):
        if([x+1,y] in low_quality_pts):
            surrounding_pts.append([x+1,y])
        if(y-1>=0 and [x+1, y-1] in low_quality_pts):
            surrounding_pts.append([x+1,y-1])
        if(y+1<=img_shape[1] and [x+1, y+1] in low_quality_pts):
            surrounding_pts.append([x+1, y+1])

    return surrounding_pts


def pts_direction(curr_point, reference_point):
    """
    Calculate points derivative direction
    :param curr_point: current point
    :param reference_point: reference point
    :return: direction of points
    """
    EPSILON = 0.0001
    [x1, y1] = curr_point
    [x2, y2] = reference_point
    if(x2-x1!=0):
        derivative = (y2-y1)/(x2-x1)
    else:
        derivative = (y2-y1)/EPSILON

    return atan(derivative)


def get_segment_direction(x, y):
    """
    calculate mean points derivative direction
    :param img: a numpy array of a scan
    :param x, y : reference point
    :return: direction of points
    """
    pts_directions = []
    for i in range(0,len(x)-1):
        if(x[i+1]-x[i]!=0):
            angle = np.arctan((y[i+1]-y[i])/(x[i+1]-x[i]))
        else:
            angle = 1.571 #pi/2
        pts_directions.append(angle)
    segment_direction = np.mean(pts_directions)
    return segment_direction


def get_direction_old(img, x, y):
    """
     calculate points derivative direction
    :param img: a numpy array of a scan
    :param x, y : reference point
    :return: direction of points
    """
    z = len(x)
    sx = sobel_v(img)
    sy = sobel_h(img)
    sx[sx == 0] = float(inf)
    grad_direction = np.arctan(sy/sx)
    #z = len(grad_direction[grad_direction != 0])
    deg = (np.sum(grad_direction[x, y]))/z
    return deg


def low_quality_segments(im_dict, T_quality, T_direction):
    """
    Find low quality segments to expand later on
    :param im_dict: dictionary storing a scan, its contour and the ground truth
    :param T_quality: Quality threshold
    :param T_direction: Direction threshold
    :return: low quality segments represented by lists of points
    """
    intensity_prior_img = im_dict['intensity_pr']
    segments = []
    low_quality_natrix = np.matrix(np.where((intensity_prior_img < T_quality) & (intensity_prior_img > 0))).T
    low_quality_pts = low_quality_natrix.tolist()
    while (len(low_quality_pts) > 0):
        current_point = low_quality_pts.pop()
        surrounding_pts = get_surrounding_contour_pts(current_point, low_quality_pts, intensity_prior_img.shape)

        if (len(surrounding_pts) == 0):  # we need at least 2 points to form a segment
            continue

        # segment initialization
        segment_direction = pts_direction(current_point, surrounding_pts[0])
        segment_pts = [current_point, surrounding_pts[0]]
        segment_stack = [surrounding_pts[0]]
        segment_directions = [segment_direction]
        low_quality_pts.remove(surrounding_pts[0])

        for i in range(1, len(surrounding_pts)):
            direction = pts_direction(current_point, surrounding_pts[i])
            if (abs(segment_direction - direction) < T_direction):
                segment_pts.append(surrounding_pts[i])
                segment_stack.append(surrounding_pts[i])
                segment_directions.append(direction)
                segment_direction = np.average(segment_directions)
                low_quality_pts.remove(surrounding_pts[i])

        while (len(segment_stack) > 0):
            current_point = segment_stack.pop()
            surrounding_pts = get_surrounding_contour_pts(current_point, low_quality_pts, intensity_prior_img.shape)
            if (len(surrounding_pts) == 0):
                continue

            for i in range(0, len(surrounding_pts)):
                direction = pts_direction(current_point, surrounding_pts[i])
                if (abs(segment_direction - direction) < T_direction):
                    segment_pts.append(surrounding_pts[i])
                    segment_stack.append(surrounding_pts[i])
                    segment_directions.append(direction)
                    segment_direction = np.average(segment_directions)
                    low_quality_pts.remove(surrounding_pts[i])

        segments.append(segment_pts)

    return segments


def segment_expand(img, gt, curr_mask, y_tot, x_tot, dy, dx, T_length, TSPQ, w, k ):

    """
    Find low quality segments to expand later on
    :param img: numpy array of a scan
    :param gt: ground truth
    :param curr_mask: contour with the updated expanded segments
    :param y_tot: all y coordinates of the current segment
    :param x_tot: all x coordinates of the current segment
    :param dy: segment y direction

    :param dx: segment x direction
    :param T_length: maximum steps allowed to expand the segment
    :param TSPQ: total segmentation prior quality,
    :param k: kernel size
    :param w: arbitrary wight for the intensity prior

    :return: low quality segment represented by lists of points
    """

    new_segments = np.zeros(img.shape)
    curr_xy = (x_tot, y_tot)
    for t in range(1, T_length):

        y_new = np.around(y_tot + t * dy).astype(int)
        x_new = np.around(x_tot + t * dx).astype(int)

        [x_new, y_new] = remove_border_indices(img.shape, x_new, y_new)

        curr_mask[x_new, y_new] = 1

        TSPQ_new = np.sum(intensity_prior(img, gt, curr_mask, w, k))

        if abs(TSPQ - TSPQ_new) < 2:  # 9
            if not np.array_equal(x_new, x_tot) or not np.array_equal(y_new, y_tot):
                new_segments[x_new, y_new] = 1
            else:
                continue
        else:
            x_final, y_final = curr_xy
            new_segments[x_final, y_final] = 1  # contains only the new segments
            break
    return new_segments


def remove_border_indices(shape, x_arr, y_arr):
    x_new = []
    y_new = []
    for i in range(x_arr.size):
        if x_arr[i]>=0 and x_arr[i]<shape[0] and y_arr[i]>=0 and y_arr[i]<shape[1]:
            x_new.append(x_arr[i])
            y_new.append(y_arr[i])

    return x_new, y_new


def get_segmentation_variability(im_dict, T_length, intervals, w, k):
    """
    This function performs variability estimation using intensity prior
    :param k: kernel size
    :param w: arbitrary wight for the intensity prior
    :param im_dict: dictionary storing a scan its contour and the ground truth
    :param intervals: list of low quality intervals
    :param T_length: maximum steps allowed to expand the segment
    :return: dictionary with the variability estimation in directions pi, 0
    """
    img, gt, contour, intensity_prior_img = im_dict['img_mat'], im_dict['gt_mat'], im_dict['contour'], im_dict[
        'intensity_pr']
    new_seg = {}
    TSPQ = np.sum(intensity_prior_img)

    new_seg['segments_0'] = np.zeros(img.shape)
    new_seg['segments_pi'] = np.zeros(img.shape)  # saves only the new intervals
    for i in range(len(intervals)):
        #print(str(i) + r'/' + str(len(intervals)))
        curr_mask_0 = contour.copy()
        curr_mask_pi = contour.copy()

        x_tot = np.asarray(intervals[i])[:, 0]
        y_tot = np.asarray(intervals[i])[:, 1]

        curr_mask_0[x_tot, y_tot] = 0
        curr_mask_pi[x_tot, y_tot] = 0

        direction = get_segment_direction(x_tot, y_tot)
      #  direction = get_direction_old(img, x_tot, y_tot)
        segment_direction_y = sin(direction)
        segment_direction_x = cos(direction)

        dy_0 = segment_direction_y
        dx_0 = segment_direction_x

        dy_pi = -dy_0 # ceil
        dx_pi = -dx_0

        new_seg['segments_0'] += segment_expand(img, gt, curr_mask_0, y_tot, x_tot, dy_0, dx_0, T_length, TSPQ, w, k)
        new_seg['segments_pi'] += segment_expand(img, gt, curr_mask_pi, y_tot, x_tot, dy_pi, dx_pi, T_length, TSPQ, w, k)

    new_seg['segments_0'][new_seg['segments_0'] > 0] = 1
    new_seg['segments_pi'][new_seg['segments_pi'] > 0] = 1

    return new_seg


def calc_prior_based_var_est(data_path, save_path = None, k=5, T_quality=0.7, T_length=5, T_direction=np.pi / 6, w=1.5,
                             vol_filename='volume.nii.gz', truth_filename='truth.nii.gz'):

    """
    variability estimation
    :param data_path: data path to directory with a segmentation and scans with the names truth & volume
    :param save_path: path to save images of the variability estimation, if not specified images will not be saved
                      possible, consensus and variability are saved
    :param k: kernel size
    :param T_length: maximum steps allowed to expand the segment
    :param T_quality: Quality threshold
    :param T_direction: Direction threshold
    :param w: arbitrary wight for the intensity prior
    :return: directory with variability estimations for the scans, a .txt file with the chosen parameters.
    """

    if os.path.exists(os.path.join(data_path, vol_filename)) is False:
        vol_filename += '.gz'
    if os.path.exists(os.path.join(data_path, truth_filename)) is False:
        truth_filename += '.gz'
    im_data, swap_axis = get_im_data(data_path, w, k, vol_filename, truth_filename)

    variability_volume = {}
    variability_weighted = {}
    total_gt_area = 0
    total_uncertainty_area = 0

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            file = open(os.path.join(save_path, "parameters.txt"), "w")
            file.write('Parameters:' + '\n')
            file.write("k=" + str(k) + '\n')
            file.write("T_quality=" + str(T_quality) + '\n')
            file.write("T_direction=" + str(T_direction) + '\n')
            file.write("T_length=" + str(T_length) + '\n')
            file.write("w=" + str(w) + '\n')
            file.close()
    variability_seg = np.zeros(im_data['img_mat'].shape)
    for idx in range( im_data['img_mat'].shape[2]):
        slice_normal, slice_truth, slice_contour, res_ints = im_data['img_mat'][:, :, idx], im_data['gt_mat'][ :, :, idx], im_data['contour'][ :, :, idx], im_data['intensity_pr'][ :, :, idx]

        if slice_truth.sum() == 0:
            continue
        else:
            curr_im_data = {}

            curr_im_data['img_mat'] = im_data['img_mat'][:, :, idx]
            curr_im_data['gt_mat'] = im_data['gt_mat'][:, :, idx]
            curr_im_data['contour'] = im_data['contour'][:, :, idx]
            curr_im_data['intensity_pr'] = im_data['intensity_pr'][:, :, idx]

            low_quality_seg = low_quality_segments(curr_im_data, T_quality, T_direction)

            res_dict = get_segmentation_variability(curr_im_data, T_length, low_quality_seg, w, k)

            variability = slice_contour - (res_dict['segments_0'] + res_dict['segments_pi'])
            variability[variability == 1] = 0
            variability[variability != 0] = 1
            variability_plot = ndimage.median_filter(variability, size=(3, 3))
      #      variability_plot = ndimage.median_filter(variability + slice_contour, size=(3, 3))
            variability_plot[variability_plot != 0] = 1
            low_quality = np.zeros(slice_normal.shape)
            low_quality[res_ints < 1] = 1
            low_quality[res_ints == 0] = 0

            curr = variability + slice_contour
            curr[curr>0] = 1
            variability_seg[:, :, idx] = variability_plot  #curr

        variability_volume[idx] = round((variability.sum()/slice_truth.sum())*100, 2)
        variability_weighted[idx] = variability.sum()
        total_gt_area += slice_truth.sum()
        total_uncertainty_area += variability.sum()

    im_data['gt_mat'] = swap_to_original_axis(swap_axis, im_data['gt_mat'])
    variability_seg = swap_to_original_axis(swap_axis, variability_seg)
    save_variability_results(data_path, im_data['gt_mat'], save_path, variability_seg, vol_filename, truth_filename)


def save_variability_results(data_path, gt_data, save_path, variability_seg, vol_filename, truth_filename):
    """
    save data and variability estimation results
    :param data_path: path to data folder to copy data
    :param gt_data: ground truth
    :param save_path: path to save variability
    :param variability_seg: variability estimation
    :param vol_filename: filename of the volume
    :return:
    """
    shutil.copy(os.path.join(data_path, vol_filename), os.path.join(save_path, vol_filename))
    shutil.copy(os.path.join(data_path, truth_filename), os.path.join(save_path, truth_filename))
    #smooth consensus
    consensus_seg = np.copy(gt_data)
    consensus_seg[np.nonzero(variability_seg)] = 0
  #  consensus_seg = ndimage.median_filter(consensus_seg, size=(3, 3,3))
    save_nifti(consensus_seg, os.path.join(save_path, 'consensus.nii.gz'))

    possible_seg = np.copy(gt_data)
    possible_seg[np.nonzero(variability_seg)] = 1

    #update variability
    new_variability_seg = possible_seg - consensus_seg
    new_variability_seg[new_variability_seg<=0]=0
    save_nifti(new_variability_seg, os.path.join(save_path, 'variability.nii.gz'))
    new_possible_seg = np.copy(gt_data)
    new_possible_seg[np.nonzero(new_variability_seg)] = 1
    save_nifti(new_possible_seg, os.path.join(save_path, 'possible.nii.gz'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--save_path", help="specifies dir to save variability estimation",
                        type=str, required=False)

    parser.add_argument("--k", help="kernel_size",
                        type=int,default=5, required=False) #True
    parser.add_argument("--T_quality", help="segment quality threshold",
                        type=float, default=0.7, required=False)
    parser.add_argument("--T_length", help="maximum steps allowed to expand a segment",
                        type=int, default=5, required=False)
    parser.add_argument("--T_direction", help="direction threshold for segment expansion",
                        type=float, default=np.pi/6, required=False) #0.52
    parser.add_argument("--w", help="arbitrary weight",
                        type=float, default=1.5, required=False)
    opts = parser.parse_args()

    calc_prior_based_var_est(opts.data_path, opts.save_path, opts.k, opts.T_quality, opts.T_length, opts.T_direction, opts.w)