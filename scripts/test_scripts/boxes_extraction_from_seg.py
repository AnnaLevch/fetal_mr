import glob
import os
import nibabel as nib
import numpy as np
from evaluation.detection.boxes.boxes_from_seg import ROIsFromSeg
from utils.read_write_data import save_nifti
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis


def draw_boxes(case_boxes, case_data):
    boxes_vis = np.zeros_like(case_data)
    for slice in range(0, case_data.shape[2]):
        slice_boxes = case_boxes[slice]
        for case_box in slice_boxes:
            boxes_vis[case_box[0][0]:case_box[1][0], case_box[0][1]:case_box[1][1], slice]=1
    return boxes_vis


def extract_cases_boxes(cases_path, filename, min_area, box_filename, save_box, boxes_min_dist):
    dirs = glob.glob(os.path.join(cases_path,'*/'))
    cases_boxes = {}
    for directory in dirs:
        dir_name = os.path.dirname(directory)
        print('extracting boxes for case ' + dir_name)
        case_data = nib.load(os.path.join(directory, filename)).get_data()
        case_data, swap_axis = move_smallest_axis_to_z(case_data)
        case_boxes = ROIsFromSeg.boxes_from_seg2D(case_data)
        case_boxes = ROIsFromSeg.unify_nearby_boxes(case_boxes, boxes_min_dist)
        case_boxes = ROIsFromSeg.remove_boxes_with_small_area(case_boxes, min_area)
        cases_boxes[dir_name] = case_boxes
        boxes_vis = draw_boxes(case_boxes, case_data)
        if save_box is True:
            boxes_vis = swap_to_original_axis(swap_axis, boxes_vis)
            save_nifti(boxes_vis, os.path.join(directory, box_filename))
    return case_boxes


if __name__ == "__main__":
    cases_path = '/home/bella/Phd/code/code_bella/log/638/output/TRUFI_qe_0/test'
    result_filename = 'prediction.nii.gz'
    gt_filename = 'truth.nii.gz'
    result_box_filename = 'pred_boxes_closed_unified.nii.gz'
    gt_box_filename = 'truth_boxes_closed_unified.nii.gz'

    case_boxes = extract_cases_boxes(cases_path, result_filename, 50, result_box_filename, True, boxes_min_dist=5)
    case_boxes = extract_cases_boxes(cases_path, gt_filename, 50, gt_box_filename, True, boxes_min_dist=5)

