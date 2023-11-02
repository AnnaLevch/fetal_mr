import numpy as np
from utils.image_manipulation import connected_components_above_thresh
from scipy.ndimage.morphology import binary_closing
import glob
import os
import nibabel as nib


class ROIsFromSeg:
    @staticmethod
    def boxes_from_seg3D(seg, min_connected):
        """
        Extract bounding boxes of connected components with at least min_connected number of voxels
        :param seg: segmentation mask
        :param min_connected: minimum number of voxels in a connected component
        :return: boxes in (x1, y1, x2, y2, z1, z2) coordinates.
        """
        boxes = []
        mask_large_components, components_num = connected_components_above_thresh(seg, min_connected)
        for i in components_num:
            coords = np.array(np.where(mask_large_components == i))
            start = coords.min(axis=1)
            end = coords.max(axis=1) + 1
            boxes.append([start,end])

        return boxes


    @staticmethod
    def boxes_from_seg2D(seg, min_connected=0):
        """
        Extract bounding boxes of connected components with at least min_connected number of voxels
        :param seg: segmentation mask
        :param min_connected: minimum number of voxels in a connected component
        :return: boxes
        """
        boxes_slices = {}
        seg = binary_closing(seg)
        for slice in range(0, seg.shape[2]):
            boxes_slices[slice] = []
            mask_large_components, components_num = connected_components_above_thresh(seg[:, :, slice], min_connected)
            for i in components_num:
                coords = np.array(np.where(mask_large_components == i))
                start = coords.min(axis=1)
                end = coords.max(axis=1) + 1
                num_pixels = len(coords[0])
                boxes_slices[slice].append([start,end, num_pixels])

        return boxes_slices

    @staticmethod
    def are_overlapping(box1, box2, boxes_min_dist):
        """
        Check if boxes are overlapping or within min distance between each other
        :param box1: box1 upper left and lower right points
        :param box2: box1 upper left and lower right points
        :param boxes_min_dist: Minimum distance between boxes
        :return:
        """

        if box1[1][0] + boxes_min_dist < box2[0][0]:
            return False
        if box1[1][1] + boxes_min_dist < box2[0][1]:
            return False
        if box2[1][0] + boxes_min_dist < box1[0][0]:
            return False
        if box2[1][1] + boxes_min_dist < box1[0][1]:
            return False
        return True


    @staticmethod
    def unify_nearby_boxes(case_boxes, boxes_min_dist):
        """
        Unifying nearby boxes
        :param cases_boxes: extracted boxes in the format [min_point, max_point, num_pixels_in_box]
        :param boxes_distance: Maximum distance between boxes to unify
        :return: Unified boxes
        """
        for key in case_boxes.keys():
            slice_boxes = case_boxes[key]
            if len(slice_boxes) < 2:
                continue
            new_boxes = []
            while len(slice_boxes) > 0:
                box = slice_boxes[0]
                del slice_boxes[0]
                items_to_remove=[]
                for i in range(0,len(slice_boxes)):
                    box2 = slice_boxes[i]
                    if ROIsFromSeg.are_overlapping(box,box2, boxes_min_dist) is True:
                        box[0][0] = min(box[0][0], box2[0][0])
                        box[0][1] = min(box[0][1], box2[0][1])
                        box[1][0] = max(box[1][0], box2[1][0])
                        box[1][1] = max(box[1][1], box2[1][1])
                        box[2] += box2[2]
                        items_to_remove.append(i)
                #delete unified boxes
                for ind in sorted(items_to_remove, reverse=True):
                    del slice_boxes[ind]
                new_boxes.append(box)
            case_boxes[key] = new_boxes
        return case_boxes


    @staticmethod
    def remove_boxes_with_small_area(case_boxes, min_area):
        for key in case_boxes.keys():
            slice_boxes = case_boxes[key]
            items_to_remove=[]
            for i in range(0,len(slice_boxes)):
                if slice_boxes[i][2] < min_area:
                    items_to_remove.append(i)
            for ind in sorted(items_to_remove, reverse=True):
                del slice_boxes[ind]
            case_boxes[key] = slice_boxes

        return case_boxes


