import os
from glob import glob
import nibabel as nib
import numpy as np
from pathlib import Path
from nilearn.image.image import _crop_img_to as crop_img_to, new_img_like
import argparse
from utils.read_write_data import save_nifti


def cut_with_roi_data(gt, volume, mask, gt_save_path, vol_save_path, mask_save_path, mask_filename, padding):

    #cut using GT data
        bbox_start, bbox_end = find_bounding_box(gt)
        if padding is not None:
            bbox_start = np.maximum(bbox_start - padding, 0)
            bbox_end = np.minimum(bbox_end + padding, gt.shape)

        volume = cut_bounding_box(volume, bbox_start, bbox_end)
        gt = cut_bounding_box(gt, bbox_start, bbox_end)
        nib.save(volume, vol_save_path)
        nib.save(gt, gt_save_path)

        if mask_filename is not None:
            mask = cut_bounding_box(mask, bbox_start, bbox_end)
            nib.save(mask, mask_save_path)

def cut_with_roi_data_np(gt, volume, mask, gt_save_path, vol_save_path, mask_save_path, mask_filename,
                         padding=np.array([16, 16, 8])):

    #cut using GT data
        bbox_start, bbox_end = find_bounding_box(gt)
        if padding is not None:
            bbox_start = np.maximum(bbox_start - padding, 0)
            bbox_end = np.minimum(bbox_end + padding, gt.shape)

        volume = volume[bbox_start[0]:bbox_end[0], bbox_start[1]:bbox_end[1], bbox_start[2]:bbox_end[2]]
        gt = gt[bbox_start[0]:bbox_end[0], bbox_start[1]:bbox_end[1], bbox_start[2]:bbox_end[2]]
        save_nifti(volume, vol_save_path)
        save_nifti(gt, gt_save_path)

        if mask_filename is not None:
            mask = mask[bbox_start[0]:bbox_end[0], bbox_start[1]:bbox_end[1], bbox_start[2]:bbox_end[2]]
            save_nifti(mask, mask_save_path)


def extract_mask_roi(src_dir, dst_dir, padding, volume_filename, gt_filename, mask_filename):
    for sample_folder in glob(os.path.join(src_dir, '*')):
        subject_id = Path(sample_folder).name
        dest_folder = os.path.join(dst_dir, subject_id)
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        volume_path = os.path.join(sample_folder, volume_filename)
        vol_save_path = os.path.join(dest_folder, Path(volume_path).name+'.gz')
        if(not os.path.exists(volume_path)):
            volume_path = os.path.join(sample_folder, volume_filename + '.gz')
            vol_save_path = os.path.join(dest_folder, Path(volume_path).name)
        gt_path = os.path.join(sample_folder, gt_filename)
        gt_save_path = os.path.join(dest_folder, Path(gt_path).name+'.gz')
        if(not os.path.exists(gt_path)):
            gt_path = os.path.join(sample_folder, gt_filename+'.gz')
            gt_save_path = os.path.join(dest_folder, Path(gt_path).name)

        volume = nib.load(volume_path).get_data()
        gt = nib.load(gt_path).get_data()

        mask = None
        mask_save_path = None
        if mask_filename is not None:
            mask_path = os.path.join(sample_folder, mask_filename)
            mask_save_path = os.path.join(dest_folder, Path(mask_path).name+'.gz')
            if(not os.path.exists(mask_path)):
                mask_path = os.path.join(sample_folder, mask_filename+'.gz')
                mask_save_path = os.path.join(dest_folder, Path(mask_path).name)
                mask = nib.load(mask_path)
                mask = new_img_like(volume, mask.get_data())

        cut_with_roi_data_np(gt, volume, mask, gt_save_path, vol_save_path, mask_save_path, mask_filename, padding)


def cut_bounding_box(img, start, end):
    slices = [slice(s, e) for s, e in zip(start, end)]
    return crop_img_to(img, slices, copy=True)


def find_bounding_box(mask):
    """
    Getting bounding box of segmentation mask
    :param mask: segmentaiton mask
    :return: start and end of the bounding box
    """
    try:
        coords = np.array(np.where(mask > 0.5))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1
    except:
        print("error in bounding box extraction")
        return np.asarray([0,0,0]), np.asarray(mask.shape)

#    assert np.sum(mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]) == np.sum(mask)
    return start, end


def check_bounding_box(mask, start, end):
    return np.sum(mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]) == np.sum(mask)


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", help="Source directory path",
                        type=str, required=True)
    parser.add_argument("--dst_dir", help="Directory with cutted data path",
                        type=str, required=True)
    parser.add_argument("--volume_filename", help="filename of a volume",
                        type=str, default='volume.nii')
    parser.add_argument("--gt_filename", help="mask filename",
                        type=str, default='truth.nii')
    parser.add_argument("--mask_filename", help="additional mask filename",
                        type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    """
    Script to create folder with mask ROI data for segmentation network
    """
    opts = get_arguments()

    padding = np.array([16, 16, 8])
    extract_mask_roi(opts.src_dir, opts.dst_dir, padding, opts.volume_filename, opts.gt_filename, opts.mask_filename)