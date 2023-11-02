import os
import shutil


if __name__ == "__main__":
    """
    Unifies gt from 2 directories for easy comparison. E.g. unified dataset with labeling errors and clean dataset for comparison
    """
    dir_1 = "/home/bella/Phd/data/body/FIESTA/FIESTA_origin_gt_errors/"
    dir_2 = "/home/bella/Phd/data/body/FIESTA/FIESTA_origin_clean/"
    out_dir = "/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/unified_gt_errors_clean/"
    prefix_dir1 = 'lerrors_'
    prefix_dir2 = 'clean_'

    dirs = os.listdir(dir_1)
    for scan_dir in dirs:
        os.mkdir(os.path.join(out_dir, scan_dir))
        volume_filename = 'volume.nii'
        dir_1_vol_path = os.path.join(dir_1, scan_dir, volume_filename)
        if not os.path.exists(dir_1_vol_path):
            volume_filename = 'volume.nii.gz'
            dir_1_vol_path = os.path.join(dir_1, scan_dir, volume_filename)
        truth1_filename = 'truth.nii'
        dir_1_truth_path = os.path.join(dir_1,scan_dir, truth1_filename)
        if not os.path.exists(dir_1_truth_path):
            truth1_filename = 'truth.nii.gz'
            dir_1_truth_path = os.path.join(dir_1,scan_dir, truth1_filename)
        truth2_filename = 'truth.nii'
        dir_2_truth_path = os.path.join(dir_2,scan_dir, truth2_filename)
        if not os.path.exists(dir_2_truth_path):
            truth2_filename = 'truth.nii.gz'
            dir_2_truth_path = os.path.join(dir_2,scan_dir, truth2_filename)
        out_vol_path = os.path.join(out_dir, scan_dir, volume_filename)
        out_truth1_path = os.path.join(out_dir, scan_dir, prefix_dir1 + truth1_filename)
        out_truth2_path = os.path.join(out_dir, scan_dir, prefix_dir2 + truth2_filename)

        shutil.copyfile(dir_1_vol_path, out_vol_path)
        shutil.copyfile(dir_1_truth_path, out_truth1_path)
        shutil.copyfile(dir_2_truth_path, out_truth2_path)