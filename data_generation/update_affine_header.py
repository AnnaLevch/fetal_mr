import glob
import os
from utils.read_write_data import read_nifti_vol_meta, save_nifti_with_metadata
import nibabel as nib


if __name__ == "__main__":
    data_path = '\\\\10.101.119.14\\Dafna\\Bella\\data\\Body\\TRUFI\\TRUFI\\'
    filenames_with_metadata = ['volume', 'data']
    filenames_without_metadata = ['truth']

    scans_dirs = glob.glob(os.path.join(data_path, '*'))
    for scan_path in scans_dirs:
        for volume_filename in filenames_with_metadata:
            scan_filapath = os.path.join(scan_path, volume_filename + '.nii.gz')
            if os.path.exists(scan_filapath):
                break
            else:
                scan_filapath = os.path.join(scan_path, volume_filename + '.nii')
                if os.path.exists(scan_filapath):
                    break
        for truth_filename in filenames_without_metadata:
            truth_filepath = os.path.join(scan_path, truth_filename + '.nii.gz')
            if os.path.exists(truth_filepath):
                break
            else:
                truth_filepath = os.path.join(scan_path, truth_filename + '.nii')
                if os.path.exists(truth_filepath):
                    break

        volume, affine, header = read_nifti_vol_meta(scan_filapath)
        y_true = nib.load(truth_filepath).get_data()
        save_nifti_with_metadata(y_true, affine, header, truth_filepath)
