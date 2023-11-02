import os
import glob
import nibabel as nib
from utils.read_write_data import save_nifti


if __name__ == "__main__":
    """
    compress all uncompressed files with a given name
    """
    src_dir = '/home/bella/Phd/data/body/TRUFI/TRUFI'
    filename = 'truth.nii'

    dirs_path =  glob.glob(os.path.join(src_dir, '*'))
    for subject_dir in dirs_path:
        data_path = os.path.join(src_dir, subject_dir, filename)
        if os.path.exists(data_path) is True:
            data = nib.load(data_path).get_data()
            save_nifti(data, data_path + '.gz')
            os.remove(data_path)