import argparse
import nibabel as nib
import os
import glob
from utils.read_write_data import save_nifti


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="Source directory path",
                        type=str, required=True)
    parser.add_argument("--gt1_filename", help="filename of a volume",
                        type=str, required=True)
    parser.add_argument("--gt2_filename", help="mask filename",
                        type=str, required=True)
    parser.add_argument("--out_mask_filename", help="additional mask filename",
                        type=str, default='truth_all.nii.gz')
    return parser.parse_args()


if __name__ == '__main__':
    opts = get_arguments()

    folders = glob.glob(os.path.join(opts.data_dir, '*'))
    for folder in folders:
        y_true1 = nib.load(os.path.join(folder, opts.gt1_filename)).get_data()
        y_true2 = nib.load(os.path.join(folder, opts.gt2_filename)).get_data()
        y_true = y_true1 + y_true2
        os.remove(os.path.join(folder, 'truth_all.nii'))
        save_nifti(y_true, os.path.join(folder, opts.out_mask_filename))
