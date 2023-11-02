import argparse
import glob
import os
import nibabel as nib
import pandas as pd
import re
from utils.arguments import str2bool


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--output_folder", help="specifies output csv dir",
                        type=str, required=True)
    parser.add_argument("--out_filename", help="output filename",
                        type=str, default='resolutions_sizes.csv')
    parser.add_argument("--all_in_one_dir", help="whether files are all in one dir or in directories",
                        type=str2bool, default=False)

    return parser.parse_args()


def get_scan_data(filename, scan_re, all_in_one_dir, scan_re2=None):
    scan_name = os.path.basename(filename)
    res = scan_re.findall(filename)
    pat_id = None
    if res:
        pat_id, ser_num, res_x, res_y,  res_z = res[0]
    else:
        res = scan_re2.findall(filename)
        if res:
            pat_id1, pat_id2, ser_num, res_x, res_y,  res_z = res[0]
            pat_id = pat_id1 + '_' + pat_id2
        else:
            res_x = res_y = res_z = None

    base_filename = filename
    if all_in_one_dir is False:
        filename = os.path.join(base_filename, 'volume.nii.gz')
    if not os.path.exists(filename):
        filename = os.path.join(base_filename,'volume.nii')
    if not os.path.exists(filename):
        filename = os.path.join(base_filename,'data.nii.gz')
    vol = nib.load(filename).get_data()
    x_size, y_size, z_size = vol.shape


    return scan_name, pat_id, res_x, res_y, res_z, x_size, y_size, z_size


if __name__ == '__main__':
#TODO: Change to take slice spacing from metadata
    opts = get_arguments()

    filenames = glob.glob(os.path.join(opts.input_dir, '*'))

    if(opts.all_in_one_dir == True):
        scan_re = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+).nii")
    else:
        scan_re = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+)")
        scan_re2= re.compile("Pat(?P<patient_id1>[\d]+)_(?P<patient_id2>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+)")
    data = []

    for filename in filenames:
        scan_id, vol_id, res_x, res_y, res_z, x_size, y_size, z_size = get_scan_data(filename, scan_re, opts.all_in_one_dir, scan_re2)
        data.append([scan_id, vol_id, res_x, res_y, res_z, x_size, y_size, z_size])

    columns = ['scan_id','vol_id', 'x_res', 'y_res', 'z_res', 'x_size', 'y_size', 'z_size']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(opts.output_folder + opts.out_filename)






