import SimpleITK as sitk
import os
import sys
from data_generation.preprocess import correct_bias
import shutil

if __name__ == "__main__":
    subjects_folder = '/home/bella/Phd/data/brain/HASTE/HASTE/'
    out_folder = '/home/bella/Phd/data/brain/HASTE/HASTE_bfc/'
    all_in_one_dir = False
    vol_filename = 'volume.nii.gz'
    truth_file = 'truth.nii'

    if(all_in_one_dir == True):
        files = os.listdir(subjects_folder)
        for file in files:
            print('processing file: ' + file)
            correct_bias(os.path.join(subjects_folder,file), os.path.join(out_folder, file))
    else:
        dirs = os.listdir(subjects_folder)
        for dir in dirs:
            print('processing dir: ' + dir)
            out_dir_path = os.path.join(out_folder, dir)
            if(os.path.exists(out_dir_path)):
                continue

            os.mkdir(out_dir_path)

            correct_bias(os.path.join(subjects_folder, dir, vol_filename), os.path.join(out_dir_path, vol_filename))
            truth_path = os.path.join(subjects_folder,dir,truth_file)
            if(not os.path.exists(truth_path)):
                truth_path = os.path.join(subjects_folder,dir,'truth.nii.gz')
            shutil.copyfile(truth_path, os.path.join(out_dir_path,truth_file))