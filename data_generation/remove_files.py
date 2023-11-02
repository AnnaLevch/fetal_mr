import glob
import os

"""
remove all files with specific name, be careful to use!
"""
if __name__ == '__main__':
    src_dir = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin/'
    filename = "distance_mask.nii"
    dirs_path =  glob.glob(os.path.join(src_dir, '*'))
    for dir in dirs_path:
        filepath = os.path.join(dir, filename)
        if(os.path.exists(filepath)):
            os.remove(filepath)
        else:
            print('file does not exist: ' + filename)