import glob
import os

if __name__ == "__main__":
    input_dir = '/home/bella/Phd/code/code_bella/log/638/output/TRUFI_qe_0/test/'
    file_name = 'prediction_unified.nii.gz'
    cases_pathes = glob.glob(os.path.join(input_dir,'*'))
    for case_path in cases_pathes:
        files = glob.glob(os.path.join(case_path,file_name))
        for file in files:
            os.remove(file)