import glob
import os
import shutil


if __name__ == "__main__":
    """
    Pick ground truth with specified annotators
    copy data to a new folder with picked ground truth
    """
    src_dir = '/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/CHEO/CHEO2/CHEO2_annotated/'
    dst_dir= '/media/bella/8A1D-C0A6/Phd/data/Body/FIESTA/CHEO/CHEO2/CHEO2_annotated_eval/'
    level1_name = 'dafi'
    level2_name = 'liat'
    dirs_path =  glob.glob(os.path.join(src_dir, '*'))
    for src_dir in dirs_path:
        basedir = os.path.basename(src_dir)
        print('Getting ground truth annotator for case'+ basedir)
        nifti_data = glob.glob(os.path.join(src_dir, '*'))
        dst_path = os.path.join(dst_dir, basedir)
        if os.path.exists(dst_path) is False:
            os.mkdir(dst_path)
        shutil.copy(os.path.join(src_dir, 'data.nii.gz'), os.path.join(dst_dir, basedir, 'data.nii.gz'))
        level2 = False
        level1 = False
        for data_path in nifti_data:
            if level2_name in data_path.lower():
                shutil.copy(data_path, os.path.join(dst_dir, basedir, 'truth.nii.gz'))
                print('found level 2')
                level2 = True
                break
        if level2 is False: #search for level 1
            for data_path in nifti_data:
                if level1_name in data_path.lower():
                    shutil.copy(data_path, os.path.join(dst_dir,basedir, 'truth.nii.gz'))
                    print('found level 1')
                    level1 = True
                    break

        if level2 is False and level1 is False:
            print('did mot found neither level2 nor level 1 annotations')





