from scripts.run_multiple_inferences import run_inference
from scripts.run_multiple_evals import evalute_dir
import os


if __name__ == "__main__":
    log_dir_path = '/home/bella/Phd/code/code_bella/log'
    input_path = '/home/bella/Phd/data/placenta/placenta_clean/'
    metadata_dir = '/home/bella/Phd/data/data_description/index_all_unified.csv'

    detection = [738, 742, 745, 737, 741]
    segmentation = [739, 744, 748, 740, 743]

    for fold in range(0,5):
        eval_dirname = 'placenta_fold_' + str(fold)
        if detection[fold] is not None and segmentation[fold] is not None:
            run_inference(detection[fold], segmentation[fold], fold, log_dir_path, input_path, eval_dirname)
            res_dir = os.path.join(log_dir_path, str(segmentation[fold]), 'output', eval_dirname, 'test')
            evalute_dir(res_dir, metadata_dir)
