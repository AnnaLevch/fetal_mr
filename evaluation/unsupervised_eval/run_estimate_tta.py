from scripts.run_multiple_inferences import run_inference_tta, run_inference_tta_unlabeled
import os
from transferability.self_training import SelfTraining


if __name__ == '__main__':
    """
    Run TTA inference and estimate Dice
    """
    log_dir_path = '/home/bella/Phd/code/code_bella/log/'
    detection_dir = None
    segmentation_dir = 691
    input_path = '/home/bella/Phd/code/code_bella/log/691/output/FIESTA_all_TTA/'
    out_dirname = 'quality_eval'
    metadata_path = '/home/bella/Phd/data/data_description/body_article_cases_info.csv'

    run_inference_tta_unlabeled(detection_dir, segmentation_dir, log_dir_path, input_path, out_dirname, 'True',
                                z_autoscale=True, xy_autoscale=True, metadata_path=metadata_path, num_augment=16,
                                save_soft=False)
    tta_dir = os.path.join(log_dir_path, str(segmentation_dir), 'output', out_dirname)
    dice_estimation_path = os.path.join(tta_dir, 'test', 'tta_quality_estimation.xlsx')
    st = SelfTraining(os.path.join(tta_dir, 'test'))
    st.run_eval_estimation_tta(dice_estimation_path, metadata_path)
    st.clean_tta_results()