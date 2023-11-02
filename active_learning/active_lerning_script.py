from active_learning.active_learning_uniqueness import ActiveLearningUniqueness
import os
from utils.read_write_data import list_load
import glob
import json
import pandas as pd


if __name__ == "__main__":
    """
    This script tests uniqueness of a sample compared to training set
    """
    config_dir = '/home/bella/Phd/code/code_bella/log/525/'
    training_data_path = '/home/bella/Phd/data/placenta/placenta_clean'
    training_ids_list = '/home/bella/Phd/code/code_bella/log/525/0/training_ids.txt'
    unsupervised_cases_path = '/home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/'
    csv_path = '/home/bella/Phd/code/code_bella/log/531/output/placenta_FIESTA_unsupervised_cases/uniqueness.csv'

    training_scans_list = list_load(training_ids_list)
    training_samples_pathes = []
    for training_dir in training_scans_list:
        training_samples_pathes.append(os.path.join(training_data_path, training_dir))
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    unsupervised_cases_pathes = glob.glob(os.path.join(unsupervised_cases_path, '*'))
    active_learning = ActiveLearningUniqueness(config_dir)
    samples_uniqueness = active_learning.calc_samples_uniqueness(training_samples_pathes, unsupervised_cases_pathes,
                                                                 config)
    pd_uniqueness = pd.DataFrame(samples_uniqueness, index=[0]).T
    pd_uniqueness.to_csv(csv_path)