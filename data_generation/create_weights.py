from utils.read_write_data import list_load, list_dump
import os


if __name__ == "__main__":
    all_list_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/debug_split/training_ids.txt'
    split_dir = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/partial/4/debug_split_6/'
    other_cases_weight = 0.5

    all_cases = list_load(all_list_path)
    whole_cases = list_load(os.path.join(split_dir, 'training_ids.txt'))
    weights = []
    for case in all_cases:
        if case in whole_cases:
            weights.append(1)
        else:
            weights.append(other_cases_weight)

    list_dump(weights, os.path.join(split_dir, 'weights.txt'))