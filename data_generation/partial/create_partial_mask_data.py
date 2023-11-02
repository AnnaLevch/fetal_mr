import os
from utils.read_write_data import list_load, save_nifti


if __name__ == '__main__':
    """
    Create partial masks from whole data
    Can be used to train a network with partial annotation as mask input in addition to scan input
    """
    data_path = '/home/bella/Phd/data/body/FIESTA/FIESTA_origin/'
    data_ids_list = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_body/partial_6/0/comparison/debug_split/'
    num_partial = 10

    whole_training = list_load(os.path.join(data_ids_list, 'training_ids.txt'))
