import os
import numpy as np
import random
from utils.read_write_data import list_dump
import argparse
from utils.arguments import str2bool


def get_split(percentage, indices):
    num_samples = len(indices)
    num_train = int(np.round(num_samples)*percentage)
    train_samples = indices[0:num_train]
    valid_samples = indices[num_train:num_samples]
    return [train_samples,valid_samples]


def get_cross_valid_splits(k, num_samples):
    indices = random.sample(range(0,num_samples),num_samples)
    cross_valid_splits = []
    start_ind=0
    num_test_lower = int(num_samples/k)
    num_ramain = num_samples%k

    for split in range(0,k):
        if(num_ramain!=0):
            n_test = num_test_lower + 1
        else:
            n_test = num_test_lower
        test_indices = indices[start_ind:start_ind + n_test]
        train_indices = indices[0:start_ind] + indices[start_ind + n_test:num_samples]
        cross_valid_splits.append([train_indices,test_indices])
        start_ind += n_test
    return cross_valid_splits


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", help="path to data folder",
                        type=str, required=True)
    parser.add_argument("--splits_path", help="path to splits directories",
                        type=str, required=True)
    parser.add_argument("--k", help="number of folds in k-fold cross validation",
                        type=int, default=5)
    parser.add_argument("--large_train", help="large training set? If true, training set is large and test set is small. "
                                              "If false training set is small and test is large",
                        type=str2bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    """
    This script generates k-fold cross validation data for a folder
    """
    valid_percent = 1
    opts = get_arguments()

    scans_path = os.listdir(opts.folder_path)
    cross_valid_indices = get_cross_valid_splits(opts.k, len(scans_path))

    for i in range(0,opts.k):
        [train,test] = cross_valid_indices[i]
        [train,valid] = get_split(valid_percent, train)
        training_list = [scans_path[i] for i in train]
        validation_list = [scans_path[i] for i in valid]
        test_list = [scans_path[i] for i in test]
        if os.path.exists(os.path.join(opts.splits_path, str(i))) is False:
            os.mkdir(os.path.join(opts.splits_path, str(i)))

        list_dump(validation_list, os.path.join(opts.splits_path, str(i), 'validation_ids.txt'))
        if opts.large_train is True:
            list_dump(training_list, os.path.join(opts.splits_path, str(i), 'training_ids.txt'))
            list_dump(test_list, os.path.join(opts.splits_path, str(i), 'test_ids.txt'))
        else:
            list_dump(training_list, os.path.join(opts.splits_path, str(i), 'test_ids.txt'))
            list_dump(test_list, os.path.join(opts.splits_path, str(i), 'training_ids.txt'))

