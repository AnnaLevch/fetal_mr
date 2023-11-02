import argparse
import os
import json

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config_path", help="path to baseline_config",
                        type=str, required=True)
    parser.add_argument("--splits_path", help="path to splits directories",
                        type=str, required=True)
    parser.add_argument("--out_dir", help="output directory for cross validation configs",
                        type=str, required=True)
    parser.add_argument("--k", help="number of folds in k-fold cross validation",
                        type=int, default=5)

    return parser.parse_args()

if __name__ == "__main__":
    """
    Clone existing config with updated cross validation splits
    """
    opts = get_arguments()

    with open(opts.base_config_path, 'r') as f:
        config = json.load(f)

    cfg_basename = os.path.basename(os.path.splitext(opts.base_config_path)[0])

    for i in range(0,opts.k):
        config_new = config.copy()
        config_new["data_dir"] = config["data_dir"][:-1]+'_' + str(i) + '/'
        config_new["split_dir"] = os.path.join(opts.splits_path, str(i))
        config_new["training_file"] = os.path.join(config_new["split_dir"], "training_ids.txt")
        config_new["validation_file"] = os.path.join(config_new["split_dir"], "test_ids.txt")
        config_new["test_file"] = os.path.join(config_new["split_dir"], "test_ids.txt")

        config_path = os.path.join(opts.out_dir, cfg_basename + '_' + str(i) + '.json')
        with open(config_path, mode='w') as f:
            json.dump(config_new, f, indent=2)