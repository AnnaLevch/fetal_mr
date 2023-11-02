import subprocess
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    cfg_pathes = ["./config/config_placenta/cross_valid/student_networks/configs_soft/config_all_semi_supervised_4.json"]

    for cfg_path in cfg_pathes:
        print('training with ' + cfg_path)
        subprocess.call("python -m train with " + cfg_path, shell=True)