import subprocess
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    cfg_pathes = ["./config/config_brain/hemispheres/config_roi_all_perimeter_large_small.json"]

    for cfg_path in cfg_pathes:
        print('training with ' + cfg_path)
        subprocess.call("python -m train with " + cfg_path, shell=True)