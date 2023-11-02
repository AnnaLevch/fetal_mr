import subprocess
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    cfg_pathes = ["/media/df4-dafna/Anna_for_Linux/Bella/training/config.json"]

    for cfg_path in cfg_pathes:
        print('training with ' + cfg_path)
        subprocess.call("python -m train with " + cfg_path, shell=True)