import os
import subprocess
import sys


if __name__ == "__main__":
 #os.environ["CUDA_VISIBLE_DEVICES"]="1"

 args = "--input_path S:/Anna/Test/Data/ABRAHAM_AVIYA/body/" \
        " --output_folder S:/Anna/Test/Data/ABRAHAM_AVIYA/Results/" \
        " --config_dir S:/Anna/Lab_studies/Bella/best_networks/best_networks_15.08.2023/body/TRUFI_body/1_Anna --labeled False --preprocess window_1_99  --all_in_one_dir True" 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

# --augment all --num_augment 16 

 print('running with arguments:')
 print(args)
 subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

