import os
import subprocess
import sys


if __name__ == "__main__":
  
 args = "--input_path /media/df4-dafna/Anna/Dana_Brain/Haste/Data/" \
        " --output_folder /media/df4-dafna/Anna/Dana_Brain/Haste/Results/" \
         " --config_dir /media/df4-dafna/Anna/Lab_studies/Body-Bella/best_networks/best_networks_01.08.2022/HASTE_brain/225/ --config2_dir /media/df4-dafna/Anna/Lab_studies/Body-Bella/best_networks/best_networks_01.08.2022/HASTE_brain/232/ --labeled False --fill_holes false --preprocess window_1_99 --preprocess2 window_1_99"
# --augment2 all --num_augment2 16" 

#--all_in_one_dir True"
 
 print('running with arguments:')
 print(args)
 subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

