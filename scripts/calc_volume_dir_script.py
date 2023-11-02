import os
import subprocess
import sys


if __name__ == "__main__":
  
 args = "--src_dir /media/df4-dafna/Anna/Dana_Brain/Results/test --metadata /media/df4-dafna/Anna/Dana_Brain/Results/metadata.csv --out_path /media/df4-dafna/Anna/Dana_Brain/Results/volume_res.csv --mask_filename prediction_correct.nii" 

 
 print('running with arguments:')
 print(args)
 subprocess.call("python -m evaluation.clinical_calculations.calc_volume_dir " + args, shell=True)

