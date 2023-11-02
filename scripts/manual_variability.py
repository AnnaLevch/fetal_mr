import subprocess
import sys

sys.path.append('/home/bella/anaconda2/envs/keras_tensorflow/lib/python3.6')

#Bella_Elka
args = "--src_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Haste_Brain_Variability/" \
       " --res_map /home/bella/Phd/data/index_all.csv --gt_filename annotation_bella_EM.nii --result_filename annotation_bella.nii.gz " \
       "--out_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Variability_Results/Bella_Elka --volume_filename data.nii"
print('running with arguments:')
print(args)
subprocess.call("python -m evaluation.evaluate " + args, shell=True)

#Dafi_Elka
args = "--src_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Haste_Brain_Variability/" \
       " --res_map /home/bella/Phd/data/index_all.csv --gt_filename annotation_bella_EM.nii --result_filename prediction_Dafi.nii " \
       "--out_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Variability_Results/Dafi_Elka --volume_filename data.nii"
print('running with arguments:')
print(args)
subprocess.call("python -m evaluation.evaluate " + args, shell=True)

# #Liat_Elka
# args = "--src_dir /media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/variability_cases_TRUFI/Trufi-v_Liat/variability_cases_TRUFI_body_bella/" \
#        " --res_map /home/bella/Phd/data/index_all.csv --gt_filename annotation_bella_EM.nii --result_filename prediction_tuvia+Dafi+Liat.nii " \
#        "--out_dir /media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/variability_cases_TRUFI/variability_results/Liat_Elka --volume_filename data.nii"
# print('running with arguments:')
# print(args)
# subprocess.call("python -m evaluation.evaluate " + args, shell=True)


#Dafi_Network
args = "--src_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Haste_Brain_Variability/" \
       " --res_map /home/bella/Phd/data/index_all.csv --gt_filename prediction_Dafi.nii --result_filename prediction.nii.gz " \
       "--out_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Variability_Results/NN_Dafi"
print('running with arguments:')
print(args)
subprocess.call("python -m evaluation.evaluate " + args, shell=True)

#Elka_Network
args = "--src_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Haste_Brain_Variability/" \
       " --res_map /home/bella/Phd/data/index_all.csv --gt_filename annotation_bella_EM.nii --result_filename prediction.nii.gz " \
       "--out_dir /media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE-v/Variability_Results/NN_Elka"
print('running with arguments:')
print(args)
subprocess.call("python -m evaluation.evaluate " + args, shell=True)