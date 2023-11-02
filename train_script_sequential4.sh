#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load cuda/10.0
module load tensorflow/1.14.0
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' --

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks/configs/config_roi_semi_supervised_0.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks/configs/config_roi_semi_supervised_1.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks/configs/config_roi_semi_supervised_2.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks/configs/config_roi_semi_supervised_3.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_placenta/cross_valid/student_networks/configs/config_roi_semi_supervised_4.json > log_train_1.txt