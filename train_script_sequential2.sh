#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load cuda/10.0
module load tensorflow/1.14.0
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' --


python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_skull.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_csf.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_lv.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_cbm.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_sgm.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_bs.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_cbm_roi.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_sgm_roi.json > log_train_1.txt

wait

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_feta/config_bs_roi.json > log_train_1.txt


