#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load cuda/10.0
module load tensorflow/1.14.0
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' --


python3 train.py with ./config/config_feta/config_skull_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_csf_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_gm_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_lv_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_cbm_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_wm_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_sgm_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_bs_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_cbm_roi_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_sgm_roi_contour_dice.json > log_train_1.txt

wait

python3 train.py with ./config/config_feta/config_bs_roi_contour_dice.json > log_train_1.txt

