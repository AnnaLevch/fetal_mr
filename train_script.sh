#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load cuda/10.0
module load tensorflow/1.14.0
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' --

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_body/config_FIESTA_small_large_fetuses.json > log_train_1.txt
#echo 'finished running train 1' | mutt -a '/cs/casmip/bella_fadida/code/bella_code/log/log_train_1.txt' -s 'finished running train 1' -- bella.specktor@mail.huji.ac.il
