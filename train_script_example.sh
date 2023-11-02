#!/bin/csh


echo 'starting train script' | mutt -s 'starting train script' --

python3 train.py with /cs/labs/josko/bella_fadida/code/code_bella/fetal_mr/config/config_body/config_FIESTA_small_large_fetuses.json > log_train_1.txt

