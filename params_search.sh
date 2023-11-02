#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load tensorflow
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' -- 

python -m training.params_search.params_search_ac_loss > log_params_search.txt


#echo 'finished running train 1' | mutt -a '/cs/casmip/bella_fadida/code/bella_code/log/log_train_1.txt' -s 'finished running train 1' -- bella.specktor@mail.huji.ac.il
