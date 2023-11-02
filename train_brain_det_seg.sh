#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load tensorflow
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' -- 

python -m training.training_multiple_networks.train_detection_segmentation
 > log_train_brain_1.txt


#echo 'finished running train 1' | mutt -a '/cs/casmip/bella_fadida/code/bella_code/log/log_train_1.txt' -s 'finished running train 1' -- bella.specktor@mail.huji.ac.il
