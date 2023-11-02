#!/bin/csh

source /cs/labs/josko/bella_fadida/virtualenvs/bella_env/bin/activate.csh
module load tensorflow
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' -- 

python3 -m evaluation.predict_nifti_dir --input_path /cs/labs/josko/bella_fadida/code/code_bella/data/Body/TRUFI/TRUFI_body_chosen/ --output_folder /cs/labs/josko/bella_fadida/code/code_bella/log/34/output/TRUFI/ --config_dir /cs/labs/josko/bella_fadida/code/code_bella/log/34/  --preprocess window_1_99 --all_in_one_dir True  > log_predict_1.txt
#echo 'finished running train 1' | mutt -a '/cs/casmip/bella_fadida/code/bella_code/log/log_train_1.txt' -s 'finished running train 1' -- bella.specktor@mail.huji.ac.il
