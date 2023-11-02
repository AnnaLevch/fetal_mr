import subprocess
import sys

#sys.path.append('/home/bella/anaconda2/envs/keras_tensorflow/lib/python3.6')
#runs = [92, 335]

def evalute_dir(dir_path, metadata):
    args = "--src_dir {dir_path} --metadata_path {metadata}".format(dir_path=dir_path, metadata=metadata)
    print('evaluation with arguments:')
    print(args)
    subprocess.call("python -m evaluation.evaluate " + args, shell=True)


if __name__ == "__main__":
  #  runs = [1032]
    runs = [1102, 1103]
    is_body = True

    for i in range(0,len(runs)):
        if(is_body):
            args = "--src_dir \\\\10.101.119.14\\Dafna\\Bella\\tmp\\brain_networks\\{run}\\output\\brain_large\\test\\" \
             "  --metadata_path C:\\Bella\\data\\description\\index.csv".format(run=runs[i])

         #   " --metadata_path /home/bella/Phd/data/data_description/placenta/TRUFI_placenta/TRUFI_placenta_metadata.csv "
               #    " --metadata_path /home/bella/Phd/data/data_description/FIESTA/data_Elka/CHEO2/geometric_cheo2.csv " \


            # args = "--src_dir /home/bella/Phd/code/code_bella/log/{run}/output/TRUFI_qe_42/test/" \
            #        " --metadata_path /home/bella/Phd/data/data_description/index_all_unified.csv".format(run=runs[i])
        else: #Brain
             args = "--src_dir /home/bella/Phd/code/code_bella/log/331/output/FIESTA_diff/test/ --metadata_path /home/bella/Phd/data/data_description/index_all.csv " \
                    "".format(run=runs[i])

        print('running with arguments:')
        print(args)
        subprocess.call("python -m evaluation.evaluate " + args, shell=True)