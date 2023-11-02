import os
from scripts.run_multiple_inferences import run_inference
import subprocess


if __name__ == "__main__":

  #  structure_networks = {'skull': [502], 'csf': [510], 'lv': [508], 'cbm': [506,509 ], 'sgm':[511,513], 'bs': [515,516]}
    structure_networks = {'placenta_FIESTA':[611,608]}
    log_dir_path = '/home/bella/Phd/code/code_bella/log'
    data_path = '/home/bella/Phd/data/placenta/placenta_clean/'

    for structure in structure_networks:
         networks = structure_networks[structure]
    #     input_path = os.path.join(data_path, 'feta_' + structure + '/')
         input_path = data_path

         if len(networks) == 1:
            run_inference(None, networks[0], '4', log_dir_path, input_path, structure)
         else:#len==2
            run_inference(networks[0], networks[1], '4', log_dir_path, input_path, structure)

         if len(networks) == 1:
            src_dir = os.path.join(log_dir_path, str(networks[0]), 'output',structure, 'test/')
         else:
            src_dir = os.path.join(log_dir_path, str(networks[1]), 'output',structure, 'test/')
         args = "--src_dir {src_dir}" \
                   " --metadata_path /home/bella/Phd/data/data_description/feta.csv".format(src_dir=src_dir)
         print('running with arguments:')
         print(args)
         subprocess.call("python -m evaluation.evaluate " + args, shell=True)

