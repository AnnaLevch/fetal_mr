import subprocess
import os

if __name__ == '__main__':

    brain_structures = ['skull', 'csf', 'gm', 'wm', 'lv', 'cbm', 'sgm', 'bs']
    eval_dir = '/media/bella/8A1D-C0A6/Phd/data/FeTA/Pred_CSM_multiclass_test/eval'
    for structure in brain_structures:

        src_dir = os.path.join(os.path.join(eval_dir, structure))
        args = "--src_dir {src_dir} --volume_filename volume.nii.gz" \
                   " --metadata_path /home/bella/Phd/data/data_description/feta.csv".format(src_dir=src_dir)
        print('running with arguments:')
        print(args)
        subprocess.call("python -m evaluation.evaluate " + args, shell=True)