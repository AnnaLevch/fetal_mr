import pandas as pd
from pathlib import Path
import pickle
import os


if __name__ == "__main__":
    observed_anomalies_path = '/home/bella/Phd/code/code_bella/log/self_consistency/anomalies_true.csv'
    eval_dir = '/home/bella/Phd/code/code_bella/log/229/output/FIESTA_gt_errors/test/'
    anomalies_df = pd.read_csv(observed_anomalies_path, header=None)
    labeling_errors = set(anomalies_df.iloc[:,0].tolist())

    if Path(eval_dir+'/pred_scores_per_slice.pkl').exists():
        with open(eval_dir +'/pred_scores_per_slice.pkl', 'rb') as f:
            pred_scores_per_slice = pickle.load(f)

    error_slices_eval = {}
    for error_slice in labeling_errors:
        splitted = error_slice.rsplit('_',1)
        vol_id = splitted[0]
        slice_ind = int(splitted[1])
        slice_eval = {}
        try:
            slice_eval['Dice'] = pred_scores_per_slice['dice_zero'][vol_id][slice_ind]
        except:
            print('undefined dice zero for case {id} slice {slice_num}'.format(id=vol_id, slice_num=slice_ind))
            slice_eval['Dice'] = None
        try:
            slice_eval['ASSD'] = pred_scores_per_slice['assd_lits'][vol_id][slice_ind]
        except:
            print('undefined ASSD for case {id} slice {slice_num}'.format(id=vol_id, slice_num=slice_ind))
            slice_eval['ASSD'] = None
        try:
            slice_eval['Hausdorff'] = pred_scores_per_slice['hausdorff_lits'][vol_id][slice_ind]
        except:
            print('undefined Hausdorff for case {id} slice {slice_num}'.format(id=vol_id, slice_num=slice_ind))
            slice_eval['Hausdorff'] = None

        error_slices_eval[error_slice] = slice_eval

    error_slices_eval_df = pd.DataFrame.from_dict(error_slices_eval).T
    error_slices_eval_df.to_csv(os.path.join(eval_dir, 'error_slices_eval.csv'))