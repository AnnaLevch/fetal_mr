from utils.read_write_data import list_load
import os
import pandas as pd
from data_curation.helper_functions import patient_series_id_from_filepath


if __name__ == "__main__":
    """
    Get metadata of a given Ids list
    """

    ids_lists_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_brain/hemispheres/debug_split'
    all_metadata_path = '/home/bella/Phd/data/data_description/index_all_unified.csv'
    out_metadata_path = '/home/bella/Phd/meta_calculations/contour_dice/brain_hemispheres/metadata.csv'

    train_ids = list_load(os.path.join(ids_lists_path, 'training_ids.txt'))
    valid_ids = list_load(os.path.join(ids_lists_path, 'validation_ids.txt'))
    test_ids = list_load(os.path.join(ids_lists_path, 'test_ids.txt'))
    all_ids = train_ids + valid_ids + test_ids

    metadata_df = pd.read_csv(all_metadata_path, encoding='latin')
    new_metadata_ds = pd.DataFrame(columns=['Subject', 'series_name', 'path'])

    ind = 0
    for id in all_ids:
        patient_id, series_id = patient_series_id_from_filepath(id)
        patient_series = metadata_df[(metadata_df['Subject']==int(patient_id)) & (metadata_df['Series']==int(series_id))]
        new_metadata_ds.loc[ind] = [id, patient_series['Name'], patient_series['path']]
        # new_metadata_ds.loc[ind]['series_name'] =
        # new_metadata_ds.loc[ind]['path'] =
        ind+=1

    new_metadata_ds.to_csv(out_metadata_path)