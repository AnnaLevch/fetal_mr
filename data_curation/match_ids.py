import pandas as pd
from data_curation.helper_functions import patient_underscore_series_id_from_filepath

if __name__ == "__main__":
    trufi2020_ids_path = '\\\\fmri-df3\\dafna\\Bella\\TRUFI2020\\TRUFI2020_normal.csv'
    mapping_path = '\\\\fmri-df3\\dafna\\Bella\\TRUFI2020\\TRUFI2020_mapping.csv'
    out_path = '\\\\fmri-df3\\dafna\\Bella\\TRUFI2020\\out_mapping.csv'

    ids_pd = pd.read_csv(trufi2020_ids_path)
    ids = ids_pd['Subject'].tolist()
    mapping_df = pd.read_csv(mapping_path)
    new_mapping = {}
    for id in ids:
        base_id, series_id = patient_underscore_series_id_from_filepath(id)#TODO: get values
        id_series = mapping_df.loc[mapping_df['DirName'].str.contains(base_id)]
        new_mapping[id] = id_series['st_index'].iloc[0]

    new_mapping_pd = pd.DataFrame(new_mapping, index=[0]).T
    new_mapping_pd.to_csv(out_path)