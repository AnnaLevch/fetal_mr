import pandas as pd
from data_curation.helper_functions import get_metadata_value
import os


def match_ids(ids_path, old_ids_uid_mapping, new_ids_uid_mapping, save_path):
    """
    Match ids and save mapping in save_path
    :param ids_path:
    :param old_ids_uid_mapping:
    :param new_ids_uid_mapping:
    :param save_path:
    :return:
    """
    ids_df = pd.read_csv(ids_path, encoding="utf-8")
    ids = ids_df['Subject'].to_numpy()

    mapping_df_old= pd.read_csv(old_ids_uid_mapping, encoding="latin_1")
    mapping_df_new = pd.read_csv(new_ids_uid_mapping, encoding="latin_1")
    id_uid_map={}

    for id in ids:
        if(id == '232'):
            print('stop')
        if len(id)<5 :
            try:
                row_df = mapping_df_old[(mapping_df_old['st_index'] == int(id))]
                uid = row_df.iloc[0]['uid']
            except:
                print('No mapping for series ' + id)
                continue

            id_uid_map[id]=uid
        else:
            uid_path = get_metadata_value(id, 'path',  df=mapping_df_new)
            uid = os.path.basename(uid_path)
            id_uid_map[id]=uid

    id_uid_pd = pd.DataFrame(id_uid_map, index=[0]).T
    id_uid_pd.to_excel(save_path)

def match_uids_reports(mapping_path,filtered_reports_path, save_path):
    mapping_df = pd.read_excel(mapping_path)
    reports_df = pd.read_excel(filtered_reports_path)

    unified_df = pd.merge(mapping_df, reports_df, how = 'left', left_on=0, right_on='uid')
    unified_df.to_excel(save_path)


if __name__ == "__main__":
    """
    This script first performs matching between id and series id and then extract relevant report 
    """
    ids_path = 'C:\\Bella\\data\\description\\FIESTA\\volume_calc_FIESTA.csv'
    old_ids_uid_mapping = 'C:\\Bella\\data\\description\\FIESTA\\ges_old_ids.csv'
    new_ids_uid_mapping = 'C:\\Bella\\data\\description\\Index_.csv'
    metadata_path = 'C:\\Bella\\data\\description\\metadata_netanel.xlsx'
    mapping_path = 'C:\\Bella\\data\\description\\FIESTA\\ids_mapping.xlsx'
    filtered_reports_path = 'C:\\Bella\\data\\description\\FIESTA\\FIESTA_reports.xlsx'

    if not os.path.exists(mapping_path):
        match_ids(ids_path, old_ids_uid_mapping, new_ids_uid_mapping, mapping_path)

    match_uids_reports(mapping_path,metadata_path,filtered_reports_path)

