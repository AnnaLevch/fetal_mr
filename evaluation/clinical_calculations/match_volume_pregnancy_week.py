import pandas as pd
import argparse
import math
from data_curation.helper_functions import patient_series_id_from_filepath, get_metadata_value, \
    patient_underscore_series_id_from_filepath
from utils.arguments import str2bool

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_calc_path", help="path of volume calculation csv",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to pregnancy week information csv",
                        type=str, required=True)
    parser.add_argument("--out_path", help="specifies the path of output volume calculation",
                        type=str, required=True)
    parser.add_argument("--is_normal_column", help="specifies the column inside medatada with normal vs not normal classification",
                        type=str, default=None)
    parser.add_argument("--special_ids", help="specifies whether 10,000 was added to ids",
                        type=str2bool, default=False)
    parser.add_argument("--underscore_pat_id", help="specifies whether 10,000 was added to ids",
                        type=str2bool, default=False)
    return parser.parse_args()

"""
This script matches volume calculation with pregnancy week
If normal vs. not normal classification exists, it filters out only notmal cases
"""
if __name__ == "__main__":

    opts = parse_arguments()
    volume_info = pd.read_csv(opts.volume_calc_path)
    metadata_info = pd.read_csv(opts.metadata_path)
    pregnancy_week_lst = []
    normal_lst = []
    subject_id_lst = []
    for index, row in volume_info.iterrows():
        subject_name = row['Subject']
        if len(str(subject_name))>5: #this is a full id
            if opts.underscore_pat_id is False:
                subject_id, series_id = patient_series_id_from_filepath(subject_name)
            else:
                subject_id, series_id = patient_underscore_series_id_from_filepath(subject_name)
        else:
            subject_id = subject_name
        if opts.special_ids is True:
            subject_id = int(subject_id) - 10000
        subject_id_lst.append(subject_id)
        #check if subject is normal if this information exist
        if opts.is_normal_column is not None:
            is_normal = get_metadata_value(subject_id, opts.is_normal_column, df=metadata_info)
            if is_normal!=1:
                print('Case ' + str(subject_id) + ' is not normal')
            normal_lst.append(is_normal)

        #check pregnancy week
        pregnancy_week = get_metadata_value(int(subject_id), 'GA_WEEKS', df=metadata_info)
        if (pregnancy_week is None) or (math.isnan(pregnancy_week)):
            print('no pregnancy week information for subject ' + str(subject_id))
        pregnancy_day = get_metadata_value(subject_id, 'GA_DAYS', df=metadata_info)
        if (pregnancy_day is not None) and (math.isnan(pregnancy_day) is False):
            pregnancy_week += pregnancy_day/7
        else:
            print('No days information for case ' + str(subject_id))

        pregnancy_week_lst.append(pregnancy_week)

    volume_info['pregnancy_week'] = pregnancy_week_lst
    volume_info['subject_id']=subject_id_lst
    if opts.is_normal_column is not None:
        volume_info['normal'] = normal_lst
    volume_info.to_csv(opts.out_path)