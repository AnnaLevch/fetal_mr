import pandas as pd
import argparse
from data_curation.helper_functions import patient_series_id_from_filepath, patient_underscore_series_id_from_filepath

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", help="path to a metadata file where resolutions are specified",
                        type=str, required=True)
    parser.add_argument("--new_metadata_path", help="specifies the updated metadata path",
                        type=str, required=True)
    return parser.parse_args()



if __name__ == '__main__':
    opts = parse_arguments()
    df = pd.read_csv(opts.metadata_path, encoding ="unicode_escape")

    df['Subject'] = ""
    df = df.reset_index()
    for index, row in df.iterrows():
        scan_id = row['scan_id']
        subject_id = None
        try:
            subject_id, series_id = patient_series_id_from_filepath(scan_id)
        except:
            print('regular subject id and series id cannot be extracted, trying with underscore id')
        if subject_id is None:
            try:
                subject_id, series_id = patient_underscore_series_id_from_filepath(scan_id)
            except:
                print('subject id and series id cannot be extracted! Using subject id as scan id ')
        if subject_id is None:
            subject_id = scan_id
        df.loc[df['scan_id']==scan_id,'Subject'] = subject_id

    df.to_csv(opts.new_metadata_path)