import pandas as pd

def get_subject_id(dir_name):
    splitted = dir_name.split('_')
    return splitted[-2] + '_' + splitted[-1]

if __name__ == "__main__":
    metadata_path = 'C://Bella//data//NewCases2020//TRUFI_all.csv'
    new_metadata_path = 'C://Bella//data//NewCases2020//TRUFI_all_subject_extracted.csv'
    df = pd.read_csv(metadata_path, encoding ="unicode_escape")

    subjet_ids = []
    for index, row in df.iterrows():
        dir_name = row['DirName']
        subject_id = get_subject_id(dir_name)
        subjet_ids.append(subject_id)

    df['Subject'] = subjet_ids
    df.to_csv(new_metadata_path, line_terminator='\n')