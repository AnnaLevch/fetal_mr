import pandas as pd

if __name__ == "__main__":
    metadata_path = '/home/bella/Phd/data/data_description/TRUFI_Body/AllGeom2020.csv'
    updated_metadata_path = '/home/bella/Phd/data/data_description/TRUFI_Body/AllGeom2020.csv'
    df = pd.read_csv(metadata_path, encoding ="unicode_escape")
    dir_names = df['DirName'].tolist()

    subjects = []
    for dir_name in dir_names:
        splitted = dir_name.split('_')
        subject_id = splitted[-2] + '_' + splitted[-1]
        subjects.append(subject_id)

    df['Subject'] = subjects
    df.to_csv(updated_metadata_path)
