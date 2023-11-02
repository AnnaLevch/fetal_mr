import pandas as pd
import os
from shutil import copyfile


def extract_from_one_folder(dir_names, data_path):
    for filename in dir_names:
        splitted_path = os.path.split(filename)
        filename = splitted_path[-1]
        search_path = os.path.join(data_path, filename)
        if(os.path.isfile(search_path)):
            out_path = os.path.join(outdir, splitted_path[-1])
            copyfile(search_path, out_path)

def extract_from_directories(dir_names, data_path):
    for filename in dir_names:
        splitted_path = os.path.split(filename)
        filename = splitted_path[-1]
        dirname = os.path.split(splitted_path[0])[-1]
        search_path = os.path.join(data_path, dirname, filename)
        if(os.path.isfile(search_path)):
            out_path = os.path.join(outdir, splitted_path[-1])
            copyfile(search_path, out_path)

if __name__ == "__main__":
    csv_path = '/home/bella/Phd/data/data_description/TRUFI_body_above60slices.csv'
    data_path = '/media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/TRUFI_Body_unlabeled'
    outdir = '/media/bella/8A1D-C0A6/Phd/data/Body/TRUFI/TRUFI_body_above60slices/'
    all_in_one = True
    df = pd.read_csv(csv_path)
    dir_names = df['Filename'].tolist()


    if all_in_one is True:
       extract_from_one_folder(dir_names, data_path)
    else:
        extract_from_directories(dir_names, data_path)
