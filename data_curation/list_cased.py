import os
import pandas as pd
from utils.read_write_data import list_dump

if __name__ == "__main__":

    path = "/media/bella/8A1D-C0A6/Phd/data/Brain/brain_chambers/"
    csv_path = '/home/bella/Phd/data/data_description/brain/hemispheres/hemispheres.csv'
    txt_path = '/home/bella/Phd/tmp/firsta_placenta.txt'
    path_type = 'csv'

    files = os.listdir(path)

    if(path_type is 'csv'):
        df = pd.DataFrame(files)
        df.to_csv(csv_path)
    elif(path_type is 'txt'):
        list_dump(files, txt_path)