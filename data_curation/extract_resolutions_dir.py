from evaluation.evaluate import get_spacing_between_slices, resolution_from_scan_name
import glob
import os
import pandas as pd

if __name__ == "__main__":
    src_dir = '/media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE_to_annotate/'
    res_map_path = '/home/bella/Phd/data/index_all.csv'
    outpath = '/media/bella/8A1D-C0A6/Phd/data/Brain/HASTE/HASTE_data/HASTE_51_cases.csv'
    default_resolution = (1.56,1.56,3)

    files = glob.glob(os.path.join(src_dir, '*'))
    filepath_res_map = {}

    for filepath in files:
        print('processing filepath ' + filepath)
        resolution = resolution_from_scan_name(filepath)
        # if(resolution == None):
        #     filepath_res_map[filepath] = default_resolution
        #     continue
        spacing = get_spacing_between_slices(os.path.basename(filepath), res_map_path)
        if(spacing!=None):
            resolution[2] = spacing

        filepath_res_map[filepath] = resolution

    df = pd.DataFrame.from_dict(filepath_res_map).T
    df.to_csv(outpath)