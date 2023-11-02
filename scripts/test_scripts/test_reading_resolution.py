from data_curation.helper_functions import get_spacing_between_slices

if __name__ == "__main__":
    subject_folder = 'Pat11_Se05_Res0.7422_0.7422_Spac5'
    res_map_path = '/home/bella/Phd/data/index_all.csv'
    spacing = get_spacing_between_slices(subject_folder, res_map_path)