from data_curation.helper_functions import patient_series_id_from_filepath

if __name__ == "__main__":
    id = 'Pat20200510_094232286_Se17_Res0.78125_0.78125_Spac2.0'
    study_id, series_id = patient_series_id_from_filepath(id)
    print('case is: '+ str(study_id))
    print('series is:\ '+ str(series_id))