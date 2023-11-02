import pandas as pd


if __name__ == "__main__":
    unified_labels_path = '/home/bella/Phd/data/data_description/data_Elka/CHEO2/CHEO2_clinical_labels.xlsx'
    new_unified_labels_path = '/home/bella/Phd/data/data_description/data_Elka/CHEO2/CHEO2_clinical_labels_unified.xlsx'
    additional_labels = '/home/bella/Phd/data/data_description/data_Elka/CHEO2/clinical_labels_dafi.xlsx'
    match_column_name = 'ID'
    update_column_name = 'Body'
    update_column_name2 = 'Brain'


    unified_df = pd.read_excel(unified_labels_path)
    additional_df = pd.read_excel(additional_labels)

    additional_ids = additional_df[match_column_name].tolist()

    for id in additional_ids:
        additional_row = additional_df.loc[additional_df[match_column_name] == id]
        updated_column_val = additional_row.iloc[0][update_column_name]
        unified_df.loc[unified_df[match_column_name] == id, update_column_name] = updated_column_val

        if update_column_name2 is not None:
            updated_column_val2 = additional_row.iloc[0][update_column_name2]
            unified_df.loc[unified_df[match_column_name] == id, update_column_name2] = updated_column_val2

    unified_df.to_excel(new_unified_labels_path)
