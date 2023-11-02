import json
import os
import shutil
import pandas as pd


if __name__ == '__main__':
    comparison_path = '\\\\10.101.119.14\\Dafna\\Bella\\meta_calculations\\brain_large\\'
    cfg_filename = 'brain_fine_tuned.json'
    log_dir = '\\\\10.101.119.14\\Dafna\\Bella\\tmp\\brain_networks\\'
    relative_test_folder = '\\output\\brain_large\\test\\'

    with open(os.path.join(comparison_path, cfg_filename)) as json_file:
        comparisons = json.load(json_file)

    # for dir_name in comparisons.keys():
    #     eval_path = log_dir + dir_name + relative_test_folder +'eval_truth_prediction.xlsx'
    #     shutil.copy(eval_path, os.path.join(comparison_path, 'eval_' + dir_name + '.xlsx'))

    comparison_df = None
    for dir_name in comparisons.keys():
        eval_path_csv = os.path.join(comparison_path, 'eval_' + dir_name + '.csv')
        dir_df = pd.read_csv(eval_path_csv)
        if comparison_df is None:
            comparison_df = pd.DataFrame(columns=dir_df.columns)
        dir_df.to_csv(os.path.join(comparison_path, 'test.csv'))
        avg_row = dir_df[dir_df.iloc[:,0]=='AVERAGE']
        min_row = dir_df[dir_df.iloc[:,0]=='MIN']
        max_row = dir_df[dir_df.iloc[:,0]=='MAX']
        std_row = dir_df[dir_df.iloc[:,0]=='STDEV.P']
        avg_row.iloc[0, 0] = 'AVERAGE ' + comparisons[dir_name]
        min_row.iloc[0, 0] = 'MIN ' + comparisons[dir_name]
        max_row.iloc[0, 0] = 'MAX ' + comparisons[dir_name]
        std_row.iloc[0, 0] = 'STDEV.P ' + comparisons[dir_name]
        comparison_df = comparison_df.append(avg_row, ignore_index=True)
        comparison_df = comparison_df.append(min_row, ignore_index=True)
        comparison_df = comparison_df.append(max_row, ignore_index=True)
        comparison_df = comparison_df.append(std_row, ignore_index=True)

    comparison_name = os.path.basename(comparison_path)
    comparison_df.to_csv(os.path.join(comparison_path, 'comparison_' + comparison_name + '.csv'))
