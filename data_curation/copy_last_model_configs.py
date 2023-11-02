import glob
import os
import pandas as pd
import shutil


def get_best_network(df_metrics, dice_column, log_dir, ascending):
    df_metrics = df_metrics.sort_values(by=[dice_column], ascending=ascending)
    best_result = df_metrics.iloc[0]
    if best_result['epoch'] < 15:
        print('best result occured on an early epoch of ' + str(best_result['epoch']) + ", using second best")
        print('first best is: ' + str(best_result[dice_column]))
        best_result = df_metrics.iloc[1]
        print('second best is: ' + str(best_result[dice_column]))
    best_result_epoch = int(best_result['epoch'] + 1)
    chosen_path = glob.glob(os.path.join(log_dir, "*" + str(best_result_epoch)+"-*"))
    return chosen_path[0]



if __name__ == "__main__":
    """
    Script to copy last model  of at least 15 epochs
    """
    log_dirs_path = '/media/df4-dafna/Anna_for_Linux/Bella/TMP/all_networks'
    new_log_dirs_path = '/media/df4-dafna/Anna_for_Linux/Bella/TMP/selected_networks'
    # log_dirs_path = '/media/bella/8A1D-C0A6/tmp/cross_valid_/'
    # new_log_dirs_path = '/media/bella/8A1D-C0A6/tmp/cross_valid/'
    log_dirs = glob.glob(os.path.join(log_dirs_path, '*'))
    for log_dir in log_dirs:
        new_log_dir = os.path.join(new_log_dirs_path, os.path.basename(log_dir))
        if os.path.exists(new_log_dir) is False:
            os.mkdir(new_log_dir)
        shutil.copy(os.path.join(log_dir, 'config.json'), os.path.join(new_log_dir, 'config.json'))
        shutil.copy(os.path.join(log_dir, 'metrics.csv'), os.path.join(new_log_dir, 'metrics.csv'))
        shutil.copy(os.path.join(log_dir, 'norm_params.json'), os.path.join(new_log_dir, 'norm_params.json'))
        shutil.copy(os.path.join(log_dir, 'run.json'), os.path.join(new_log_dir, 'run.json'))
        ids_dir_path = glob.glob(os.path.join(log_dir, '*/'))[0]#we expect to have a single directory with ids
        ids_dir = os.path.basename(os.path.dirname(ids_dir_path))
        if os.path.exists(os.path.join(new_log_dir, ids_dir)) is False:
            os.mkdir(os.path.join(new_log_dir, ids_dir))
        shutil.copy(os.path.join(log_dir, ids_dir, 'training_ids.txt'),
                    os.path.join(new_log_dir, ids_dir, 'training_ids.txt'))
        shutil.copy(os.path.join(log_dir, ids_dir, 'validation_ids.txt'),
                    os.path.join(new_log_dir, ids_dir, 'validation_ids.txt'))
        shutil.copy(os.path.join(log_dir, ids_dir, 'test_ids.txt'),
                    os.path.join(new_log_dir, ids_dir, 'test_ids.txt'))
        df_metrics = pd.read_csv(os.path.join(log_dir, 'metrics.csv'))
        if set(['val_dice_coefficient']).issubset(df_metrics.columns):
            best_network_path = get_best_network(df_metrics, 'val_dice_coefficient', log_dir, ascending = False)
        else:
            best_network_path = get_best_network(df_metrics, 'val_loss', log_dir, ascending=True)
        print('best network path is: ' + best_network_path)
        shutil.copy(best_network_path, os.path.join(new_log_dir, os.path.basename(best_network_path)))