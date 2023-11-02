import argparse
import glob
from multiprocess.dummy import Pool
from tqdm import tqdm_notebook as tqdm
import math
from data_curation.helper_functions import *
from evaluation.eval_utils.eval_functions import dice, calc_overlap_measure_per_slice, nvd, volume, volume_difference_ratio
from utils.arguments import str2bool
DEFAULT_RESOLUTION=[1.56,1.56,3]

def get_ids(ids_path):
    metadata = pd.read_csv(ids_path)
    ids = set(metadata['Subject'].tolist())
    return ids


def estimate_metrics_all(path_list, metrics_3D=[dice, nvd], metrics_3D_with_scaling = [volume_difference_ratio], metrics_2D = [dice], calculations=[volume], metadata_path=""):
    print('-----------------------------------------')
    print('calculating estimated metrics')
    estimated_vol_metrics = {skey.__name__ : {} for skey in metrics_3D}
    estimated_vol_metrics_with_scaling = {skey.__name__ : {} for skey in metrics_3D_with_scaling}
    estimated_2D_metrics = {skey.__name__ : {} for skey in metrics_2D}
    estimated_calculations = {}
    for calc in calculations:
        estimated_calculations[calc.__name__ + '_median']={}
        estimated_calculations[calc.__name__ + '_min']={}
        estimated_calculations[calc.__name__ + '_max']={}
        estimated_calculations[calc.__name__ + '_std']={}
    if metadata_path is not None:
        df = pd.read_csv(metadata_path, encoding ="unicode_escape")
    else:
        df=None

    def process_sub(case_dir):
        subject_id = os.path.basename(case_dir)
        print('processing case ' + subject_id)
        mean_prediction = nib.load(os.path.join(case_dir, 'prediction.nii.gz')).get_data()
        mean_prediction, swap_axis = move_smallest_axis_to_z(mean_prediction)

        #get resolution
        if df is not None:
            resolution = get_resolution(subject_id, df=df)
            print('resolution is: ' + str(resolution))
        else:
            print('using default resolution of ' + str(DEFAULT_RESOLUTION))
            resolution = DEFAULT_RESOLUTION

        #calculate deviations from mean prediction for each one of TTA
        tta_predictions_pathes = glob.glob(os.path.join(case_dir,'tta*'))
        calc_est_volumetric = {skey.__name__ : [] for skey in calculations}
        metrics_est_volumetric = {skey.__name__ : [] for skey in metrics_3D}
        metrics_est_volumetric_with_scaling = {skey.__name__ : [] for skey in metrics_3D_with_scaling}
        metrics_est_per_slice = {skey.__name__ : [] for skey in metrics_2D}
 #       resolution = get_resolution(subject_id, df=df)

        for tta_path in tta_predictions_pathes:
            tta = nib.load(tta_path).get_data()
            tta, swap_axis = move_smallest_axis_to_z(tta)
            for calc in calculations:
                calc_est = calc(tta, resolution)
                calc_est_volumetric[calc.__name__].append(calc_est)
            for metric_3D in metrics_3D:
                vol_est_metric = metric_3D(mean_prediction, tta)
                metrics_est_volumetric[metric_3D.__name__].append(vol_est_metric)
            for metric_3D_with_scaling in metrics_3D_with_scaling:
                vol_est_metric = metric_3D_with_scaling(mean_prediction, tta, resolution)
                metrics_est_volumetric_with_scaling[metric_3D_with_scaling.__name__].append(vol_est_metric)
            for metric_2D in metrics_2D:
                slices_est_metric = calc_overlap_measure_per_slice(mean_prediction, tta, metric_2D)
                metrics_est_per_slice[metric_2D.__name__].append(slices_est_metric)
            del tta

        print('unifying estimations')
        #unify estimations
        for calc in calculations:
            estimated_calculations[calc.__name__ + '_median'][subject_id] = np.median(calc_est_volumetric[calc.__name__])
            estimated_calculations[calc.__name__ + '_min'][subject_id] = np.min(calc_est_volumetric[calc.__name__])
            estimated_calculations[calc.__name__ + '_max'][subject_id] = np.max(calc_est_volumetric[calc.__name__])
            estimated_calculations[calc.__name__ + '_std'][subject_id] = np.std(calc_est_volumetric[calc.__name__])
        for metric_3D in metrics_3D:
            median_metric_est = np.median(metrics_est_volumetric[metric_3D.__name__])
            estimated_vol_metrics[metric_3D.__name__][subject_id] = median_metric_est
        for metric_3D_with_scaling in metrics_3D_with_scaling:
            median_metric_est = np.median(metrics_est_volumetric_with_scaling[metric_3D_with_scaling.__name__])
            estimated_vol_metrics_with_scaling[metric_3D_with_scaling.__name__][subject_id] = median_metric_est
        for metric_2D in metrics_2D:
            slices_data = pd.DataFrame.from_dict(metrics_est_per_slice[metric_2D.__name__]).to_dict()
            median_dice_dict = {}
            for slice_num in slices_data:
                median_dice_dict[slice_num] = np.nanmedian(list(slices_data[slice_num].values()))
            estimated_2D_metrics[metric_2D.__name__][subject_id] = median_dice_dict

        del mean_prediction

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_sub, path_list), total=len(path_list)))

    return estimated_vol_metrics, estimated_vol_metrics_with_scaling, estimated_2D_metrics, estimated_calculations


def calc_estimated_dice(case_dir):
    mean_prediction = nib.load(os.path.join(case_dir, 'prediction.nii.gz')).get_data()
    mean_prediction, swap_axis = move_smallest_axis_to_z(mean_prediction)

    tta_predictions_pathes = glob.glob(os.path.join(case_dir,'tta*'))
    vol_est_dices = []
    slices_est_dices = []
    for tta_path in tta_predictions_pathes:
        tta = nib.load(tta_path).get_data()
        tta, swap_axis = move_smallest_axis_to_z(tta)
        vol_est_dice = dice(mean_prediction, tta)
        slices_est_dice = calc_overlap_measure_per_slice(mean_prediction, tta, dice)
        vol_est_dices.append(vol_est_dice)
        slices_est_dices.append(slices_est_dice)

    estimated_dice = np.median(vol_est_dices)
    slices_data = pd.DataFrame.from_dict(slices_est_dices).to_dict()
    median_dice_dict = {}
    for slice_num in slices_data:
        median_dice_dict[slice_num] = np.median(list(slices_data[slice_num].values()))

    return estimated_dice, median_dice_dict


def write_estimation_to_excel(excel_path, estimated_vol_metrics, estimated_vol_metrics_with_scaling, estimated_2D_metrics, estimated_calculations, cases_dir, ga_info_path, include_2d_eval):
    """
    Write metric estimations and estimated volume calculations to excel
    :param excel_path: path
    :param estimated_vol_metrics: estimated volumetric metrics
    :param estimated_2D_metrics: estimated 2D metrics
    :param estimated_calculations: estimated volumetric calculations
    :param cases_dir:
    :return:
    """
    pred_scores_vol={}
    volume_ids = estimated_vol_metrics['dice'].keys()
    for metric in estimated_vol_metrics.keys():
        pred_scores_vol[metric]=estimated_vol_metrics[metric]
    for metric in estimated_vol_metrics_with_scaling.keys():
        pred_scores_vol[metric]=estimated_vol_metrics_with_scaling[metric]
    for calc in estimated_calculations.keys():
        pred_scores_vol[calc] = estimated_calculations[calc]
    for metric_2d in estimated_2D_metrics.keys():
        metric_name = '2D_' + metric_2d
        pred_scores_vol[metric_name] = {}
        for id in volume_ids:
            pred_scores_vol[metric_name][id]=np.average(list(estimated_2D_metrics[metric_2d][id].values()))
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    #write volumetric evaluation
    df = pd.DataFrame.from_dict(pred_scores_vol)
    df = df.round(3)
    df.to_excel(writer,  sheet_name='vol_eval')

    #write volume sizes
    sizes_dict = get_vol_sizes(volume_ids, cases_dir, 'prediction.nii.gz')
 #   volumes_info = get_volumes_info(sorted_keys, metadata_path, ['resX','resY', 'SpacingBetweenSlices','FOV'])
    volumes_info={}
    scores_2D = {}
    sorted_keys = sorted(estimated_2D_metrics['dice'].keys())
    ga_df = None
    if ga_info_path is not None:
        ga_df = pd.read_csv(opts.ga_info_path, encoding ="unicode_escape")

    for id in sorted_keys:
        volumes_info[id] = {}
        volumes_info[id]['size']=sizes_dict[id]
        scores_2D[id]={}
        if ga_df is not None:
            subject_id, series_id = patient_series_id_from_filepath(id)
            pregnancy_week = get_metadata_value(subject_id, 'GA_weeks', df=ga_df)
            if (pregnancy_week is None) or (math.isnan(pregnancy_week)):
                print('no pregnancy week information for subject ' + id)
            pregnancy_day = get_metadata_value(subject_id, 'GA_DAYS', df=ga_df)
            if (pregnancy_day is not None) and (math.isnan(pregnancy_day) is False):
                pregnancy_week += pregnancy_day/7
            else:
                print('No days information for case ' + id)
            volumes_info[id]['GA'] = pregnancy_week

        for metric_2d in estimated_2D_metrics.keys():
            scores_2D[id][metric_2d] = estimated_2D_metrics[metric_2d][id]
    df_vol_info = pd.DataFrame.from_dict(volumes_info).T
    df_vol_info.to_excel(writer, sheet_name='vol_info')

    if include_2d_eval:
        for vol_id in volume_ids:
            df_2D = pd.DataFrame.from_dict(scores_2D[vol_id], orient='index').T
            df_2D = df_2D.round(3)
            sheet_name = None
            if(len(vol_id)>10):
                sheet_name = patient_series_name_from_filepath(vol_id)
            if(sheet_name is None):
                sheet_name = vol_id
            df_2D.to_excel(writer, sheet_name=sheet_name)
            num_slices = len(scores_2D[vol_id]['dice'])
            #write slice evaluation graph
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            chart = workbook.add_chart({'type': 'line'})
            #[sheetname, first_row, first_col, last_row, last_col]
            chart.add_series({
                'categories': [sheet_name, 1, 0, num_slices, 0],
                'values':     [sheet_name, 1, 1, num_slices, 1],
            })
            chart.set_x_axis({'name': 'Slice number', 'position_axis': 'on_tick'})
            chart.set_y_axis({'name': '2D dice', 'major_gridlines': {'visible': False}})

            chart.set_legend({'position': 'none'})
            chart.set_size({'x_scale': 1.5, 'y_scale': 1.5})
            worksheet.insert_chart('J2', chart)
    writer.save()


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--tta_dir", help="specifies TTA dir path",
                        type=str, required=True)
    parser.add_argument("--output_path", help="specifies output path, expected to be .xlsx format",
                        type=str, required=True)
    parser.add_argument("--ids_path", help="ids path where needed ids are specified",
                        type=str, default=None)
    parser.add_argument("--metadata_path", help="metadata path for resolution extraction",
                        type=str, default=None)
    parser.add_argument("--ga_info_path", help="metadata with gestational age information",
                        type=str, default=None)
    parser.add_argument("--include_2d_eval", help="should 2d result be saved in a tab for each case",
                        type=str2bool, required=False, default=False)
    return parser.parse_args()



if __name__ == '__main__':
    """
    This script performs unsupervised results evaluation using TTA
    """
    opts = get_arguments()

    ids = None
    if opts.ids_path is not None:
        ids = get_ids(opts.ids_path)

    dirs = glob.glob(os.path.join(opts.tta_dir,'*'))
    cases_pathes = []
    for case_dir in dirs:
        id = os.path.basename(case_dir)
        if ids is not None and id not in ids:
            print('id' + id + ' is not in list, skipping....')
            continue
        cases_pathes.append(case_dir)

    print('calculating metrics')
    estimated_vol_metrics, estimated_vol_metrics_with_scaling,  estimated_2D_metrics, estimated_calculations = \
        estimate_metrics_all(cases_pathes, metadata_path=opts.metadata_path)
    print('writing to excel')
    write_estimation_to_excel(opts.output_path, estimated_vol_metrics, estimated_vol_metrics_with_scaling,
                              estimated_2D_metrics, estimated_calculations, opts.tta_dir, opts.ga_info_path, opts.include_2d_eval)