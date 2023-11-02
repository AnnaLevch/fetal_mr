import argparse
import pickle
from glob import glob
from pathlib import Path
import string
from astropy.stats.sigma_clipping import sigma_clipped_stats
from multiprocess.dummy import Pool
from scipy.stats import gmean
from tqdm import tqdm_notebook as tqdm
from utils.arguments import str2bool
from evaluation.eval_utils.eval_functions import *
from evaluation.eval_utils.key_images import *
from evaluation.eval_utils.postprocess import postprocess_prediction
from data_curation.helper_functions import *
from utils.str_manipulations import get_basic_name
from evaluation.eval_utils.utils import *

scaling = (1.56,1.56,3)#resolution for FIESTA dataset
#scaling = (0.78125, 0.78125,2) #TRUFI 2020 resolution
#scaling = (1,1,1)

def mean(data):
    return data.mean(axis=0)
def meadian(data):
    return np.median(data, axis=0)

def rob_mean(data):
    return sigma_clipped_stats(data, axis=0, sigma=3, maxiters=1)[0]

def rob_median(data):
    return sigma_clipped_stats(data, axis=0, sigma=3, maxiters=1)[1]

def geometric_mean(data):
    return gmean(data, axis=0)

def postprocess(data):
    return postprocess_prediction(data, threshold=0.5,
                                  fill_holes=False, connected_component=True)


def evaluate_all(path_list, truth_filename, result_filename, name='', scan_series_id_from_name=True,
                 metrics_without_rescaling=[dice, vod, surface_dice],
                 without_rescaling_2D=[dice, surface_dice],
                 metrics_with_rescaling_2D=[hausdorff_lits, assd_lits, hausdorff_robust_lits],
                 metrics_with_rescaling=[hausdorff_lits, assd_lits, hausdorff_robust_lits, hausdorff, hausdorff_robust, assd, volume_difference, volume_difference_ratio],
                 metadata_path='', in_plane_res_name='PixelSpacing'):
    print(name)
    print('-----------------------------------------')
    print('-----------------------------------------')
    metrics = metrics_without_rescaling + metrics_with_rescaling
    metrics_2D = without_rescaling_2D + metrics_with_rescaling_2D
    for aggr_method in [meadian]: #, mean
        print('-----------------------------------------')
        print(aggr_method.__name__, flush=True)
        pred_scores_per_slice = {skey.__name__ : {} for skey in metrics_2D}
        pred_scores_per_slice['dice_zero'] = {}
        pred_scores_vol = {skey.__name__ : {} for skey in metrics}
        df = pd.read_csv(metadata_path, encoding ="unicode_escape")

        def process_sub(subject_folder):
            subject_id = os.path.basename(os.path.dirname(subject_folder))
            print('loading case: ' + subject_id)
            truth = np.int16(nib.load(os.path.join(subject_folder, truth_filename)).get_data())
            pred = np.int16(nib.load(os.path.join(subject_folder, result_filename)).get_data())

            #make sure smaller axis is z to ensure correct calculation of 2D contour
            truth, swap_axis = move_smallest_axis_to_z(truth)
            pred, swap_axis = move_smallest_axis_to_z(pred)

            resolution = get_resolution(subject_id, df=df, extract_scan_series_id=scan_series_id_from_name, in_plane_res_name=in_plane_res_name)
            if resolution is None:#if resolution cannot be retrived from name and from metadata, use constantot
                print('resolution is not found in name nor in metadata! using constant resolution of ' + str(scaling))
                resolution = scaling
            print('resolution is: ' + str(resolution))

            if(truth.shape != pred.shape):
                print("in case + " + subject_folder + " there is a mismatch")

            try:
                #calculate measures
                for score_method in metrics_without_rescaling:
                    pred_scores_vol[score_method.__name__][subject_id] = score_method(truth, pred)
                for score_method in without_rescaling_2D:
                    if(score_method.__name__ == 'dice'):
                        pred_scores_per_slice['dice_zero'][subject_id] = calc_overlap_measure_per_slice(truth, pred, score_method)
                    pred_scores_per_slice[score_method.__name__][subject_id] = calc_overlap_measure_per_slice_no_zero_pixels(truth, pred, score_method)
                for score_method in metrics_with_rescaling:
                    pred_scores_vol[score_method.__name__][subject_id] = score_method(truth, pred, resolution)
                for score_method in metrics_with_rescaling_2D:
                    pred_scores_per_slice[score_method.__name__][subject_id] = calc_distance_measure_per_slice(truth, pred, resolution, score_method)
            except Exception as e:
                print('exception occurred when calculating 2D metrics in case: ' + subject_folder + '!')
                print(e)
                return None

            del pred
            del truth

            #can give to Pool() less workers as a parameter
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(process_sub, path_list), total=len(path_list)))

        print('\t\t volumetric measures')
        for score_method in metrics:
            score_key = score_method.__name__
            print('{}\t - {:.3f} ±({:.3f}))'.format(score_key,
                                                np.mean(list(pred_scores_vol[score_key].values())),
                                                np.std(list(pred_scores_vol[score_key].values()))))
        return pred_scores_per_slice, pred_scores_vol


def save_to_csv(pred_scores_vol, path):
    pred_df = pd.DataFrame.from_dict(pred_scores_vol)
    pred_df.to_csv(path)


def path_to_name(path):
    basename = os.path.basename(path)
    basename = os.path.splitext(basename)[0]
    splitted = basename.split('_')
    title = 'slice ' + splitted[-3] + ': ' + splitted[-2] + '=' + splitted[-1]
    return title

def write_formula_row(df, row_ind, worksheet, formula_name):
    num_columns = df.shape[1]
    num_samples = df.shape[0]
    column_names = list(string.ascii_uppercase)
    worksheet.write('A' + str(row_ind), formula_name)
    for i in range(1, num_columns + 1):
        formula_cell = column_names[i] + str(row_ind)
        start_range = column_names[i] + '2'
        end_range = column_names[i] + str(num_samples + 1)
        formula = '=' + formula_name + '(' + start_range + ':' + end_range + ')'
        worksheet.write_formula(formula_cell, formula)


def write_to_excel(pred_scores_vol, pred_scores_per_slice, excel_path, eval_folder, out_folder, vol_filename,
                   gt_filename, result_filename, metadata_path, num_key_images=4, thresh_value=0.92):

    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    sorted_keys = sorted(pred_scores_per_slice['dice'].keys())
    pred_scores_vol['dice_2D_avg']={}
    pred_scores_vol['dice_2D_zero_avg']={}
    pred_scores_vol['hausdorff_2D_avg'] = {}
    pred_scores_vol['hausdorff_2D_max'] = {}
    pred_scores_vol['hausdorff_robust_2D_avg'] = {}
    pred_scores_vol['hausdorff_robust_2D_max'] = {}
    pred_scores_vol['assd_2D_avg'] = {}
    pred_scores_vol['assd_2D_max'] = {}
    pred_scores_vol['surface_dice_2D_avg'] = {}
    pred_scores_per_slice_by_id = {}
    for id in sorted_keys:
        pred_scores_per_slice_by_id[id]={}
        pred_scores_vol['dice_2D_avg'][id] = list_avg(list(pred_scores_per_slice['dice'][id].values()))
        pred_scores_vol['dice_2D_zero_avg'][id] = list_avg(list(pred_scores_per_slice['dice_zero'][id].values()))
        if id in pred_scores_per_slice['hausdorff_lits']:
            pred_scores_vol['hausdorff_2D_avg'][id] = list_avg(list(pred_scores_per_slice['hausdorff_lits'][id].values()))
            pred_scores_vol['hausdorff_2D_max'][id] = list_max(list(pred_scores_per_slice['hausdorff_lits'][id].values()))
        if id in pred_scores_per_slice['hausdorff_robust_lits']:
            pred_scores_vol['hausdorff_robust_2D_avg'][id] = list_avg(list(pred_scores_per_slice['hausdorff_robust_lits'][id].values()))
            pred_scores_vol['hausdorff_robust_2D_max'][id] = list_max(list(pred_scores_per_slice['hausdorff_robust_lits'][id].values()))
        if id in pred_scores_per_slice['assd_lits']:
            pred_scores_vol['assd_2D_avg'][id] = list_avg(list(pred_scores_per_slice['assd_lits'][id].values()))
            pred_scores_vol['assd_2D_max'][id] = list_max(list(pred_scores_per_slice['assd_lits'][id].values()))
        pred_scores_vol['surface_dice_2D_avg'][id] = list_avg(list(pred_scores_per_slice['surface_dice'][id].values()))
        pred_scores_per_slice_by_id[id]['dice_2D'] = pred_scores_per_slice['dice'][id]
        if id in pred_scores_per_slice['hausdorff_lits']:
            pred_scores_per_slice_by_id[id]['hausdorff_2D'] = pred_scores_per_slice['hausdorff_lits'][id]
        if id in pred_scores_per_slice['hausdorff_robust_lits']:
            pred_scores_per_slice_by_id[id]['hausdorff_robust_2D'] = pred_scores_per_slice['hausdorff_robust_lits'][id]
        if id in pred_scores_per_slice['assd_lits']:
            pred_scores_per_slice_by_id[id]['assd_2D'] = pred_scores_per_slice['assd_lits'][id]
        pred_scores_per_slice_by_id[id]['surface_dice_2D'] = pred_scores_per_slice['surface_dice'][id]

    #write volumetric evaluation
    df = pd.DataFrame.from_dict(pred_scores_vol)
    df = df.round(3)
    df.to_excel(writer,  sheet_name='vol_eval')
    #write volumetric formulas
    worksheet = writer.sheets['vol_eval']
    write_formula_row(df, df.shape[0] + 2, worksheet, 'AVERAGE')
    write_formula_row(df, df.shape[0] + 3, worksheet, 'MIN')
    write_formula_row(df, df.shape[0] + 4, worksheet, 'MAX')
    write_formula_row(df, df.shape[0] + 5, worksheet, 'STDEV.P')

    #write volume sizes
    sizes_dict = get_vol_sizes(sorted_keys, eval_folder, gt_filename)
    volumes_info = get_volumes_info(sorted_keys, metadata_path, ['PixelSpacing', 'SpacingBetweenSlices','FOV'])
    for id in sorted_keys:
        volumes_info[id]['size']=sizes_dict[id]
    df_vol_info = pd.DataFrame.from_dict(volumes_info).T
    df_vol_info.to_excel(writer, sheet_name='vol_info')

    #write 2D evaluations in different tab for each volume
    for vol_id in sorted_keys:

        #write slice evaluation data
        vol_slice_eval = pred_scores_per_slice_by_id[vol_id]
        df_2D = pd.DataFrame.from_dict(vol_slice_eval, orient='index').T
        df_2D = df_2D.round(3)

        sheet_name = None
        if(len(vol_id)>10):
            try:
                sheet_name = patient_series_name_from_filepath(vol_id)
            except:
                sheet_name = vol_id

        if(sheet_name is None):
            sheet_name = vol_id
        df_2D.to_excel(writer, sheet_name=sheet_name)

        #write slice evaluation graph
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        chart = workbook.add_chart({'type': 'line'})

        num_slices = len(vol_slice_eval['dice_2D'])

        #[sheetname, first_row, first_col, last_row, last_col]
        chart.add_series({
            'categories': [sheet_name, 1, 0, num_slices, 0],
            'values':     [sheet_name, 1, 1, num_slices, 1],
        })
        chart.add_series({
            'categories': [sheet_name, 1, 0, num_slices, 0],
            'values':     [sheet_name, 1, 2, num_slices, 2],
            'y2_axis': 2,
        })
        chart.set_x_axis({'name': 'Slice number', 'position_axis': 'on_tick'})
        chart.set_y_axis({'name': '2D dice', 'major_gridlines': {'visible': False}})
        chart.set_y2_axis({'name': 'Hausdorff (mm)', 'major_gridlines': {'visible': False}})
        chart.set_legend({'position': 'none'})
        chart.set_size({'x_scale': 1.5, 'y_scale': 1.5})
        worksheet.insert_chart('J2', chart)

        #write key images
        key_images_indices_dice = get_key_slices_indexes(vol_slice_eval['dice_2D'], num_key_images, thresh_value)
        if 'hausdorff_2D' in vol_slice_eval:
            key_images_indices_hausdorff = get_key_slices_indexes_largest(vol_slice_eval['hausdorff_2D'], 2)
        images_pathes_dice = save_key_images(key_images_indices_dice, opts.src_dir, out_folder, vol_id, vol_filename, gt_filename, result_filename, 'dice') #save images to load from in excel
        images_pathes_hausdorff = save_key_images(key_images_indices_hausdorff, opts.src_dir, out_folder, vol_id, vol_filename, gt_filename, result_filename, 'hausdorff')
        images_pathes = images_pathes_dice
        images_pathes.update(images_pathes_hausdorff)
        sorted_slices = sorted(images_pathes.keys())
        start_row = 30
        figure_hight = 18
        for slice_num in sorted_slices:
            plotname = path_to_name(images_pathes[slice_num])
            worksheet.insert_image('J' + str(start_row), images_pathes[slice_num])
            worksheet.write('K' + str(start_row-2), plotname)
            start_row = start_row + figure_hight + 1

    writer.save()


def write_eval_per_slice(eval_per_slice, save_path):
    eval_all_slices = {}

    for key in eval_per_slice:
        vol_eval = np.full(150,None, dtype=float)
        vol_dict = eval_per_slice[key]
        for slice_num in vol_dict:
          vol_eval[slice_num] = vol_dict[slice_num]
        eval_all_slices[key] = vol_eval
    pd_scores = pd.DataFrame.from_dict(eval_all_slices)
    pd_scores.to_csv(save_path)


def parse_eval_arguments():
    """
    Parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--metadata_path", help="path to scans details for correct resolution",
                        type=str, required=True)
    parser.add_argument("--num_key_imgs", help="specifies source directory for evaluation",
                        type=int, required=False, default=4)
    parser.add_argument("--dice_thresh", help="specifies source directory for evaluation",
                        type=int, required=False, default=0.92)
    parser.add_argument("--gt_filename", help="name of the gt file",
                        type=str, required=False, default='truth.nii.gz')
    parser.add_argument("--result_filename", help="name of the result file",
                        type=str, required=False, default='prediction.nii.gz')
    parser.add_argument("--volume_filename", help="name of the volume file",
                        type=str, required=False, default='data.nii.gz')
    parser.add_argument("--out_dir", help="name of the output directory, by default is the same as src_dir",
                        type=str, required=False, default=None)
    parser.add_argument("--save_to_excel", help="should results be saved to excel",
                        type=str2bool, required=False, default=True)
    parser.add_argument("--write_eval_per_slice", help="should we save slice evaluation in separate files",
                        type=str2bool, required=False, default=False)
    parser.add_argument("--scan_series_id_from_name", help="should we extract scan and series id from name to get metadata. Set to True if metadata has this values",
                        type=str2bool, required=False, default=True)
    parser.add_argument("--in_plane_res_name", help="in-plane resolution column name in metadata file",
                        type=str, required=False, default='PixelSpacing')
    return parser.parse_args()


if __name__ == '__main__':
    """
    Performs evaluation on data with evaluation functions specified in the beginning of the script
    This scripts assumes there is a directory with results in the needed format (after running predict_nifti_dir.py)
    """
#   Evaluation functions
    metrics_without_rescaling=[dice, vod, surface_dice]
    without_rescaling_2D=[dice, surface_dice]
    metrics_with_rescaling_2D=[hausdorff_lits, assd_lits, hausdorff_robust_lits]
    metrics_with_rescaling=[hausdorff_lits, assd_lits, hausdorff_robust_lits, hausdorff, hausdorff_robust, assd, volume_difference, volume_difference_ratio]

    opts = parse_eval_arguments()

    if opts.out_dir is None:
        output_dir = opts.src_dir

#    calculate evaluation metrics if they do not already exist
    if Path(output_dir+'/pred_scores_per_slice.pkl').exists() and Path(output_dir+'/pred_scores_vol.pkl').exists():
        print('scores were already calculated, loading')
        with open(output_dir +'/pred_scores_per_slice.pkl', 'rb') as f:
            pred_scores_per_slice = pickle.load(f)
        with open(output_dir +'/pred_scores_vol.pkl', 'rb') as f:
            pred_scores_vol = pickle.load(f)
    else:
        pred_scores_per_slice, pred_scores_vol = evaluate_all([_ for _ in (glob(os.path.join(opts.src_dir, '*/')))], truth_filename=opts.gt_filename,
                                                              result_filename=opts.result_filename, metadata_path=opts.metadata_path,
                                                              scan_series_id_from_name=opts.scan_series_id_from_name,
                                                              in_plane_res_name=opts.in_plane_res_name)
        print('--------------------\nsaving...')
        with open(output_dir +'/pred_scores_per_slice.pkl', 'wb') as f:
            pickle.dump(pred_scores_per_slice, f)
        with open(output_dir +'/pred_scores_vol.pkl', 'wb') as f:
            pickle.dump(pred_scores_vol, f)

#   Save evaluation to excel file. True by default
    if opts.save_to_excel is True:
        filename = 'eval_' + get_basic_name(opts.gt_filename) + '_' + get_basic_name(opts.result_filename) + '.xlsx'
        output_filename = os.path.join(output_dir, filename)
        write_to_excel(pred_scores_vol,  pred_scores_per_slice, output_filename, opts.src_dir,output_dir, opts.volume_filename,
                       opts.gt_filename,opts.result_filename, opts.metadata_path, opts.num_key_imgs, opts.dice_thresh)

    if opts.write_eval_per_slice is True:
        write_eval_per_slice(pred_scores_per_slice['hausdorff_lits'], os.path.join(opts.src_dir, 'hausdorff_per_slice.csv'))
        write_eval_per_slice(pred_scores_per_slice['assd_lits'], os.path.join(opts.src_dir ,'assd_per_slice.csv'))
        write_eval_per_slice(pred_scores_per_slice['dice_zero'], os.path.join(opts.src_dir ,'dice_zero_per_slice.csv'))
     #   write_eval_per_slice(pred_scores_per_slice['pixel_difference'], os.path.join(src_dir,'pixel_difference.csv'))