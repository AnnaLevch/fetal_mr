import pandas as pd
import argparse
from utils.arguments import str2bool
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import math
from scipy import stats


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_comparison_path", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--data1_column", help="specifies column name of data1",
                        type=str, required=True)
    parser.add_argument("--data2_column", help="specifies column name of data2",
                        type=str, required=True)
    return parser.parse_args()


def ss_within_between_icc(data):
    """
    Calculates Sum of Squares within, between and ICC. Also calculated SS within relative to group mean
    :param data:
    :return:
    ms_w - mean within
    ms_w_ratio - mean within relative to group mean
    ms_b - mean between
    icc - Intraclass correlation
    , ms_w_ratio, ms_b, icc
    """
    ssw_sum = 0
    ssw_ratio_sum = 0
    num_data_points = 0
    group_means = {}
    group_sizes = {}
    all_data = []
    #Calculate sd_w and update collect data and group means
    for group in data:
        group_data = data[group].to_list()
        group_data = [x for x in group_data if math.isnan(x) == False]
        mean = np.mean(group_data)
        group_sum = np.sum(((group_data - mean))**2)
        group_sum_ratio = np.sum(((group_data - mean)/mean)**2)
        ssw_ratio_sum += group_sum_ratio
        ssw_sum+= group_sum
        num_data_points += len(group_data)
        group_means[group] = mean
        group_sizes[group] = len(data)
        all_data = all_data + group_data
    ms_w = ssw_sum/(num_data_points-len(data.columns))
    ms_w_ratio = ssw_ratio_sum/(num_data_points-len(data.columns))

    #calculate sd_b
    total_mean = np.mean(all_data)
    ssb_sum = 0
    for group in data:
        group_sum = np.sum((group_means[group]-total_mean)**2)
        ssb_sum += group_sum
    ms_b = ssb_sum/(len(data.columns)-1)

    icc = (ms_b-ms_w)/(ms_b+ms_w)

    return ms_w, ms_w_ratio, ms_b, icc


if __name__ == "__main__":
    """
    Calculation of repeatability coefficient, its CI95 and distribution visualization
    """
    opts = parse_arguments()
    df = pd.read_csv(opts.volume_comparison_path, encoding="unicode_escape")
    # data1 = list(df[opts.data1_column])
    # data2 = list(df[opts.data2_column])
    #print('number of cases: ' + str(len(data1)))

    ms_w, ms_w_ratio, ms_b, icc = ss_within_between_icc(df)
    repeatability = 1.96*np.sqrt(ms_w*2)
    ratio_repeatability = 1.96*np.sqrt(ms_w_ratio*2)
    print('repeatability coefficient is: ' + str(repeatability))
    print('ratio repeatability coefficient is: ' + str(ratio_repeatability))
    print('ms_b is: ' + str(ms_b))
    print('ICC is: ' + str(icc))