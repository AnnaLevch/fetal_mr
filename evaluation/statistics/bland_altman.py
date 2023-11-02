import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.colors as mcolors
from utils.arguments import str2bool
import math


def bland_altman_plot(data1, data2, ground_truth=True):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    if ground_truth is True:
         diff  = (data1 - data2)/(data1)*100
    else:
        diff = (data1 - data2)/(0.5*(data1+data2))*100                   # Proportional Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    cmap, norm = mcolors.from_levels_and_colors([-40, -10, 9.99, 15], ['red', 'steelblue', 'red'])
    plt.scatter(mean, diff, c=diff, cmap=cmap, norm=norm)
    bias = md
    min_interval = md - 1.96*sd
    max_interval = md + 1.96*sd
    repeatability = 1.96*sd*math.sqrt(2)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax=plt.gca()
    ax.set_xlim([0,3500])
    #ax.set_ylim([-10,16])#general
    ax.set_ylim([-12,12])
    plt.xlabel('mean weight (g)')
    plt.ylabel('Volume Difference Ratio (VDR) (%)')
    plt.show()
    print('bias = ' + str(bias))
    print('min_interval = ' + str(min_interval))
    print('max_interval = ' + str(max_interval))
    print('max_interval = ' + str(max_interval))
    print('repeatability coefficient = ' + str(repeatability))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_comparison_path", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--data1_column", help="specifies column name of data1",
                        type=str, required=True)
    parser.add_argument("--data2_column", help="specifies column name of data2",
                        type=str, required=True)
    parser.add_argument("--data1_truth", help="specifies whether data1 is ground truth",
                        type=str2bool, default=True)

    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_arguments()
    df = pd.read_csv(opts.volume_comparison_path, encoding ="unicode_escape")

    data1 = list(df[opts.data1_column])
    data2 = list(df[opts.data2_column])
    print('number of cases: ' + str(len(data1)))
    bland_altman_plot(data1, data2, opts.data1_truth)