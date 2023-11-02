import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def update_group_data(df):
    means = np.array(df['Mean'])
    std = np.array(df['STD'])
    mins = np.array(df['Min'])
    maxes = np.array(df['Max'])
    return means, std, mins, maxes


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


def plot_bars_groups(df1, df2, df3, y_name):
    means1, std1, mins1, maxes1 = update_group_data(df1)
    means2, std2, mins2, maxes2 = update_group_data(df2)
    means3, std3, mins3, maxes3 = update_group_data(df3)
   # x_axis = np.array(df['xValues'])

    ind = np.arange(len(means1))
    width = 0.3
    plt.errorbar(ind - width/2, means1, std1, fmt='ok', lw=1, ecolor='forestgreen', elinewidth = 5, capsize=10, label="full")
    plt.errorbar(ind - width/2, means1, [means1 - mins1, maxes1 - means1], fmt='.k', ecolor='gray', lw=1, capsize=5)
    plt.errorbar(ind , means2, std2,  fmt='ok', ecolor='darkcyan', elinewidth = 5, capsize=10, label="partial \wo border slices")
    plt.errorbar(ind, means2, [means2 - mins2, maxes2 - means2],fmt='.k', ecolor='gray', lw=1, capsize=5)
    plt.errorbar(ind + width/2, means3, std3,  fmt='ok', ecolor='royalblue', elinewidth = 5, capsize=10, label="partial \w border slices")
    plt.errorbar(ind + width/2, means3, [means3 - mins3, maxes3 - means3], fmt='.k', ecolor='gray', lw=1, capsize=5)
    plt.xticks(np.arange(2), ['\wo fine tuning',  '\w fine tuning with restarts'])
    if y_name == 'ASSD':
        y_name = y_name + " (mm)"
    plt.legend(loc="lower center")
    plt.ylabel(y_name)
    # a[1] = 'change'
    # plt.xticks = a

 #    ind = np.arange(len(means1))  # the x locations for the groups
 #    width = 0.35  # the width of the bars
 #    fig, ax = plt.subplots()
 #    rects1 = ax.errorbar(ind - width/2, means1, width, yerr=std1, label='/wo fine tuning')
 #    rects2 = ax.errorbar(ind + width/2, means2, width, yerr=std1, label='/w fine tuning')
 #    ax.set_ylabel('Scores')
 #    ax.set_title('Scores by group and gender')
 #    ax.set_xticks(ind)
 #    ax.set_xticklabels(x_axis)
 #    ax.legend()
 #
 # #   plt.errorbar(x_axis, means, [means - mins, maxes - means],
 # #            fmt='.k', ecolor='gray', lw=1)
 #
 #
 #    fig.tight_layout()

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_comparison_path_group1", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--results_comparison_path_group2", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--results_comparison_path_group3", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--y_axis", help="the name of the y axis",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_arguments()
    df1 = pd.read_csv(opts.results_comparison_path_group1, encoding ="unicode_escape")
    df2 = pd.read_csv(opts.results_comparison_path_group2, encoding ="unicode_escape")
    df3 = pd.read_csv(opts.results_comparison_path_group3, encoding ="unicode_escape")

    plot_bars_groups(df1, df2, df3, opts.y_axis)