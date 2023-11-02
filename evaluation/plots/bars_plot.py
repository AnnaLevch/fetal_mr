import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def update_group_data(df):
    means = np.array(df['Mean'])
    std = np.array(df['STD'])
    mins = np.array(df['Min'])
    maxes = np.array(df['Max'])
    x_values = np.array(df['x_values'])
    return means, std, mins, maxes, x_values



def plot_bars(df, y_name):
    means, std, mins, maxes, x_values = update_group_data(df)
    fig = plt.figure()
   # x_axis = np.array(df['xValues'])

    ind = np.arange(len(means))
    plt.errorbar(ind , means, std, fmt='ok', lw=1, ecolor='forestgreen', elinewidth = 5, capsize=10)
 #   plt.errorbar(ind , means, std, fmt='ok', lw=1, ecolor='royalblue', elinewidth = 5, capsize=10)
    plt.errorbar(ind, means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1, capsize=5)

    if y_name == 'ASSD':
        y_name = y_name + " (mm)"

    plt.ylabel(y_name)


    plt.xticks(range(len(x_values)), x_values, rotation=20)
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
    fig.tight_layout()
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_comparison_path", help="specifies source path for evaluation (csv format)",
                        type=str, required=True)
    parser.add_argument("--y_axis", help="the name of the y axis",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_arguments()
    df = pd.read_csv(opts.results_comparison_path, encoding ="unicode_escape")

    plot_bars(df, opts.y_axis)