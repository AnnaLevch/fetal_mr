import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import argparse
from utils.arguments import str2bool


def plot_quantile_curve(df, df_iugr, df_ultrasound_curve, plot_points=True, table_save_path = None):
    mod = smf.quantreg('weight ~ I(GA**3.0)', df)
    quantiles = [0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
 #   quantiles = [0.03, 0.1, 0.5, 0.9, 0.97]
    res_all = [mod.fit(q=q) for q in quantiles]
    plt.figure(figsize=(7 * 1.5, 7))

    #plot ultrasound curve if it is not None
    if df_ultrasound_curve is not None:
        for quantile in quantiles:
            plt.plot(df_ultrasound_curve.GA, df_ultrasound_curve[str(quantile)], linestyle='--', lw=1, color='Gold')

    x_p = np.linspace(df.GA.min(), df.GA.max(), 100)
    out_x_p = np.linspace(19, 37, 19)
    y_p = {}
    for qm, res in zip(quantiles, res_all):
        print('qm is: ' + str(qm))
        print(res.summary())
        y_p[qm] = res.predict({'GA': out_x_p})
        plt.plot(x_p, res.predict({'GA': x_p}), linestyle='-', lw=1, color='grey')
    if plot_points is True:
        plt.scatter(df.GA, df['weight'], alpha=0.25)
    #plot IUGR points if they are not None
    if df_iugr is not None:
        plt.scatter(df_iugr.GA, df_iugr['weight'],color='FireBrick')


    plt.xlim((18.1, 39.1))
    plt.ylim((0, 4500))
    plt.xlabel('GA (weeks)', fontsize=12)
    plt.ylabel('Weight (g)', fontsize=12)
   # plt.title(title, fontsize=14)
    plt.show()

    if table_save_path is not None:
        pred_df = pd.DataFrame.from_dict(y_p)
        pred_df.to_csv(table_save_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to GA and weight data (csv format)",
                        type=str, required=True)
    parser.add_argument("--iugr_path", help="path to IUGR fetuses to be marked in red (csv format)",
                        type=str, default=None)
    parser.add_argument("--ultrasound_curve_path", help="path to utrasound development curve table (csv format)",
                        type=str, default=None)
    parser.add_argument("--plot_points", help="Should MRI points be plotted ",
                        type=str2bool, default=True)
    parser.add_argument("--table_save_path", help="Should MRI points be plotted ",
                        type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_arguments()
    iugr_df = None
    ultrasound_curve_df = None

    growth_curve_df = pd.read_csv(opts.data_path, encoding ="unicode_escape")
    if opts.iugr_path is not None:
        iugr_df = pd.read_csv(opts.iugr_path, encoding ="unicode_escape")
    if opts.ultrasound_curve_path is not None:
        ultrasound_curve_df = pd.read_csv(opts.ultrasound_curve_path, encoding ="unicode_escape")


    plot_quantile_curve(growth_curve_df, iugr_df, ultrasound_curve_df, opts.plot_points, opts.table_save_path)