import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri_chart_path", help="path to GA and weight data (csv format)",
                        type=str, required=True)
    parser.add_argument("--who_ultrasound_chart", help="path to ultrasound who development curve table (csv format)",
                        type=str, required=True)
    parser.add_argument("--intergrowth_21_ultrasound_chart", help="path to ultrasound intergrowth-21 development curve table (csv format)",
                        type=str, required=True)
    parser.add_argument("--hadlock_ultrasound_chart", help="path to ultrasound hadlock development curve table (csv format)",
                        type=str, required=True)
    parser.add_argument("--fmf_ultrasound_chart", help="path to ultrasound hadlock development curve table (csv format)",
                        type=str, required=True)
    parser.add_argument("--percentile", help="percentile for comparison (percentile ratio)",
                        type=float, required=True)
    parser.add_argument("--MRI_data", help="MRI data to plot on top of the charts",
                        type=str, default=None)
    parser.add_argument("--iugr_data", help="MRI data to plot on top of the charts",
                        type=str, default=None)
    return parser.parse_args()


#def calc_points_position(MRI_data_df, line2D):



def plot_percentiles_comparison(mri_chart_df, hadlock_ultrasound_chart, who_ultrasound_chart,
                                intergrowth_21_ultrasound_chart, fmf_ultrasound_chart, percentile, MRI_data_df,
                                iugr_df):

    plt.figure(figsize=(7 * 1.5, 7))
    plt.plot(mri_chart_df.GA, mri_chart_df[str(percentile)], linestyle='-', lw=1, color='blue')

    #percentile_and_below, above = calc_points_position(MRI_data_df, line2D)
    if str(percentile) in hadlock_ultrasound_chart:
        plt.plot(hadlock_ultrasound_chart.GA, hadlock_ultrasound_chart[str(percentile)], linestyle='--', lw=1, color='green')
    else:
        print('percentile ' + str(percentile) + "is not in hadlock chart")
    if str(percentile) in who_ultrasound_chart:
        plt.plot(who_ultrasound_chart.GA, who_ultrasound_chart[str(percentile)], linestyle='--', lw=1, color='orange')
    else:
        print('percentile ' + str(percentile) + "is not in WHO chart")
    if str(percentile) in intergrowth_21_ultrasound_chart:
        plt.plot(intergrowth_21_ultrasound_chart.GA, intergrowth_21_ultrasound_chart[str(percentile)], linestyle='--', lw=1, color='lime')
    else:
        print('percentile ' + str(percentile) + "is not in intergrowth_21 chart")
    if str(percentile) in fmf_ultrasound_chart:
        plt.plot(fmf_ultrasound_chart.GA, fmf_ultrasound_chart[str(percentile)], linestyle='--', lw=1, color='grey')
    else:
        print('percentile ' + str(percentile) + "is not in fmf chart")

    if MRI_data_df is not None:
        plt.scatter(MRI_data_df.GA, MRI_data_df['weight'], alpha=0.25, color='lightslategray')

    if iugr_df is not None:
        plt.scatter(iugr_df.GA, iugr_df['weight'],color='FireBrick')

    plt.xlim((19, 37.1))
    plt.ylim((0, 3800))
    plt.xlabel('GA (weeks)', fontsize=12)
    plt.ylabel('Weight (g)', fontsize=12)
    plt.title('Percentile ' + str(percentile), fontsize=14)
    plt.show()


if __name__ == "__main__":
    opts = parse_arguments()
    mri_chart_df = pd.read_csv(opts.mri_chart_path, encoding ="unicode_escape")
    hadlock_ultrasound_chart = pd.read_csv(opts.hadlock_ultrasound_chart, encoding ="unicode_escape")
    who_ultrasound_chart = pd.read_csv(opts.who_ultrasound_chart, encoding ="unicode_escape")
    intergrowth_21_ultrasound_chart = pd.read_csv(opts.intergrowth_21_ultrasound_chart, encoding ="unicode_escape")
    fmf_ultrasound_chart = pd.read_csv(opts.fmf_ultrasound_chart, encoding ="unicode_escape")
    if opts.MRI_data is not None:
        data_df = pd.read_csv(opts.MRI_data, encoding ="unicode_escape")
    else:
        data_df = None
    if opts.iugr_data is not None:
        iugr_df = pd.read_csv(opts.iugr_data, encoding ="unicode_escape")
    else:
        iugr_df = None

    plot_percentiles_comparison(mri_chart_df, hadlock_ultrasound_chart, who_ultrasound_chart,
                                intergrowth_21_ultrasound_chart, fmf_ultrasound_chart, opts.percentile, data_df, iugr_df)