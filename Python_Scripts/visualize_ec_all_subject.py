import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

import pandas as pd
import glob
# # directory-related
ROOT = Path(__file__).resolve().parent
# ALL_RESIDUALS = ROOT / "all_residuals"
EC_PLOTS = ROOT / "ec_plots"


def plot_ecs(data):
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.style.use('fivethirtyeight')
    data = data.rename(columns=lambda x: re.sub('Mean_target','Time_point',x))
    data = data.rename(columns={"Time_point_1": "0.0hr",
                        "Time_point_2": "0.5hr",
                        "Time_point_3": "1.0hr",
                        "Time_point_4": "1.5hr",
                        "Time_point_5": "2.0hr",
                        "Time_point_6": "2.5hr",
                        "Time_point_7": "3.0hr",
                        "Time_point_8": "3.5hr",
                        "Time_point_9": "4.0hr",
                        "Time_point_10": "4.5hr",
                        "Time_point_11": "5.0hr",
                        "Time_point_12": "5.5hr",
                        "Time_point_13": "6.0hr",
                        "Time_point_14": "6.5hr",
                        "Time_point_15": "7.0hr",
                        "Time_point_16": "7.5hr",
                        "Time_point_17": "8.0hr",
                        "Time_point_18": "8.5hr",
                        "Time_point_19": "9.0hr",
                        "Time_point_20": "9.5hr",
                        "Time_point_21": "10.0hr",
                        "Time_point_22": "10.5hr",
                        "Time_point_23": "11.0hr",
                        "Time_point_24": "11.5hr",
                        "Time_point_25": "12.0hr",
                        }
                        )
    data = data.set_index("Method").T

    ax = data.plot(linewidth=2, fontsize=12, marker='o')

    # Additional customizations
    ax.set_xlabel('Time in every 30 minutes interval', fontsize=20)
    ax.set_ylabel("EC values", fontsize=20)
    plt.suptitle(f'Combined Median error consistency across all subjects', fontsize=25)
    plt.yticks(np.arange(0, 1.05, 0.15))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.legend(fontsize=20, title_fontsize=20)
    filename = "all_median_ec_plot_combined.png"
    plt.savefig(EC_PLOTS / filename)
    plt.show()



if __name__=="__main__":
    ALL_SIDS_ECs_PATH = "/home/mostafiz/Desktop/Final_Thesis/Regression-EC-MIMIC-LSTM/all_residuals/*.csv"
    combined = pd.DataFrame()
    for sid_ec_file in glob.glob(ALL_SIDS_ECs_PATH): 
        with open(sid_ec_file, "rb") as infile:
            summary_file = pd.read_csv(infile)
            combined = pd.concat([combined, summary_file], axis=0)
    data = combined.drop(["EC_Mean", "Unnamed: 0"], axis=1).drop(combined.filter(regex="sd").columns, axis=1)
    data = data.groupby('Method', as_index=False).median() # Use mean() and change the text in the title
    plot_ecs(data)
    print(data.describe())
    print(data)
