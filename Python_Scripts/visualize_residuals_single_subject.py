from math import comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# # directory-related
ROOT = Path(__file__).resolve().parent
# ALL_RESIDUALS = ROOT / "all_residuals"
EC_PLOTS = ROOT / "all_residuals" / "residual_csv" / "residual_plots"

import pandas as pd
import glob


TEST_SIDS = [
        "p006365", "p013593", "p017822", "p023339", 
        "p031284","p046429", "p058242", "p064965", "p074438", 
        "p085639", "p091881", "p093117", "p098226",
    ]

def plot_histogram_residuals_combined(residual_mean, residual_std, sid):
    # rep_residual_data.shape == (N_TARGETS, K*N_Rep, N_WINDOWS)

    # plt.hist(x=rep_residual, orientation='horizontal', bins=10)
    # rep_residual = rep_residual_data.mean(axis=(1, 2))
    # rep_residual_err = rep_residual_data.mean(axis=-2).std(ddof=1, axis=-1)
    # rep_residual = '{:f}'.format(rep_residual)
    # print(sid, (rep_residual), rep_residual_err)
    
    # rep_residual = rep_residual_data.values.reshape((25,))
    # print(residual_mean)
    timepoints = ["0.0hr", "0.5hr","1.0hr", "1.5hr","2.0hr", "2.5hr","3.0hr", "3.5hr",
                    "4.0hr","4.5hr","5.0hr","5.5hr","6.0hr", "6.5hr","7.0hr","7.5hr",
                    "8.0hr","8.5hr","9.0hr","9.5hr", "10.0hr","10.5hr","11.0hr","11.5hr","12.0hr",
                    ]
    plt.rcParams['figure.figsize'] = (12, 8)
    labels = np.arange(0, 25)
    _, ax = plt.subplots()
    ax.bar(labels, residual_mean, yerr=residual_std)
    ax.set_xlabel("Time in every 30 minutes interval", fontsize=15)
    ax.set_ylabel(f"Mean Residual Value", fontsize=15)
    ax.set_xticks(labels)
    ax.set_xticklabels(timepoints, rotation=70, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_title(f"Combined Mean Residual Distribution of subject {sid}", fontsize=20)
    # ax.legend(fontsize=12)
    plt.savefig(EC_PLOTS / f"residual_plot_err_{sid}.png", bbox_inches='tight')
    plt.close()


if __name__=="__main__":
    for sid in TEST_SIDS:

        ALL_SIDS_ECs_PATH = f"/home/mostafiz/Downloads/MIMIC-get/all_residuals/residual_csv/Residuals_mean_sd_{sid}.csv"
        combined = pd.DataFrame()
        for sid_ec_file in glob.glob(ALL_SIDS_ECs_PATH): 
            with open(sid_ec_file, "rb") as infile:
                summary_file = pd.read_csv(infile)
                combined = pd.concat([combined, summary_file], axis=0)
        data = combined.drop(combined.columns[[0, 1, 2]], axis=1).mean(axis=0)
        residual_mean = data.loc["Mean_residual_1":"Mean_residual_25"]
        residual_std = data.loc["Mean_residual_sd_1":"Mean_residual_sd_25"]
        # print(residual_mean)
    

        # # rep_residual = rep_residual_data.values.reshape((25,))
        # print(residual_mean)
        plot_histogram_residuals_combined(residual_mean, residual_std, sid)

