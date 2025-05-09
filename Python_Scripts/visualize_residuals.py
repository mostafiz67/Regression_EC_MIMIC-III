# export SLURM_ARRAY_TASK_ID=1
# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
# fmt: on
import pickle
from pathlib import Path
import glob
import numpy as np
from src.models.deeplearning.validation import ValidationResults
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

import pandas as pd
import glob
PLOT_ROOT = Path(__file__).resolve().parent
ALL_RESIDUALS = PLOT_ROOT / "all_residuals"
EC_PLOTS = PLOT_ROOT / "ec_plots"





def plot_histogram_residuals(rep_residual_data, sid):
    # rep_residual_data.shape == (N_TARGETS, K*N_Rep, N_WINDOWS)

    # plt.hist(x=rep_residual, orientation='horizontal', bins=10)
    rep_residual = rep_residual_data.mean(axis=(1, 2))
    rep_residual_err = rep_residual_data.mean(axis=-2).std(ddof=1, axis=-1)
    # rep_residual = '{:f}'.format(rep_residual)
    # print(sid, (rep_residual), rep_residual_err)

    timepoints = ["0.0hr", "0.5hr","1.0hr", "1.5hr","2.0hr", "2.5hr","3.0hr", "3.5hr",
                    "4.0hr","4.5hr","5.0hr","5.5hr","6.0hr", "6.5hr","7.0hr","7.5hr",
                    "8.0hr","8.5hr","9.0hr","9.5hr", "10.0hr","10.5hr","11.0hr","11.5hr","12.0hr",
                    ]
    plt.rcParams['figure.figsize'] = (12, 8)
    labels = np.arange(0, 25)
    _, ax = plt.subplots()
    ax.bar(labels, rep_residual, yerr=rep_residual_err)
    ax.set_xlabel("Time in every 30 minutes interval", fontsize=25)
    ax.set_ylabel(f"Mean Residual Value", fontsize=25)
    ax.set_xticks(labels, fontsize=25)
    ax.set_xticklabels(timepoints, rotation=70, fontsize=25)
    ax.set_title(f"Mean Residuals Distribution of subject {sid}", fontsize=25)
    # ax.legend(fontsize=12)
    plt.savefig(ALL_RESIDUALS / "residual_plot" / f"residual_plot_err_{sid}.png", bbox_inches='tight')
    plt.close()







if __name__=="__main__":

    ALL_PICKLE_PATH = "/home/mostafiz/Downloads/MIMIC-get/logs/dl_logs/LSTM+prev_lact/lightning_logs/lightning_logs/version_*/pred_batches/results00.pickle"

    for n_sids in range(0, 13):
        ALL_REP_FOLD_RESIDUALS = []
        RESULT_SIDS=""
        for pikle_file in glob.glob(ALL_PICKLE_PATH):
            with open(pikle_file, "rb") as handle:
                RESULTS: ValidationResults = pickle.load(handle)
                RESULT = RESULTS.results[n_sids]
                RESULT_RESI = RESULT.preds - RESULT.targets
                ALL_REP_FOLD_RESIDUALS.append(RESULT_RESI)
                RESULT_SIDS=RESULT.sid
                # print(RESULT_SIDS)
        ALL_REP_FOLD_RESIDUALS_TRANSPOSE = np.transpose(ALL_REP_FOLD_RESIDUALS, (2, 0, 1))
        # print(np.shape(ALL_REP_FOLD_RESIDUALS), np.shape(ALL_REP_FOLD_RESIDUALS_TRANSPOSE), 
        #                 np.shape(ALL_REP_FOLD_RESIDUALS_TRANSPOSE.mean(axis=(1, 2))))
        # print((ALL_REP_FOLD_RESIDUALS_TRANSPOSE.mean(axis=-2).std(ddof=1, axis=-1)))
        plot_histogram_residuals(ALL_REP_FOLD_RESIDUALS_TRANSPOSE, RESULT_SIDS)
        # PICKLE_FILENAME = "/home/mostafiz/Downloads/MIMIC-get/all_residuals/" + "all_rep_fold_residuals_" + str(RESULT_SIDS)
        # with open(PICKLE_FILENAME, 'wb') as handle:
        #     pickle.dump(ALL_REP_FOLD_RESIDUALS_TRANSPOSE, handle)


