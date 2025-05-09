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





def plot_histogram_residuals(rep_residual, sid):
    # plt.hist(x=rep_residual, orientation='horizontal', bins=10)
    print(np.shape(rep_residual))
    labels = np.arange(1,26)
    plt.bar(labels, rep_residual)
    plt.xlabel("TIme-points")
    plt.ylabel(f"Prediction Ranges")
    plt.title(f"Prediction Distribution of subject {sid}")
    plt.savefig(ALL_RESIDUALS / "prediction_plot" / f"prediction_hist__{sid}.png", bbox_inches='tight')
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
                RESULT_RESI = RESULT.preds
                ALL_REP_FOLD_RESIDUALS.append(RESULT_RESI)
                RESULT_SIDS=RESULT.sid
                # print(RESULT_SIDS)
        ALL_REP_FOLD_RESIDUALS_TRANSPOSE = np.transpose(ALL_REP_FOLD_RESIDUALS, (2, 0, 1))
        print(np.shape(ALL_REP_FOLD_RESIDUALS), np.shape(ALL_REP_FOLD_RESIDUALS_TRANSPOSE), 
        np.shape(ALL_REP_FOLD_RESIDUALS_TRANSPOSE.mean(axis=(1, 2))))
        plot_histogram_residuals(ALL_REP_FOLD_RESIDUALS_TRANSPOSE.mean(axis=(1, 2)), RESULT_SIDS)
        # PICKLE_FILENAME = "/home/mostafiz/Downloads/MIMIC-get/all_residuals/" + "all_rep_fold_residuals_" + str(RESULT_SIDS)
        # with open(PICKLE_FILENAME, 'wb') as handle:
        #     pickle.dump(ALL_REP_FOLD_RESIDUALS_TRANSPOSE, handle)


