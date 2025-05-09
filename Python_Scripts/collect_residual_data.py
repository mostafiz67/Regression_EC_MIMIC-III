# fmt: off
import sys  # isort:skip
from pathlib import Path
from telnetlib import NAWS  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
# fmt: on
import pickle
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.models.deeplearning.validation import ValidationResults
PLOT_ROOT = Path(__file__).resolve().parent
ALL_RESIDUALS = PLOT_ROOT / "all_residuals"
EC_PLOTS = PLOT_ROOT / "ec_plots"


ALL_PICKLE_PATH = "/home/mostafiz/Downloads/MIMIC-get/logs/dl_logs/LSTM+prev_lact/lightning_logs/lightning_logs/version_*/pred_batches/results00.pickle"


for n_sids in range(0, 13):
    dfs = []
    ALL_REP_FOLD_RESIDUALS, ALL_REP_FOLD_NMAE = [], []
    RESULT_SIDS=""
    for pikle_file in glob.glob(ALL_PICKLE_PATH):
        with open(pikle_file, "rb") as handle:
            RESULTS: ValidationResults = pickle.load(handle)
            RESULT = RESULTS.results[n_sids]
            RESULT_SIDS=RESULT.sid
            residuals = (RESULT.preds - RESULT.targets) 
            ALL_REP_FOLD_RESIDUALS.append(residuals)

            print(RESULT_SIDS)
    ALL_REP_FOLD_RESIDUALS_TRANSPOSE = np.transpose(ALL_REP_FOLD_RESIDUALS, (2, 0, 1))
    # print(np.shape(ALL_REP_FOLD_MAE_TRANSPOSE), np.shape(ALL_REP_FOLD_RESIDUALS))
    
    ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN = ALL_REP_FOLD_RESIDUALS_TRANSPOSE
    # print(np.shape(ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN))
    summaries= []
    cols = [f'Mean_residual_{i+1}' for i in range(ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.shape[0])]
    cols_sd = [f'Mean_residual_sd_{i+1}' for i in range(ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.shape[0])]
    # df = DataFrame([ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.mean(axis=(1, 2))], columns=cols, index=[RESULT_SIDS])
    df1 = DataFrame({"Residual_Mean": ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.mean(), 
                    "Residual_Mean_sd": ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.mean(axis=0).std(ddof=1)}, index=[0])
    df2 = DataFrame([ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.mean(axis=(1, 2))], columns=cols)
    df3 = DataFrame([ALL_REP_FOLD_RESIDUALS_TRANSPOSE_MEAN.mean(axis=-2).std(ddof=1, axis=-1)], columns=cols_sd)
    summaries.append(pd.concat([df1, df2, df3], axis=1))
    summary = pd.concat(summaries, axis=0, ignore_index=True)
    
    filename = f"Residuals_mean_sd_{RESULT_SIDS}.csv"
    outfile = ALL_RESIDUALS / filename
    print(summary)
    summary.to_csv(outfile)
    # print(f"Saved results for error to {outfile}")
    # print(np.shape(df), df.describe())