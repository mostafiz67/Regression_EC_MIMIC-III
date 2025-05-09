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


from src.constants import INTERP_THRESHOLD, LACT_MAX_MMOL
from src.constants import MILLIMOLAR_TO_MGDL as MGDL

ALL_PICKLE_PATH = "/home/mostafiz/Downloads/MIMIC-get/logs/dl_logs/LSTM+prev_lact/lightning_logs/lightning_logs/version_*/pred_batches/results00.pickle"


for n_sids in range(0, 13):
    dfs = []
    ALL_REP_FOLD_MAE, ALL_REP_FOLD_NMAE = [], []
    RESULT_SIDS=""
    for pikle_file in glob.glob(ALL_PICKLE_PATH):
        with open(pikle_file, "rb") as handle:
            RESULTS: ValidationResults = pickle.load(handle)
            RESULT = RESULTS.results[n_sids]
            RESULT_SIDS=RESULT.sid
            errs = (RESULT.preds - RESULT.targets) * LACT_MAX_MMOL * MGDL # N_FRAME, T_POINTS
            ds = (RESULT.distances) # N_FRAME, T_POINTS
            idx = ds < INTERP_THRESHOLD # N_FRAME, T_POINTS
            # print("True",idx.sum())
            aes = np.abs(errs) # N_FRAME, T_POINTS
            naes = np.abs(errs[idx]) # (idx.sum())
            ALL_REP_FOLD_MAE.append(aes)
            ALL_REP_FOLD_NMAE.append(naes)

            print(RESULT_SIDS)
    ALL_REP_FOLD_MAE_TRANSPOSE = np.transpose(ALL_REP_FOLD_MAE, (2, 0, 1))
    print(np.shape(ALL_REP_FOLD_MAE_TRANSPOSE), np.shape(ALL_REP_FOLD_MAE))
    
    ALL_REP_FOLD_MAE_TRANSPOSE_MEAN = ALL_REP_FOLD_MAE_TRANSPOSE
    print(np.shape(ALL_REP_FOLD_MAE_TRANSPOSE_MEAN))
    cols = [f'Mean_target_{i+1}' for i in range(ALL_REP_FOLD_MAE_TRANSPOSE_MEAN.shape[0])]
    df = DataFrame([ALL_REP_FOLD_MAE_TRANSPOSE_MEAN.mean(axis=(1, 2))], columns=cols, index=[RESULT_SIDS])
    filename = f"MAE_mean_error_{RESULT_SIDS}.csv"
    outfile = ALL_RESIDUALS /"all_mae" / filename
    # df.to_csv(outfile)
    # print(f"Saved results for error to {outfile}")
    print(np.shape(df), df.describe())