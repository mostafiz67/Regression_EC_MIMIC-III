# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import traceback
from time import ctime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.dataframes import subject_chart_ABP_mean_df

from src._logging.base import LOGGER
from src.acquisition.headers.subject import RawSubject
from src.constants import DATA, ENV_LINKED_IDS, ON_NIAGARA, RESULTS
from src.preprocess.wave_subject import WaveRawSubject

"""We have a number of reasons to distrust from provided timestamps across the different MIMIC
databases. However, some data, as such as ABP data from the matched waveform database,is duplicated
at a much lower sampling frequency in the clinical database under various names, such as "ABP mean".

We implemented an algorithm to align these differently-sampled duplicated waveforms. This script
runs this algorithm with some different settings (clipping or not, e.g. removing noise) and creates
a table estimates of time discrepancies for all subjects.
"""

OUTFILE = RESULTS / "time_discrepancies.parquet"
CHART = pd.read_parquet(DATA / "CHARTEVENTS_CORE_ABP_mean.parquet")
LACT = pd.read_parquet(RESULTS / "LABEVENTS_lactate_minimal.parquet")
SUBJECTS = RawSubject.load_by_ids(ENV_LINKED_IDS)


def make_discrep_row(subject: WaveRawSubject) -> DataFrame:
    chart_df = subject_chart_ABP_mean_df(subject.subject, CHART)
    discreps, clip_discreps = subject.compute_time_discrepancies(
        chart_df, med_filter_size=5, perc_window_hrs=0.25
    )
    return DataFrame(
        dict(
            {
                "ID": subject.sid,
                "Mean Lags": np.mean(discreps),
                "Var Lags": np.var(discreps),
                "Std Lags": np.std(discreps),
                "Mean Clip Lags": np.mean(clip_discreps),
                "Var Clip Lags": np.var(clip_discreps),
                "Std Clip Lags": np.std(clip_discreps),
            }
        ),
        index=[0],
    )


def to_wave_subject(subj: RawSubject) -> Optional[DataFrame]:
    """Takes a subject and computes unclipped and clipped time discrepancy estimates, and returns
    various statistics about these in a DataFrame row"""

    try:
        data_subject = WaveRawSubject(subj)
        return make_discrep_row(data_subject)
    except Exception:
        LOGGER.error(traceback.format_exc())
        LOGGER.error(f"Float / Timestamp for subject causing above error: {subj.id}")
        return None


if __name__ == "__main__":
    LOGGER.info(f"Beginning to compute time discrepancy estimates at {ctime()}")

    if ON_NIAGARA:  # use parallelism
        dfs = process_map(to_wave_subject, SUBJECTS)
        dfs = [df for df in dfs if df is not None]
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:  # use a generator to limit memory usage, and reduced subset for testing
        DATA_SUBJECTS = (WaveRawSubject(subj) for subj in SUBJECTS)

        subject: WaveRawSubject
        rows = []
        for subject in tqdm(
            DATA_SUBJECTS, total=len(ENV_LINKED_IDS), desc="Computing discrepancy estimates"
        ):
            row = make_discrep_row(subject)
            rows.append(row)
        df = pd.concat(rows, axis=0, ignore_index=True)
    df.to_parquet(OUTFILE)
    LOGGER.info(f"File saved successfully to {OUTFILE} at {ctime()}.")
