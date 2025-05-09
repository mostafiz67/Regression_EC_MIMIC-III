# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()  # isort: skip
# fmt: on


from pathlib import Path
from typing import Any, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d
from scipy.signal import resample_poly

from src.constants import BASE_FREQUENCY, PREDECIMATED_DATA

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def window_dimensions(desired_minutes: float, decimation: int) -> Tuple[int, float]:
    """Convert a desired window duration to size based on constructor args

    Returns
    -------
    n_points: int
        Number of points in window given `decimation` and `desired_minutes`.j

    actual_minutes: float
        Actual number of minutes for the window given `n_points`.
    """
    freq = BASE_FREQUENCY / decimation  # Hz
    per_min = freq * 60
    exact = desired_minutes * per_min
    actual = int(exact)
    if actual == 0:
        actual = 1
    actual_minutes = actual / per_min
    if exact != actual:
        if actual_minutes < 1:
            readable = f"{actual_minutes * 60} seconds"
        elif actual_minutes < 60:
            readable = f"{actual_minutes} minutes"
        else:
            readable = f"{actual_minutes / 60} hours"

        warn(
            f"At the current decimation of {decimation} a predictor window of "
            f"{desired_minutes} minutes would require a window size of {exact}. Since window "
            f"size must be an integer, the actual size used will be {actual}, i.e. "
            f"{readable}."
        )
    return actual, actual_minutes


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def decimated_folder(decimation: int) -> Path:
    folder = PREDECIMATED_DATA / f"decimation_{decimation}"
    if not folder.exists():
        raise FileNotFoundError(f"Could not find pre-decimated directory at {folder}.")
    return folder


def validate_predecimation(decimation: int) -> None:
    DEC = [5, 25, 125, 250, 500, 1000]
    if decimation not in DEC:
        raise ValueError(f"If using pre-decimated data, `decimation` must be in {DEC}.")


def nan_remove(
    df: DataFrame, df_path: Path, decimation: int
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    start = df.first_valid_index()
    end = df.last_valid_index()
    df = df.iloc[start:end]
    x = df.to_numpy()
    trimmed_duration = start * (decimation / BASE_FREQUENCY)
    n = len(df)
    t = np.asarray(
        pd.date_range(
            start=pd.to_datetime(df_path.stem) + pd.Timedelta(seconds=trimmed_duration),
            periods=n,
            freq=f"{1/0.5}S",
        )
    )
    wave_start = t[0]
    t = np.subtract(t, wave_start) / pd.Timedelta(hours=1)
    nan_idx = np.isnan(x)
    if len(nan_idx) < 1:
        return np.asarray(x, dtype=np.float32), wave_start
    x_clean, t_clean = x[~nan_idx], t[~nan_idx]
    x_fill = interp1d(
        x=t_clean, y=x_clean, kind="previous", bounds_error=False, fill_value=0, copy=False
    )(t[nan_idx])
    x[nan_idx] = x_fill
    return np.asarray(x, dtype=np.float32), wave_start


def remove_border_nans(df: DataFrame, start: str, decimation: int) -> Tuple[DataFrame, str]:
    begin = df.first_valid_index()
    end = df.last_valid_index()
    df = df.iloc[begin:end]
    trimmed = begin * (decimation / BASE_FREQUENCY)  # seconds
    trimmed_start = pd.to_datetime(start) + pd.Timedelta(seconds=trimmed)
    new_start = trimmed_start.strftime("%Y%m%d-%H%M%S")
    return df, new_start


def remove_internal_nans(df: DataFrame) -> DataFrame:
    if df.first_valid_index() != 0:
        raise ValueError(
            "Can only remove internal nans from array without border NaNs "
            "(due to 'previous' interpolation method)."
        )
    nan_idx = df.isnull().to_numpy().ravel()
    if len(nan_idx) < 1:
        return df
    x = df.to_numpy().ravel()  # pandas indexing is stupid trash
    t = np.arange(len(x))
    x_clean, t_clean = x[~nan_idx], t[~nan_idx]
    x_fill = interp1d(
        x=t_clean, y=x_clean, kind="previous", bounds_error=False, fill_value=0, copy=False
    )(t[nan_idx])
    x[nan_idx] = x_fill
    df = pd.DataFrame(data=x, columns=df.columns)
    return df


# See https://stackoverflow.com/a/14606271 for this trick
def split_at_nans(df: DataFrame, df_path: Path) -> Tuple[List[DataFrame], List[str]]:
    x = df.to_numpy()
    split_idx = np.ma.clump_unmasked(np.ma.masked_invalid(x))
    all_waves = [x[idx] for idx in split_idx]
    hrs = np.asarray(
        pd.date_range(
            start=pd.to_datetime(df_path.stem),
            periods=len(df),
            freq=f"{1/BASE_FREQUENCY}S",
        )
    )
    start_times = [hrs[idx][0] for idx in split_idx]
    start_strs = [pd.to_datetime(str(t)).strftime("%Y%m%d-%H%M%S") for t in start_times]
    # smallest possible useful wave for us is likely 5 minutes. At decimation=5,
    # this is 7500 pts ((60s/min * 5min) / (5 / (125 pts/sec))). At no decimation
    # this is (60s/min * 5 min) / (1 pts / 125 sec) = 37 500 pts.
    THRESH = 37_500  # 5 minutes
    waves, starts = [], []
    for wave, start in zip(all_waves, start_strs):
        if len(wave) > THRESH:
            waves.append(wave)
            starts.append(start)
    return waves, starts


if __name__ == "__main__":
    path = Path("/home/derek/Desktop/MIMIC-III_Clinical_Database/21140329-124444.parquet")
    wave = pd.read_parquet(path)
    waves, starts = split_at_nans(wave, path)
    print(f"Starting wave length: {len(wave)}")
    print(f"Number of waves split into: {len(waves)}")
    print(f"Split lengths before decimation: {[len(w) for w in waves]}")
    print(f"Split starts: {starts}")

    sm = pd.DataFrame(resample_poly(wave, up=2, down=2 * 250, padtype="line"), columns=["ABP"])
    sms = [
        pd.DataFrame(resample_poly(w, up=2, down=2 * 250, padtype="line"), columns=["ABP"])
        for w in waves
    ]
    nulls = [int(s.isnull().sum()) for s in sms]
    print(f"NaN count in full resampled: {int(sm.isnull().sum())}")
    print(f"NaNs in splits: {nulls}")
    print(f"Total NaNs across splits: {np.sum(nulls)}")
    print(f"Split lengths after decimation: {[len(s) for s in sms]}")
