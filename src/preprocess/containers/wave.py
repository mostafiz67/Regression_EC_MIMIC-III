from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from math import ceil
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Optional, Tuple
from warnings import filterwarnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from numba import jit
from numpy import ndarray
from pandas import DataFrame
from pyarrow.parquet import ParquetDataset
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import BASE_FREQUENCY, WAVE_VAL
from src.models.deeplearning.containers.lactate import Lactate
from src.models.deeplearning.utils import validate_predecimation
from src.preprocess.spikes import SpikeRemoval, highpass_spike_remove


class NanInterpolation(Enum):
    previous = "previous"
    linear = "linear"


@dataclass
class WaveStatArgs:
    # x_shm_id: str
    # x_shape: Tuple[int, ...]
    # h_shm_id: str
    # h_shape: Tuple[int, ...]
    x: ndarray
    hours: ndarray
    target: Lactate
    start: int
    predictor_size: int
    lag_minutes: float
    target_window_minutes: float
    target_size: int


# NOTE: you *cannot* use fastmath=True here because that disables detections of NaNs
@jit(nopython=True, cache=True)
def interpolate_nans_previous(values: ndarray) -> None:
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            values[i] = values[i - 1]


class Wave:
    """
    Parameters
    -----------
    wave_path: Path
        Path to .parquet file with date in filename

    first_wave: Path
        Path to first wave .parquet file for normalizing times.

    decimation: int
        Amount of decimation.
    """

    def __init__(
        self,
        wave_path: Path,
        first_wave: Path,
        decimation: int,
        predecimated: bool,
        nan_interpolation: NanInterpolation = NanInterpolation.previous,
        normalize: Optional[Tuple[float, float]] = None,
        spike_removal: Optional[SpikeRemoval] = None,
    ) -> None:
        self.path: Path = wave_path
        self.decimation: int = decimation
        self.predecimated = predecimated
        if self.predecimated:
            validate_predecimation(self.decimation)
        self.nan_interpolation: NanInterpolation = NanInterpolation(nan_interpolation)
        self.should_normalize = normalize is not None
        self.spike_removal = spike_removal
        # mean, median, etc to subtract
        self.normalization_location = (
            np.float32(normalize[0]) if normalize is not None else np.float32(0)
        )
        # sd, iqr, winsorized sd, etc to divide
        self.normalization_scale = (
            np.float32(normalize[1]) if normalize is not None else np.float32(1)
        )
        self.period = self.decimation / (BASE_FREQUENCY * 3600)
        self.start: pd.Timestamp = pd.to_datetime(self.path.stem)
        self.values_: Optional[Tensor] = None
        self.subj_start: pd.Timestamp = pd.to_datetime(first_wave.stem)
        self.start_hours: pd.Timestamp = pd.to_datetime(self.path.stem)
        self.hours_: Optional[Tensor] = None
        self.raw_length = ParquetDataset(self.path).pieces[0].get_metadata().num_rows
        if self.raw_length < 2:
            raise ValueError("Wave contains insufficient data for windowing")

    @property
    def values(self) -> Tensor:
        if self.values_ is not None:
            return self.values_

        if self.predecimated:
            vals = pd.read_parquet(self.path).to_numpy(dtype=np.float32).ravel()
        else:
            vals = pd.read_parquet(self.path).to_numpy(dtype=np.float32).ravel()[:: self.decimation]
        if self.should_normalize:
            vals -= self.normalization_location
            vals /= self.normalization_scale
        self.interpolate_nans(vals)

        if self.spike_removal is not None:
            for _ in range(self.spike_removal.value):
                try:
                    vals = highpass_spike_remove(vals, self.decimation)
                except RuntimeError as e:
                    raise RuntimeError(f"Problem with wave data at {self.path}") from e

        self.values_ = torch.tensor(vals, dtype=torch.float32)
        if torch.isnan(self.values_).sum() > 0:
            raise RuntimeError(f"Data in {self.path} contains NaNs")
        return self.values_

    @property
    def hours(self) -> Tensor:
        if self.hours_ is None:
            n = ParquetDataset(self.path).pieces[0].get_metadata().num_rows
            freq = (
                f"{1/BASE_FREQUENCY}S"
                if not self.predecimated
                else f"{self.decimation/BASE_FREQUENCY}S"
            )
            h = pd.date_range(start=self.start, periods=n, freq=freq)
            h = (h - self.subj_start) / pd.Timedelta(hours=1)
            dec = 1 if self.predecimated else self.decimation
            self.hours_ = torch.tensor(h.to_numpy()[::dec], dtype=torch.float32)
        return self.hours_

    @property
    def hours_0(self) -> float:
        """We need to be able to get the start time without creating a full set of times"""
        return float((self.start - self.subj_start) / pd.Timedelta(hours=1))

    @property
    def hours_f(self) -> float:
        """We need to be able to get the last time without creating a full set of times"""
        n = ParquetDataset(self.path).pieces[0].get_metadata().num_rows
        T = self.period
        # len(x[::dec]) == ceil(len(x) / dec), and if there are t timepoints, that's t-1 periods
        n_d = ceil(n / self.decimation) - 1
        elapsed = float(pd.Timedelta(hours=T * n_d) / pd.Timedelta(hours=1))
        start = self.hours_0
        end = start + elapsed
        return end

    def hours_at(self, i: int) -> float:
        """Get the time at index `i` given the known decimation. Get's same value
        as using pd.date_range()[::decimation][i] but without it. For proof see
        tests/test_time_tricks.py::test_time_at."""
        if i < 0:
            raise IndexError("Negative indices not supported for `hours_at`")
        T = self.period
        elapsed = float(pd.Timedelta(hours=T * i) / pd.Timedelta(hours=1))
        start = self.hours_0
        end = start + elapsed
        return end

    @property
    def decimated_length(self) -> int:
        n = ParquetDataset(self.path).pieces[0].get_metadata().num_rows
        if self.predecimated:
            return n
        n_d = int(ceil(n / self.decimation))
        return n_d

    def __len__(self) -> int:
        return self.decimated_length

    def cleaned(self, method: str = "previous") -> pd.DataFrame:
        df, wave_start = self.clean_wave(self.values, method=method)
        return df, wave_start

    def is_empty(self) -> bool:
        return int(self.values[WAVE_VAL].isna().sum()) == len(self.values)

    def interpolate_nans(self, values: np.float32) -> None:
        interpolate_nans_previous(values)
        if self.nan_interpolation is NanInterpolation.previous:
            return
        else:
            raise NotImplementedError()

    def clean_wave(self, df: DataFrame, method: str = "previous") -> Tuple[ndarray, pd.Timestamp]:
        """Removes border NaNs and fills middle NaNs with directly preceding value"""
        raise RuntimeError(
            "This function is bugged because it destroys time values, start times, etc "
            "as properties, and assumes data is already pre-decimated by 250"
        )
        if self.is_empty():  # type: ignore # noqa
            raise RuntimeError("Empty wave cannot be cleaned")
        df_, trim_start = self.drop_border_nans(df)
        x = df_[WAVE_VAL]
        trimmed_duration = (trim_start / BASE_FREQUENCY) * 250  # 250 is the current decimation
        n = len(df_)
        t = np.asarray(
            pd.date_range(
                start=pd.to_datetime(self.path.stem) + pd.Timedelta(seconds=trimmed_duration),
                periods=n,
                freq=f"{1/0.5}S",
            )
        )
        wave_start = t[0]
        t = np.subtract(t, wave_start) / pd.Timedelta(hours=1)
        nan_idx = np.isnan(x)
        if len(nan_idx) < 1:
            return np.asarray(x, dtype=float32), wave_start
        x_clean, t_clean = x[~nan_idx], t[~nan_idx]
        x_fill = interp1d(
            x=t_clean, y=x_clean, kind=method, bounds_error=False, fill_value=0, copy=False
        )(t[nan_idx])
        x[nan_idx] = x_fill
        return np.asarray(x, dtype=float32), wave_start

    def drop_border_nans(self, df: DataFrame) -> Tuple[pd.DataFrame, int]:
        """Removes border NaNs per wave"""
        start = df[WAVE_VAL].first_valid_index()
        end = df[WAVE_VAL].last_valid_index()
        return df.iloc[start:end], start

    def target_correlations(
        self,
        target: Lactate,
        start: int,
        end: int,
        predictor_size: int,
        lag_minutes: float,
        target_window_minutes: float,
        target_size: int,
    ) -> DataFrame:
        # see https://github.com/joblib/joblib/issues/915#issuecomment-753314152
        # for sane use of parallelism and shared memory here
        try:
            x: ndarray = self.values.numpy().astype(np.float32)[start:end].copy()
            hours: ndarray = self.hours.double().numpy().astype(np.float64)[start:end].copy()
            # id = str(self.path.stem)
            # x_buf = SharedMemory(name=f"{id}_x", create=True, size=x.nbytes)
            # h_buf = SharedMemory(name=f"{id}_h", create=True, size=hours.nbytes)
            # x_shared = np.ndarray(x.shape, dtype=x.dtype, buffer=x_buf.buf)
            # h_shared = np.ndarray(hours.shape, dtype=hours.dtype, buffer=h_buf.buf)
            # x_shared[:] = x
            # h_shared[:] = hours

            end -= start
            start = 0
            list_dec = ceil(end / 1000)
            # args = [
            #     WaveStatArgs(
            #         x_shm_id=f"{id}_x",
            #         x_shape=x.shape,
            #         h_shm_id=f"{id}_h",
            #         h_shape=hours.shape,
            #         target=target,
            #         start=i,
            #         predictor_size=predictor_size,
            #         lag_minutes=lag_minutes,
            #         target_window_minutes=target_window_minutes,
            #         target_size=target_size,
            #     )
            #     for i in range(end)[::list_dec]
            # ]
            args = [
                WaveStatArgs(
                    x=x,
                    hours=hours,
                    target=target,
                    start=i,
                    predictor_size=predictor_size,
                    lag_minutes=lag_minutes,
                    target_window_minutes=target_window_minutes,
                    target_size=target_size,
                )
                for i in range(end)[::list_dec]
            ]
            torch.multiprocessing.set_sharing_strategy("file_system")
            # dfs = Parallel(8, mmap_mode="c")(delayed(wave_stats)(arg) for arg in args)

            # with Pool(8) as pool:
            #     dfs = pool.map(wave_stats, args, chunksize=10)

            # dfs = process_map(
            #     wave_stats,
            #     args,
            #     desc="Computing wave stats",
            #     chunksize=100,
            #     disable=True,
            #     max_workers=2,
            # )

            dfs = [wave_stats(args) for args in args]
            # x_buf.close()
            # x_buf.unlink()
            # h_buf.close()
            # h_buf.unlink()
            return pd.concat(dfs, axis=0, ignore_index=True)
        except:
            traceback.print_exc()
        # finally:
        #     id = str(self.path.stem)
        #     x_buf = SharedMemory(name=f"{id}_x")
        #     h_buf = SharedMemory(name=f"{id}_h")
        #     x_buf.close()
        #     h_buf.close()
        #     h_buf.unlink()
        #     h_buf.unlink()


def correlate_fast(x: ndarray, y: ndarray, ddof: int = 1) -> ndarray:
    """Just a copy of the np.corrcoef source, with extras removed"""
    arr = np.vstack([x, y])
    c = np.cov(arr, ddof=ddof)
    d = np.diag(c).reshape(-1, 1)

    sd = np.sqrt(d)
    if 0 in sd:
        return 0
    c /= sd
    c /= sd.T
    np.clip(c, -1, 1, out=c)

    return c[0, 1]


@jit(nopython=True)
def variance(arr: ndarray) -> float:
    """i.e. s^2"""
    n_ = len(arr)
    if n_ <= 1.0:
        return 0.0
    scale = 1.0 / (n_ - 1.0)
    mean = np.mean(arr)
    diffs = arr - mean
    squares = diffs * diffs
    summed = np.sum(squares)
    return scale * summed  # type: ignore


@jit(nopython=True)
def fast_r(x_: ndarray, y_: ndarray) -> float:
    n = x_.shape[0]
    num = np.sum(x_ * y_) - n * np.mean(x_) * np.mean(y_)
    sy = np.sqrt(variance(y_))
    if sy <= 0:
        return 0.0
    sx = np.sqrt(variance(x_))
    if sx <= 0:
        return 0.0
    denom = (n - 1.0) * sx * sy
    if denom <= 0.0:
        return 0.0
    return num / denom


def wave_stats(args: WaveStatArgs) -> DataFrame:
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # x_buf = SharedMemory(name=args.x_shm_id)
    # h_buf = SharedMemory(name=args.h_shm_id)
    # x: ndarray = np.ndarray(args.x_shape, dtype=np.float32, buffer=x_buf.buf)
    # x_hours: ndarray = np.ndarray(args.h_shape, dtype=np.float64, buffer=h_buf.buf)
    x = args.x
    x_hours = args.hours

    # x = np.frombuffer(args.x, dtype=np.float32)
    # x_hours = np.frombuffer(args.hours, dtype=np.float64)
    x_end = args.start + args.predictor_size
    # do parallel buffer shit for next two lines
    predictor = x[args.start : x_end].astype(np.float32)
    x_hours = x_hours[args.start : x_end]

    with torch.no_grad():
        y_start = x_hours[-1] + args.lag_minutes / 60
        y_end = y_start + args.target_window_minutes / 60
        y_prev = torch.from_numpy(args.target.interpolator.predict(x_hours).astype(np.float32))
        y_hours = torch.linspace(y_start, y_end, args.target_size)
        y = args.target.interpolator.predict(y_hours)
        y_x = interpolate(y.reshape(1, 1, -1), len(predictor)).squeeze().float()

        predictor = torch.from_numpy(predictor)

        # if y_x.ndim == 0:
        #     y_x = np.array([y_x], dtype=np.float64)

        return DataFrame(
            {
                "corr(x,y_prev)": torch.corrcoef(torch.vstack([predictor, y_prev]).to("cuda"))[0, 1].item(),
                "corr(x,y)": torch.corrcoef(torch.vstack([predictor, y_x]).to("cuda"))[0, 1].item(),
                "corr(y_prev,y)": torch.corrcoef(torch.vstack([y_prev, y_x]).to("cuda"))[0, 1].item(),
            },
            index=[0],
        )

    return DataFrame(
        {
            "corr(x,y_prev)": correlate_fast(predictor, y_prev),
            "corr(x,y)": correlate_fast(predictor, y_x),
            "corr(y_prev,y)": correlate_fast(y_prev, y_x),
        },
        index=[0],
    )
