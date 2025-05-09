# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import pickle
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src._logging.base import LOGGER
from src.constants import BASE_FREQUENCY, LACT_VAL, MAPPINGS_CACHE
from src.metrics.interpolated import nearest_next
from src.models.deeplearning.arguments import SmoothingMethod, WindowArgs
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.containers.lactate import InterpMethod, Lactate
from src.models.deeplearning.utils import decimated_folder, window_dimensions
from src.models.deeplearning.windows import get_wave_endpoints
from src.preprocess.containers.wave import Wave
from src.preprocess.spikes import SpikeRemoval

"""The core logic here is to determine how many usable batches can be produced given options (window
sizes, decimation).

Notes
-----
The `predictor_window_size` determines the endpoint of the last valid predictor window.
From there, with the lag, we can determine the start of the target window, and then
check that the target is interpolable over the needed range.

Basically we have an x_max_idx determined by the predictor window size (and decimation) only.
Then we just keep trying to grab windows until

Example 1: predictor_window_size=2, lag=2, target_window_size=3.
y_times_interp[0] > x_decimated[0] and y_times_interp[-1] > x_decimated[-1] + lag
All x windows are valid, all y windows are valid
_____________________________________________________________________________

x_full_times:   ...............................................
x_decimated:    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
y_times:                           x          x x  x       x           x   x
y_times_interp: *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
                    |-lag-|
x_first_window: .  .
y_first_window:           ******
x_last_window:                                            .  .
y_last_window:                                                     *  *  *
                                                                |-lag-|


Example 2: predictor_window_size=2, lag=2, target_window_size=3, less target at end
y_times_interp[0] > x_decimated[0] and y_times_interp[-1] > x_decimated[-1] + lag
_____________________________________________________________________________

x_full_times:   ...............................................
x_decimated:    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
y_times:                           x          x x  x       x           x
y_times_interp:          *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
                    |-lag-|
x_first_window: .  .
y_first_window:          *  *  *
x_last_window:                                         .  .
y_last_window:                                                  *  *  *
                                                            |-lag-|

Example 3: predictor_window_size=2, lag=2, target_window_size=3,
_____________________________________________________________________________

x_full_times:   ...............................................
x_decimated:    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
y_times:                           x          x x  x     x
y_times_interp:          *  *  *  *  *  *  *  *  *  *  *
                    |-lag-|
x_first_window: .  .
y_first_window:          *  *  *
x_last_window:                           .  .
y_last_window:                                    *  *  *
                                            |-lag-|


The correct count here is less than both:
    - number of valid predictor windows
    - number of valid target windows


"""

@dataclass
class WindowInfo:
    wave: Wave
    subject: DeepSubject
    predictor_start_idx: int  # not time, position of

Mappings = Dict[int, WindowInfo]

TrainBatch = Tuple[Tensor, Tensor, ndarray]  # x, y, interp_distances
"""Type of training batches. Tuple of `(x, y, interp_distances)`, where `interp_distances.shape ==
y.shape`, the predictor is `x`, and target is `y`"""

ValBatch = Tuple[
    Tensor, Tensor, ndarray, Tensor, Tensor, str
]  # x, y, interp_distance, y_hours, sid
"""Type of validation batches. Tuple of `(x, y, interp_distances, x_hours, y_hours, sid)`, where
`interp_distances.shape == y.shape`, the predictor is `x`, target is `y`, target timepoints are
`y_hours`, predictor timepoints are `x_hours`, and `sid` is the subject that yieled the batch."""


def get_cached_mappings(
    subjects: Optional[Union[int, Sequence[str]]],
    data_source: Path,
    predictor_size: int,
    lag_minutes: float,
    target_window_minutes: float,
    decimation: int,
    predecimated: bool,
    spike_removal: SpikeRemoval,
    interp_method: InterpMethod,
    ignore_loading_errors: bool,
    progress: bool,
) -> Tuple[Mappings, List[str], DataFrame]:
    """Creates an Info object for each mapping."""
    # dirty hashing for caching  https://stackoverflow.com/a/1151705
    hsh = sha256(str(tuple(sorted(locals().items()))).encode()).hexdigest()
    outfile = MAPPINGS_CACHE / f"{hsh}.pickle"
    if outfile.exists():
        with open(outfile, "rb") as handle:
            return cast(Tuple[Mappings, List[str], DataFrame], pickle.load(handle))

    if predecimated:
        data_source = decimated_folder(decimation)
    if subjects is None or hasattr(subjects, "__iter__"):
        deep_subjects: List[DeepSubject] = DeepSubject.initialize_from_sids(
            sids=subjects,
            source=data_source,
            predictor_window_size=predictor_size,
            lag_minutes=lag_minutes,
            target_window_minutes=target_window_minutes,
            decimation=decimation,
            predecimated=predecimated,
            spike_removal=spike_removal,
            interp_method=interp_method,
            ignore_errors=ignore_loading_errors,
            progress=progress,
        )
    elif isinstance(subjects, int):
        deep_subjects = DeepSubject.initialize_all_from_directory(
            root=data_source,
            predictor_window_size=predictor_size,
            lag_minutes=lag_minutes,
            target_window_minutes=target_window_minutes,
            decimation=decimation,
            predecimated=predecimated,
            spike_removal=spike_removal,
            interp_method=interp_method,
            ignore_errors=ignore_loading_errors,
            limit=subjects,
            progress=progress,
        )
    else:
        raise TypeError("Argument to `subjects` must be int, None, or list of strings.")
    if len(deep_subjects) == 0:
        raise FileNotFoundError(
            f"No usable subjects found at {data_source}. See above for likely cause."
        )

    mappings: Dict[int, WindowInfo] = {}
    subject: DeepSubject
    i = 0
    # dfs = []
    for subject in tqdm(
        deep_subjects, desc="Building subject index mappings", disable=not progress
    ):
        for w, wave in enumerate(subject.waves):
            result = get_wave_endpoints(
                wave,
                subject.lactate,
                predictor_window_size=predictor_size,
                lag_minutes=lag_minutes,
                target_window_minutes=target_window_minutes,
                decimation=decimation,
                full_stats=False,
            )
            if result is None:
                continue
            S, F, df = result
            # df.index = [f"{subject.sid}_wave{w}"]
            # dfs.append(df)
            for start in range(S, F + 1):
                mappings[i] = WindowInfo(wave, subject, start)
                i += 1
    if i == 0:
        raise RuntimeError("No valid data could be loaded with the current settings.")
    sids = list(map(lambda s: s.sid, deep_subjects))  # type: ignore

    # df = pd.concat(dfs, axis=0, ignore_index=False)
    df = None

    LOGGER.info("Mappings built. Caching... ")
    if not outfile.parent.exists():
        outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump((mappings, sids, df), handle, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info(f"Done caching mappings. Mappings cached at {outfile}\n")
    return mappings, sids, df


class WindowDataset(Dataset):
    def __init__(
        self,
        subjects: Optional[Union[int, Sequence[str]]],
        data_source: Path,
        local_smoothing_method: Optional[SmoothingMethod] = None,
        local_smoothing_value: Optional[Union[int, float]] = None,
        desired_predictor_window_minutes: float = 1,
        lag_minutes: float = 0,
        target_window_minutes: int = None,
        target_window_period_minutes: int = None,
        include_prev_target_as_predictor: bool = False,
        include_predictor_times: bool = False,
        is_validation: bool = False,
        decimation: int = 1,
        predecimated: bool = True,
        spike_removal: SpikeRemoval = None,
        target_interpolation: InterpMethod = InterpMethod.previous,
        target_dropout: float = 0,
        # target_noise: float = 0,  # for constant subjects
        preshuffle: bool = False,
        ignore_loading_errors: bool = True,
        progress: bool = True,
    ) -> None:
        """Summary

        Parameters
        ----------
        subjects: Optional[Union[int, Sequence[str]]],
            If a sequence of strings, subject ids from which to create DeepSubject objects.
            If an integer, loads only `n` subjects (intended only for testing).
            If None, loads aDeepSubject`s for all data in `data_source`.

        data_source: Path
            Folder holding subject folders.

        local_smoothing_method: Optional[SmoothingMethod] = None,
            How to smooth each window prior to returning from __getitem__.

        local_smoothing_value: Optional[Union[int, float]] = None,
            Smoothing degree. If `local_smoothing_method` is None, this is ignore.
            If `local_smoothing_method` is SmoothingMethod.Lowpass, uses a Butterworth sosfiltfilt
            filter with `local_smoothing_value` (float) giving the period cutoff in SECONDS.

            If `local_smoothing_method` is SmoothingMethod.Mean, then a moving average filter
            with padding=reflect and window_size=`local_smoothing_value`

            If `local_smoothing_method` is SmoothingMethod.Median, then a rolling median filter
            with padding=reflect and window_size=`local_smoothing_value`

            If `local_smoothing_method` is SmoothingMethod.Savgol, then smoothing is Savitsky-Golay
            with window_size=`local_smoothing_value`.

        desired_predictor_window_minutes: float
            Desired duration of predictor window in minutes.

        lag_minutes: float
            Desired distance (in minutes) between end of predictor (X) window last timepoint and
            start of target lactate window. That is, if our chosen predictor window size and
            decimation results in a predictors window with values occuring at timepoints
            (t_0, t_1 ... t_n), then the first target value to predict occurs at t_n +
            desired_lag_minutes. If `include_prev_target_as_predictor` is True, the additional
            channel of past target values included in the predictor will also be the values of the
            target interpolated to (t_0, t_1, ..., t_n).

        target_window_minutes: Optional[int]
            Duration (in minutes) of target window to be predicted. To simplify logic, and because
            values less than 1 minute are unnecessary, we require this to be an integer.

        target_window_period_minutes: int = None
            Period of target window when interpolating. For example, suppose the target window is to
            be 10 minutes long, and the first point of a particular sample starts at 3 minutes past
            the subject start point. Setting the target window period to 10 minutes means there will
            be 10 / 2 = 5 periods, so the target timepoints for interpolation will be: [3, 5, 7, 9,
            11, 13], i.e. there are = 5 + 1 points. Thus setting
            target_window_period_minutes=target_window_minutes produces target windows of size=2.
            To produce a target window of size=1, pass in target_window_period_minutes=0. In this
            case, the target value to predict will occur at exactly `lag_minutes` from the end of
            the predictor window.

        include_prev_target_as_predictor: bool
            If True include a channel of past target values in the predictor.

        include_predictor_times: bool = False
            If True, include a channel of the time points in the predictor.

        is_validation: bool
            If True returns extra information needed for validation in batch.

        decimation: int = 1
            By how much to decimate predictor data prior to windowing. IMPORTANT: decimation occurs
            *after* reading in the full (undecimated) data if `predecimated` is False. Note if
            using `predecimated=True`, `decimation` must be in [5, 25, 125, 250, 500, 1000].

        predecimated: bool = True
            If True, loads subjects that have been decimated with `scipy.signal.resample_poly` and
            are in PREDECIMATED_DATA.

        spike_removal: SpikeRemoval = None
            If True, use image thresholding (Yen) on the high-frequency signal
            components to identify and remove peaks.

        target_interpolation: InterpMethod = InterpMethod.previous
            How to interpolate between target values.

        target_dropout: float = 0
            How often to zero out target values when `include_prev_target_as_predictor` is True.

        preshuffle: bool = False
            If True, scramble indices. Useful for when using limit_val_batches to ensure windows
            from all validation subjects.

        ignore_loading_errors: bool = True
            If True, don't load subjects that cause errors of various kinds.

        progress: bool = True
            If True, show progress bars for various data construction phases.

        Properties
        -----------
        predictor_shape: Tuple[int, int]
            Shape (seq_length, n_channels) of predictor window.

        target_shape: Tuple[int]
            Shape (seq_length,) of target window.

        """
        args = locals()
        args.pop("self")
        self.preshuffle = args.pop("preshuffle")
        self.loader_args = WindowArgs(**args)
        self.data_source = data_source
        self.decimation = decimation
        self.predecimated = predecimated
        self.spike_removal = spike_removal
        self.preshuffle = preshuffle
        if self.predecimated:
            if self.decimation not in [5, 25, 125, 250, 500, 1000]:
                raise ValueError(
                    "Pre-decimated data available only for decimations in: [5, 25, 125, 250, 500, 1000]"
                )
        self.local_smoothing_method = local_smoothing_method
        self.local_smoothing_value = local_smoothing_value
        if target_dropout < 0 or target_dropout > 1:
            raise ValueError("`target_dropout` must be in [0, 1]")
        if self.local_smoothing_method is SmoothingMethod.Lowpass:
            # self.local_smoothing_value is period in minutes in this case
            # if `fs` and cutoffs are the same unit (e.g. all minutes, all seconds) all is good
            freq_seconds = BASE_FREQUENCY / self.decimation
            pass_T_seconds = float(self.local_smoothing_value)
            crit_freq = 1 / pass_T_seconds
            if crit_freq >= freq_seconds / 2:
                raise ValueError(
                    f"The lowpass critical frequency (cut-point) requested ({crit_freq}Hz, T = {pass_T_seconds}s) "
                    f"must be in (0, {freq_seconds / 2}) = (0, sampling_freq={freq_seconds}/2). Thus the lowest "
                    f"highest possible period for a lowpass filter in this case is {2 / freq_seconds}. "
                )
            self.sos = butter(4, crit_freq, btype="low", output="sos", fs=freq_seconds)

        if "decim" in str(data_source) and self.decimation != 1:
            raise RuntimeError(
                "If using pre-decimated data, setting `decimation` to anything other than 1 "
                "will currently result in errors in calculations."
            )

        if not isinstance(target_window_minutes, int):
            raise TypeError("`target_window_minutes` must be an integer.")
        y_W = target_window_period_minutes
        if not isinstance(y_W, int):
            raise TypeError("`target_window_period_minutes` must be an integer.")
        if target_window_period_minutes != 0 and (target_window_minutes % y_W != 0):
            raise ValueError(
                "Target window period must divide `target_window_minutes` without remainder."
            )

        self.desired_predictor_window_minutes = desired_predictor_window_minutes
        self.lag_minutes = lag_minutes
        self.target_minutes = self.target_window_minutes = target_window_minutes
        self.target_window_period_minutes = target_window_period_minutes
        self.target_size = target_window_minutes // y_W + 1 if y_W != 0 else 1
        self.target_interpolation = target_interpolation
        self.target_dropout = target_dropout
        self.include_prev_target_as_predictor = include_prev_target_as_predictor
        self.include_predictor_times = include_predictor_times
        self.is_validation = is_validation
        self.predictor_size, self.predictor_minutes = self.window_size(
            self.desired_predictor_window_minutes
        )
        predictor_shape = [self.predictor_size, 1]
        if self.include_predictor_times:
            predictor_shape[1] += 1
        if self.include_prev_target_as_predictor:
            predictor_shape[1] += 1
        self.predictor_shape = tuple(predictor_shape)
        self.target_shape = (self.target_size,)
        self.mappings, self.sids, self.predictor_stats = get_cached_mappings(
            subjects=subjects,
            data_source=self.data_source,
            predictor_size=self.predictor_size,
            lag_minutes=self.lag_minutes,
            target_window_minutes=self.target_minutes,
            decimation=self.decimation,
            predecimated=self.predecimated,
            spike_removal=self.spike_removal,
            interp_method=self.target_interpolation,
            ignore_loading_errors=ignore_loading_errors,
            progress=progress,
        )
        # if self.preshuffle:
        #     self.idx = np.random.permutation(len(self.mappings))

    def __getitem__(self, i: int) -> Union[TrainBatch, ValBatch]:
        """Returns a single batch

        Parameters
        ----------
        i: int
            index for which the mapping needs to be returned

        Returns
        -------
        predictor: Tensor
            Predictor values for the batch, with shape (seq_length, n_channels) always

        target: Tensor
            Returns interpolated lact values for that batch with shape (seq_length,)

        distances: Tensor
            Distances of interpolated target values from nearest target value. Has same shape as
            `target`.

        x_hours: Tensor
            Times (in hours) of predictor window values.

        y_hours: Tensor
            Times (in hours) of interpolated targets.

        sid: str
            The id of the source subject for the current sample.
        """
        if self.preshuffle:
            i = np.random.choice(len(self))
        # info: WindowInfo = self.mappings[self.idx[i]] if self.preshuffle else self.mappings[i]
        info: WindowInfo = self.mappings[i]
        x_start = info.predictor_start_idx
        x_end = x_start + self.predictor_size
        x = info.wave.values[x_start:x_end]
        x_hours = info.wave.hours[x_start:x_end]

        # perform local smoothing
        if self.local_smoothing_method is not None:
            win_size = self.local_smoothing_value
            if self.local_smoothing_method is SmoothingMethod.Mean:
                x = torch.tensor(uniform_filter1d(x, win_size), dtype=torch.float32)
            elif self.local_smoothing_method is SmoothingMethod.Median:
                x = torch.tensor(median_filter(x, win_size), dtype=torch.float32)
            elif self.local_smoothing_method is SmoothingMethod.Savgol:
                x = torch.tensor(savgol_filter(x, win_size), dtype=torch.float32)
            elif self.local_smoothing_method is SmoothingMethod.Lowpass:
                x = torch.tensor(sosfiltfilt(self.sos, x.numpy()).copy(), dtype=torch.float32)

        y_start = x_hours[-1] + self.lag_minutes / 60
        y_end = y_start + self.target_minutes / 60
        lactate = info.subject.lactate

        predictors: List[Tensor] = []
        predictors.append(x)
        if self.include_prev_target_as_predictor:
            if self.target_dropout > 0:
                prev_lact = (
                    lactate.interpolator.predict(x_hours)
                    if torch.rand(1) < self.target_dropout
                    else torch.zeros_like(x_hours, dtype=torch.float32)
                )
                predictors.append(prev_lact)
            else:
                predictors.append(lactate.interpolator.predict(x_hours))
        if self.include_predictor_times:
            predictors.append(x_hours)
        x = torch.stack(predictors, dim=1)  # so channels are in order [x, y, t]

        y_hours = torch.linspace(y_start, y_end, self.target_size)
        y = lactate.interpolator.predict(y_hours)
        interpolated_dist = calculate_interpolated_distance(y_hours.numpy(), lactate.hours.numpy())

        if self.is_validation:
            return Tensor(x), Tensor(y), interpolated_dist, x_hours, y_hours, info.subject.sid
        return Tensor(x), Tensor(y), interpolated_dist

    def __len__(self) -> int:
        return len(self.mappings)

    def window_size(self, minutes: float) -> Tuple[int, float]:
        """Convert a desired window duration to size based on constructor args"""
        return window_dimensions(minutes, self.decimation)

    @staticmethod
    def get_predictor_shape(dataset_args: WindowArgs) -> Tuple[int, int]:
        predictor_size, predictor_minutes = window_dimensions(
            dataset_args.desired_predictor_window_minutes, dataset_args.decimation
        )
        predictor_shape = [predictor_size, 1]
        if dataset_args.include_predictor_times:
            predictor_shape[1] += 1
        if dataset_args.include_prev_target_as_predictor:
            predictor_shape[1] += 1
        return tuple(predictor_shape)

    @staticmethod
    def singular_window_minutes(decimation: int) -> float:
        """Find out how long a one-element window is in minutes given decimation"""
        factor = decimation / (BASE_FREQUENCY * 60)
        return float(1 / factor)

    @staticmethod
    def get_window_stats(
        sids: Optional[Union[int, Sequence[str]]],
        data_source: Path,
        desired_predictor_window_minutes: float,
        lag_minutes: float = 0,
        target_window_minutes: int = 1,
        target_window_period_minutes: int = 0,
        decimation: int = 1,
        predecimated: bool = False,
        ignore_loading_errors: bool = True,
        progress: bool = True,
    ) -> DataFrame:
        if "decim" in str(data_source) and decimation != 1:
            raise RuntimeError(
                "If using pre-decimated data, setting `decimation` to anything other than 1 "
                "will currently result in errors in calculations."
            )

        if not isinstance(target_window_minutes, int):
            raise TypeError("`target_window_minutes` must be an integer.")
        y_W = target_window_period_minutes
        if not isinstance(y_W, int):
            raise TypeError("`target_window_period_minutes` must be an integer.")
        if target_window_period_minutes != 0 and (target_window_minutes % y_W != 0):
            raise ValueError(
                "Target window period must divide `target_window_minutes` without remainder."
            )
        predictor_size, predictor_minutes = window_dimensions(
            desired_predictor_window_minutes, decimation
        )
        if sids is None or hasattr(sids, "__iter__"):
            subjects: List[DeepSubject] = DeepSubject.initialize_from_sids(
                sids=sids,
                source=data_source,
                predictor_window_size=predictor_size,
                lag_minutes=lag_minutes,
                target_window_minutes=target_window_minutes,
                decimation=decimation,
                predecimated=predecimated,
                ignore_errors=ignore_loading_errors,
                interp_method=InterpMethod.linear,
                progress=progress,
            )
        elif isinstance(sids, int):
            subjects = DeepSubject.initialize_all_from_directory(
                root=data_source,
                predictor_window_size=predictor_size,
                lag_minutes=lag_minutes,
                target_window_minutes=target_window_minutes,
                decimation=decimation,
                predecimated=predecimated,
                ignore_errors=ignore_loading_errors,
                limit=sids,
                interp_method=InterpMethod.linear,
                progress=progress,
            )

        dfs = []
        for subject in tqdm(subjects, desc="Getting subject window stats", disable=not progress):
            for w, wave in enumerate(subject.waves):
                result = get_wave_endpoints(
                    wave,
                    subject.lactate,
                    predictor_window_size=predictor_size,
                    lag_minutes=lag_minutes,
                    target_window_minutes=target_window_minutes,
                    decimation=decimation,
                    full_stats=False,
                )
                if result is None:
                    continue
                S, F = result[:2]
                df = DataFrame(
                    dict(
                        id=f"{subject.sid}_wave{w}",
                        n_windows=F + 1 - S,
                        dec=decimation,
                        desired_x_mins=desired_predictor_window_minutes,
                        x_mins=predictor_minutes,
                        lag_mins=lag_minutes,
                        y_mins=target_window_minutes,
                        y_period=target_window_period_minutes,
                    ),
                    index=[0],
                )
                dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df


def calculate_interpolated_distance(
    input_lact_times: ndarray, rescaled_lact_times: ndarray, method: str = "nearest"
) -> Tensor:
    if method == "nearest":
        interpolated_dist = nearest_next(input_lact_times, rescaled_lact_times)
    else:
        raise ValueError("Method not implemented")
    return torch.tensor(interpolated_dist, dtype=torch.float32)


class DummyValidationModel:
    def __init__(self, dataset: WindowDataset, method: str = "previous") -> None:
        """Comptues prediction (one window at a time) based on two simple methods:
        1) Predicting next lact value as previous lactate value
        2) Predicting next lact value as mean of the lactate value"""
        self.dataset = dataset
        self.method = method

    def calculate_loss(self) -> Any:
        losses = []
        for i, data in enumerate(tqdm(DataLoader(self.dataset), desc="Computing losses:")):
            X, y_true = data[0], data[1]
            # X.shape (B, window_size, 2) - 2 here is ABP/ lact
            # X[0] -> goes into the window
            # X[0][:,1] -> gets the lactate column
            # X[0][:,1][-1] -> gets the last lactate value
            if X.shape == (1, 900):
                continue
            if self.method == "previous":
                y_pred = X[0][:, 1][-1]
            elif self.method == "mean":
                y_pred = X[0][:, 1].mean()
            else:
                raise ValueError(f"Method {self.method} not implemented")
            # window 70878 no y_true for some reason
            if len(y_true[0]) == 0:
                LOGGER.error(f"{i} batch does not have a y_true val")
                continue
            y_true = y_true[0][0]
            loss = np.absolute(y_pred - y_true)
            losses.append(loss)

        return np.mean(np.array(losses))


def dummy_model(d_subjects: DeepSubject) -> pd.DataFrame:
    dfs = []
    for d_subject in d_subjects:
        lact = d_subject.lactate.data
        lact_vals = lact[LACT_VAL].to_numpy()
        lact_pred = lact_vals[:-1]
        loss = np.mean(np.absolute(lact_vals[1:] - lact_pred))
        df = lact[LACT_VAL].describe()
        df["loss"] = loss
        df = pd.DataFrame(df).T
        df.index = pd.Index(data=[d_subject.id], name="sid")
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    # LOGGER.info(df)
    # LOGGER.info(df.describe())
    return df
