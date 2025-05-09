from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import traceback
from enum import Enum
from math import ceil
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import numpy.core.numeric as _nx
import seaborn as sbn
import torch
from numba import jit, prange
from numpy import ndarray
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress
from skimage.filters import threshold_otsu, threshold_yen
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import (
    BASE_FREQUENCY,
    PREDICTOR_MEDIAN_OF_IQRS,
    PREDICTOR_MEDIAN_OF_MEDIANS,
    SPIKE_PERIOD_CUTOFF_S,
)
from src.models.deeplearning.utils import best_rect


class SpikeRemoval(Enum):
    Low = 1
    Medium = 2
    High = 3

    @staticmethod
    def to_enum(level: Union[int, str]) -> Optional[SpikeRemoval]:
        level = int(level)
        if level == 0:
            return None
        if level == 1:
            return SpikeRemoval.Low
        if level == 2:
            return SpikeRemoval.Medium
        if level == 3:
            return SpikeRemoval.High
        raise ValueError("Spike removal level must be in {0, 1, 2, 3}.")


def softsign(x: ndarray) -> ndarray:
    return x / (np.abs(x) + 1)


def sigmoid(x: ndarray) -> ndarray:
    return np.exp(-np.logaddexp(0, -x))


def sosfilter(
    arr: ndarray, decimation: int, period_cutoff_s: float, lowpass: bool = True
) -> ndarray:
    freq_seconds = BASE_FREQUENCY / decimation
    pass_T_seconds = float(period_cutoff_s)
    crit_freq = 1 / pass_T_seconds
    if crit_freq >= freq_seconds / 2:
        raise ValueError(
            f"The lowpass critical frequency (cut-point) requested ({crit_freq}Hz, T = {pass_T_seconds}s) "
            f"must be in (0, {freq_seconds / 2}) = (0, sampling_freq={freq_seconds}/2). Thus the lowest "
            f"highest possible period for a lowpass filter in this case is {2 / freq_seconds}. "
        )
    sos = butter(
        4, crit_freq, btype="lowpass" if lowpass else "highpass", output="sos", fs=freq_seconds
    )
    return sosfiltfilt(sos, arr)


def split_idx(arr: ndarray, grid_size: int) -> List[slice]:
    # using the tricks in np.array_split source code
    length, extras = divmod(len(arr), grid_size)
    section_sizes = [0] + extras * [length + 1] + (grid_size - extras) * [length]
    div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()
    slices = []
    for i in range(grid_size):
        start = div_points[i]
        end = div_points[i + 1]
        slices.append(slice(start, end))
    return slices


def compute_percentiles(args: Tuple[float, slice]) -> Tuple[float, float, float]:
    percentile, idx = args
    arr = np.frombuffer(BUFFER, dtype=np.float32)
    chunk = arr[idx]
    mn, mx = np.percentile(chunk, [percentile, 100 - percentile], interpolation="midpoint")
    return mx, mn


@jit(nopython=True, parallel=True, cache=True)
def clip_to_vals(arr: ndarray, lows: ndarray, highs: ndarray) -> None:
    """Clip in place"""
    for i in prange(len(arr)):
        if arr[i] > highs[i]:
            arr[i] = highs[i]
        if arr[i] < lows[i]:
            arr[i] = lows[i]


def hard_clip(arr: ndarray, min: float = -1.0, max: float = 2.0) -> None:
    med = np.median(arr)
    arr[arr >= max] = med
    arr[arr <= min] = med


def diff_clip(arr: ndarray, thresh: float = 2.0) -> None:
    med = np.median(arr)
    diffs = np.diff(arr, prepend=arr[0])
    idx = diffs > thresh
    arr[idx] = med


def percentile_clip(
    arr: ndarray, times: ndarray, grid_minutes: int, percentile: float = 5.0
) -> ndarray:
    duration = (times[-1] - times[0]) * 60  # to minutes
    grid_size = ceil(duration / grid_minutes)
    idxs = split_idx(arr, grid_size)
    global BUFFER
    tmp = np.ctypeslib.as_ctypes(arr)
    BUFFER = RawArray(tmp._type_, tmp)
    args = [(percentile, idx) for idx in idxs]
    maxs, mins = list(zip(*process_map(compute_percentiles, args, disable=True, chunksize=500)))
    ts = []
    for idx in idxs:
        t_chunk = times[idx]
        ts.append(t_chunk[len(t_chunk) // 2])

    max_interpolator = PchipInterpolator(ts, maxs)
    min_interpolator = PchipInterpolator(ts, mins)
    max_vals = max_interpolator(times)
    min_vals = min_interpolator(times)
    return clip_to_vals(arr, min_vals, max_vals)


def robust_clip(
    arr: ndarray,
    hours: ndarray,
    thresh: float = 5.0,
    window_minutes: float = 1,
) -> None:
    """Where a robustly normalized version of `arr` would have values outside
    of (min, max), replace the values in `arr` with the median of `arr`. Also
    shrink values in arr where the robustly-standardized array has |diffs| >= diff
    by the softsign function (see Notes).

    Parameters
    ----------
    arr: ndarray
        1D array with dtype of float32

    hours: ndarray
        Timepoints in hours

    low: float = -3.0
        Minimum value on the robustly normalized wave.

    high: float = 3.0
        Maximum value on the robustly normalized wave.

    diff: float = 1.0
        Maximum allowable diff on the robustly normalized wave. Smaller values
        correspond to more aggressive spike removal.

    window_seconds: float = 10.0
        How many seconds on either side of spikes to look for computing a local
        median fill.

    Notes
    -----
    Robust standardization of X is X_std = (X - np.median(X)) / IQR, where IQR
    is the absolute difference between the 25th and 75th percentiles of X. A
    robustly-standardized variable will tend to have IQR very near 1.0, so
    setting max = -min = 3.0 is similar to clipping values more than 3 sd
    from the mean in non-robust standardization.

    The softsign function is softsign(x) = x / (|x| + 1).
    """

    def median_backfill(arr: ndarray, idx: ndarray, n: int, min_thresh: float = 0.0):
        # NOTE: with multiple pass, filling naively with the preceding median
        # can introduce flat regions which are plateaus in later passes. We can
        # fix this by adding in small noise just over the plateau threshold
        noises = np.random.uniform(-min_thresh, min_thresh, len(idx))
        for iter, i in enumerate(idx):
            if i <= 1:
                arr[i] = 0
            else:
                start = max(0, i - n)
                med = np.median(arr[start:i])
                arr[i] = med + noises[iter]

    # med = np.median(arr)
    # iqr = np.abs(np.diff(np.percentile(arr, [25, 75])))
    # if np.isnan(iqr) or np.isnan(med):
    #     return arr
    # # now iqr of normed will almost always be 1.0
    # normed = (arr - med) / iqr

    T = np.mean(np.diff(hours))  # hours
    n = ceil((window_minutes / 60) / T)

    # remove plateaus
    # diffs = np.abs(np.diff(normed))
    diffs = np.abs(np.diff(arr))
    min_thresh = np.percentile(diffs, thresh)
    min_idx = np.where(diffs <= min_thresh)[0]
    max_idx = min_idx + 1
    plateau_idx = np.array(list(set(min_idx).union(max_idx)), dtype=int)
    plateau_idx = plateau_idx[(plateau_idx > 0) & (plateau_idx < len(arr))]
    median_backfill(arr, plateau_idx, n, min_thresh)

    # shrink spikes
    # diffs = np.abs(np.diff(normed))  # recalc since plateu removal
    diffs = np.abs(np.diff(arr))  # recalc since plateu removal
    max_thresh = np.percentile(diffs, 100 - thresh)
    min_idx = np.where(diffs >= max_thresh)[0]
    max_idx = min_idx + 1
    spike_idx = np.array(list(set(min_idx).union(max_idx)), dtype=int)
    spike_idx = spike_idx[(spike_idx > 0) & (spike_idx < len(arr))]
    median_backfill(arr, spike_idx, n, min_thresh)

    # finally hard clip by replacing spikes with local median
    low, high = np.percentile(arr, [thresh, 100 - thresh])
    idx = np.where((arr >= high) | (arr <= low))[0]
    median_backfill(arr, idx, n, min_thresh)

    # shrink in small area around spikes
    # surround = 0
    # for i in range(-surround, surround + 1, 1):
    #     for idx in idxs:
    #         all_idx = all_idx.union(idx + i)
    # arr[spike_idx] = softsign(arr[spike_idx])
    # arr[np.abs(diffs) >= diff] = med
    # arr[spike_idx] = med + softsign(arr[spike_idx] - med) / 2


def robust_interpolate(
    arr: ndarray,
    hours: ndarray,
    thresh: float = 5.0,
) -> None:
    """Where a robustly normalized version of `arr` would have values outside
    of (min, max), replace the values in `arr` with the median of `arr`. Also
    shrink values in arr where the robustly-standardized array has |diffs| >= diff
    by the softsign function (see Notes).

    Parameters
    ----------
    arr: ndarray
        1D array with dtype of float32

    hours: ndarray
        Timepoints in hours

    low: float = -3.0
        Minimum value on the robustly normalized wave.

    high: float = 3.0
        Maximum value on the robustly normalized wave.

    diff: float = 1.0
        Maximum allowable diff on the robustly normalized wave. Smaller values
        correspond to more aggressive spike removal.

    window_seconds: float = 10.0
        How many seconds on either side of spikes to look for computing a local
        median fill.

    Notes
    -----
    Robust standardization of X is X_std = (X - np.median(X)) / IQR, where IQR
    is the absolute difference between the 25th and 75th percentiles of X. A
    robustly-standardized variable will tend to have IQR very near 1.0, so
    setting max = -min = 3.0 is similar to clipping values more than 3 sd
    from the mean in non-robust standardization.

    The softsign function is softsign(x) = x / (|x| + 1).
    """

    def interp_idx(arr: ndarray, hours: ndarray, idx: ndarray) -> None:
        interp = PchipInterpolator(hours[~idx], arr[~idx])
        # interp = interp1d(hours[~idx], arr[~idx])
        fill = interp(hours[idx])
        arr[idx] = fill

    T = np.mean(np.diff(hours))  # hours
    n = ceil((5 / 60) / T)  # use first five miutes and last for endpoints
    arr[0] = np.median(arr[:n])
    arr[-1] = np.median(arr[-n:])
    ADJUST = 0.5  # in (-1, 1), -1 contracts the k-means centroids, 1 expands

    # remove plateaus
    # diffs = np.abs(np.diff(normed))
    # diffs = np.abs(np.diff(arr))
    diffs = np.abs(np.gradient(arr))
    # min_thresh = np.percentile(diffs, thresh * 10)
    # NOTE: controlling n_clusters seems to be key to sensitivity
    km = KMeans(8, copy_x=False).fit(diffs.reshape(-1, 1))
    min_thresh = np.sort(km.cluster_centers_.ravel())[0] * (1 - ADJUST)
    max_thresh = np.sort(km.cluster_centers_.ravel())[-1] * (1 + ADJUST)
    min_idx = np.where(diffs <= min_thresh)[0]
    # max_idx = min_idx + 1
    idx = np.array(list(set(min_idx).union(max_idx)), dtype=int)
    if len(idx) > 2:
        idx = np.sort(idx[(idx > 0) & (idx < len(arr))])
        if idx[0] == 0:  # prevent extrapolation of first point
            idx = idx[1:]
        if idx[-1] == (len(arr) - 1):
            idx = idx[:-1]
        plateau_idx = np.zeros_like(arr, dtype=bool)
        plateau_idx[idx] = True
        interp_idx(arr, hours, plateau_idx)

    # shrink spikes
    # diffs = np.abs(np.diff(normed))  # recalc since plateu removal
    # diffs = np.abs(np.diff(arr))  # recalc since plateu removal
    diffs = np.abs(np.gradient(arr))  # recalc since plateu removal
    # km = KMeans(2, copy_x=False).fit(diffs.reshape(-1, 1))
    # max_thresh = np.sort(km.cluster_centers_.ravel())[1] * (1 + ADJUST)
    # max_thresh = np.percentile(diffs, 100 - thresh)
    min_idx = np.where(diffs >= max_thresh)[0]
    max_idx = min_idx + 1
    idx = np.array(list(set(min_idx).union(max_idx)), dtype=int)
    if len(idx) > 2:
        idx = np.sort(idx[(idx > 0) & (idx < len(arr))])
        if idx[0] == 0:
            idx = idx[1:]
        if idx[-1] == (len(arr) - 1):
            idx = idx[:-1]
        spike_idx = np.zeros_like(arr, dtype=bool)
        spike_idx[idx] = True
        interp_idx(arr, hours, spike_idx)

    # finally hard clip by replacing spikes with local median
    low, high = np.percentile(arr, [thresh, 100 - thresh])
    idx = np.sort(np.where((arr >= high) | (arr <= low))[0])
    if len(idx) > 2:
        if idx[0] == 0:
            idx = idx[1:]
        if idx[-1] == (len(arr) - 1):
            idx = idx[:-1]
        abs_idx = np.zeros_like(arr, dtype=bool)
        abs_idx[idx] = True
        interp_idx(arr, hours, abs_idx)

    # shrink in small area around spikes
    # surround = 0
    # for i in range(-surround, surround + 1, 1):
    #     for idx in idxs:
    #         all_idx = all_idx.union(idx + i)
    # arr[spike_idx] = softsign(arr[spike_idx])
    # arr[np.abs(diffs) >= diff] = med
    # arr[spike_idx] = med + softsign(arr[spike_idx] - med) / 2


def highpass_spike_remove(
    raw: ndarray, decimation: int, period_cutoff_s: float = SPIKE_PERIOD_CUTOFF_S
) -> ndarray:
    t = np.arange(len(raw), dtype=raw.dtype)
    w = sosfilter(raw, decimation, period_cutoff_s=period_cutoff_s, lowpass=False)
    diffs = np.abs(w)

    # use sorting to get cluster boundaries and show on hist
    thresh = threshold_yen(diffs, nbins=1024)
    peak_idx = diffs >= thresh
    if np.all(~peak_idx):  # no peaks identified
        return raw
    fit_idx = np.copy(~peak_idx)
    fit_idx[0] = fit_idx[-1] = True  # need edges...
    sort_idx = np.argsort(t[fit_idx])
    try:
        interp = PchipInterpolator(t[fit_idx][sort_idx], raw[fit_idx][sort_idx])
        replace = interp(t[peak_idx])
        raw[peak_idx] = replace
        return raw
    except ValueError as e:
        traceback.print_exc()
        print(
            "\n\t^^^ Some bullshit about `x` not being strictly increasing, which should be impossible.\n"
        )
        t_wtf = t[fit_idx][sort_idx]
        d = np.diff(t_wtf, prepend=[t_wtf[0]])
        for i, diff in enumerate(d):
            if i == 0:
                continue
            if diff <= 0:
                print(f"Suspect timepoints: {t_wtf}")
                print(f"Suspect peak indices: {peak_idx}")
                print(f"Number of peaks: {np.sum(peak_idx)}")
                print(f"Problem at index {i} with diff values:")
                print(d[i - 3 : i + 3 + 1])
                print(f"Time at index {i}:")
                print(t[i - 3 : i + 3 + 1])
                raise RuntimeError from e
        raise RuntimeError("Impossible bullshit.")



def preview_highpass_spike_removal(waves: List[Any], decimation: int = 500) -> None:
    fig: plt.Figure
    ax: plt.Axes
    wave: Any
    for i, wave in tqdm(enumerate(waves), total=len(waves)):
        try:
            w = wave.values.numpy()
            t = wave.hours.numpy()
            raw = np.copy(w)

            sbn.set_style("darkgrid")
            cuts = [30, 60, 120, 300, 600, 1200]
            fig, axes = plt.subplots(nrows=4, ncols=len(cuts), sharex=False, sharey=False)
            for i, cut in tqdm(enumerate(cuts), total=len(cuts)):
                w = sosfilter(raw, decimation, period_cutoff_s=cut, lowpass=False)
                diffs = np.abs(w)

                # use sorting to get cluster boundaries and show on hist
                ax = axes[0, i]
                thresh = threshold_otsu(diffs, nbins=1024)
                thresh3 = threshold_yen(diffs, nbins=1024)
                counts = ax.hist(diffs, bins=1024, color="black", lw=0)[0]
                label = lambda s: s if i == 0 else None
                ax.vlines(thresh, ymin=0, ymax=np.max(counts), color="#00f068", label=label("otsu"))
                ax.vlines(thresh3, ymin=0, ymax=np.max(counts), color="#fba609", label=label("yen"))
                ax.set_yscale("log")

                # raw
                ax1 = axes[1, i]
                ax1.plot(t, raw, color="black", lw=0.5)
                ax1.plot(t, w, color="red", lw=0.5, alpha=0.5)
                ax1.set_title(f"lowpass period = {cut}", fontsize=8)

                # smoothed
                ax = axes[2, i]
                peak_idx = diffs >= thresh3
                ax.get_shared_x_axes().join(ax, ax1)
                ax.get_shared_y_axes().join(ax, ax1)
                ax.vlines(
                    x=t[peak_idx],
                    ymin=np.zeros_like(w[peak_idx]),
                    ymax=w[peak_idx],
                    color="black",
                    lw=0.5,
                )
                ax.scatter(
                    t[peak_idx], w[peak_idx], s=0.1, color="red", label="peaks" if i == 0 else None
                )
                ax.set_title(f"Peaks", fontsize=8)

                # now plot spike-removed wave
                ax = axes[3, i]
                fit_idx = np.copy(~peak_idx)
                fit_idx[0] = fit_idx[-1] = True  # need edges...
                # interp = PchipInterpolator(t[fit_idx], w[fit_idx])
                interp = PchipInterpolator(t[fit_idx], raw[fit_idx])
                replace = interp(t[peak_idx])
                cleaned = np.copy(raw)
                cleaned[peak_idx] = replace
                ax.plot(
                    t,
                    cleaned,
                    color="black",
                    lw=0.5,
                    label="Peaks removed" if i == 0 else None,
                )
                ax.get_shared_x_axes().join(ax, ax1)
                ax.get_shared_y_axes().join(ax, ax1)

            for ax in axes.ravel():
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.legend()
            fig.set_size_inches(w=24, h=12)
            fig.tight_layout()
            plt.show()
        except:
            traceback.print_exc()
            print(f"Problem for subject {i}")


def test_multipass_highpass_spike_removal(raw_waves: List[ndarray], decimation: int):
    fig: plt.Figure
    ax: plt.Axes
    wave: ndarray
    nrows, ncols = best_rect(len(raw_waves))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
    for i, (wave, ax) in tqdm(enumerate(zip(raw_waves, axes.flat)), total=len(raw_waves)):
        try:
            # if i < 29:
            #     continue

            label = lambda s: s if i == 0 else None
            t = np.arange(len(wave))
            raw = wave.copy()
            ax.plot(t, raw, color="black", lw=0.5, alpha=0.8, label=label("raw"))

            # one pass
            smoothed = highpass_spike_remove(wave, decimation, period_cutoff_s=1200.0)
            wave = smoothed
            ax.plot(t, smoothed, color="#0550ff", lw=0.5, alpha=0.8, label=label("1-pass"))

            # two pass
            smoothed = highpass_spike_remove(wave, decimation, period_cutoff_s=1200.0)
            wave = smoothed
            ax.plot(t, smoothed, color="#00c728", lw=0.5, alpha=0.8, label=label("2-pass"))

            # three pass
            smoothed = highpass_spike_remove(wave, decimation, period_cutoff_s=1200.0)
            ax.plot(t, smoothed, color="#fc4727", lw=0.5, alpha=0.8, label=label("3-pass"))

            ax.plot(t, smoothed - 6, color="#00c728", lw=0.5, alpha=0.8)
            ax.plot(t, np.abs(raw - smoothed) - 12, color="#0550ff", lw=0.5, alpha=0.8)
            if i == 6:
                ax.set_xlim(17.2, 18.0)
            if i == 7:
                ax.set_xlim(11.75, 11.82)
            if i == 13:
                ax.set_xlim(11.4, 12.2)
            if i == 16:
                ax.set_xlim(153.0, 153.45)
            if i == 25:
                ax.set_xlim(14.6, 14.9)
            if i == 27:
                ax.set_xlim(122.65, 122.80)
            if i == 29:
                ax.set_xlim(32.2, 32.6)
        except:
            traceback.print_exc()
            print(f"Problem for subject {i}")
    fig.legend()
    fig.set_size_inches(w=16, h=12)
    fig.tight_layout()
    plt.show()
