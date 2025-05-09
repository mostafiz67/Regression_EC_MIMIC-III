import gc
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy import interpolate, signal
from scipy.ndimage.filters import median_filter

from src.preprocess.rolling_clip import clip_to_vals


def get_splits(arr: ndarray, window_size: int) -> List[ndarray]:
    """Splits the wave into n chunks"""
    w_size = int(125 * 60 * 60 * window_size)
    n = len(arr) // w_size
    idx = [w_size * i for i in range(n + 1)]
    sub_arr = []
    for i in range(len(idx)):
        if i < len(idx) - 1:
            sub_arr.append(arr[idx[i] : idx[i + 1]])
        else:
            sub_arr.append(arr[idx[i] :])
    return sub_arr


def clip_percentiles(
    w_splits: List[ndarray], t_splits: List[ndarray], percentile: int = 5
) -> Tuple[ndarray, ndarray, ndarray]:
    """Implement rolling clip"""
    lows = np.empty(len(w_splits))
    highs = np.empty(len(w_splits))
    midpoints = np.empty(len(w_splits))

    for i in range(len(w_splits)):
        low_perc = np.percentile(w_splits[i], percentile)
        high_perc = np.percentile(w_splits[i], 100 - percentile)
        midpoint_idx = len(w_splits[i]) // 2
        midpoint = t_splits[i][midpoint_idx]
        lows[i] = low_perc
        highs[i] = high_perc
        midpoints[i] = midpoint

    return lows, highs, midpoints


def clip_wave(
    times: ndarray,
    wave: ndarray,
    copy_filtered_wave: ndarray,
    window_hrs: int = 1,
    percentile: int = 5,
) -> bool:
    """Approximate a rolling percentile clip with linear interpolation.

    Parameters
    ----------
    times: ndarray
        Time values of the raw wave

    wave: ndarray
        Modality [ABP] values

    copy_filtered_wave: ndarray
        Copy of filtered wave

    window_hrs: int
        Duration of clipping window

    percentile: int
        Percentile value to determine low and high percentile i.e. percentile , 100 - percentile

    Returns
    -------
    Returns True if headergroup has wave data more than twice the window size, see NOTES

    Notes
    -----
    The last header group can be really small and the function might not have
    enough wave data give a certain window size to interpolate. If this case we
    do not clip and skip the header group.

    """
    w_splits = get_splits(wave, window_size=window_hrs)
    t_splits = get_splits(times, window_size=window_hrs)

    lows, highs, midpoints = clip_percentiles(w_splits, t_splits, percentile)
    if len(midpoints) > 2:
        f = interpolate.interp1d(midpoints, lows, bounds_error=False, fill_value=0)
        f2 = interpolate.interp1d(midpoints, highs, bounds_error=False, fill_value=0)
        low_interp = f(times)
        high_interp = f2(times)
        clip_to_vals(copy_filtered_wave, low_interp, high_interp)

        return True
    return False


def filter_wave(wave: ndarray, sos: ndarray, med_window_size: int = 5) -> ndarray:
    """Applys median and butterworth filters to the raw clipped wave"""
    wave = np.clip(wave, 25, 175)
    smoothed = median_filter(wave, med_window_size)
    filtered_wave = np.array(signal.sosfiltfilt(sos, smoothed), dtype=np.float32)
    del smoothed
    gc.collect()
    return filtered_wave


def optimal_lag(
    t_wave: ndarray,
    lag: float,
    filtered_wave: ndarray,
    chart_interp: ndarray,
) -> float:
    """Calculates distance between wave and chart waves

    Parameters
    ----------
    t_wave: ndarray
        Time values of raw wave

    lag: float
        lag estimated value

    filtered_wave: ndarray
        Filtered wave values

    chart_interp: ndarray
        Interpolated chart values

    Returns
    -------
    distance: float
        Distance between wave and chart values
    """
    t_chart_lagged = t_wave + lag
    wave_start, wave_end = t_wave[0], t_wave[-1]
    chart_start, chart_end = t_chart_lagged[0], t_chart_lagged[-1]
    start = max(wave_start, chart_start)
    end = min(wave_end, chart_end)
    if start > end:
        return float("NaN")
    start_idx = np.where(t_wave > start)[0][0]
    end_idx = np.where(t_wave <= end)[0][-1]
    # NOTE: times == t_wave == t_chart due to interpolation
    t_shared = t_wave[start_idx:end_idx]
    t_shared_lag = t_chart_lagged[start_idx:end_idx]
    wave_shared = filtered_wave[start_idx:end_idx]
    chart_shared = chart_interp[start_idx:end_idx]

    # Need to handle the conditions if the chart wave ends before or after the ABP [modality] wave.
    if t_shared[-1] > t_shared_lag[-1]:  # If wave ends after the chart wave
        # To find the trailing index of the wave
        # NOTE: int(argmin(abs(arr - val))) just finds the first smallest index of arr less than val
        end_w_idx = int(np.argmin(np.abs(t_shared - t_shared_lag[-1])))
        wave_shared = wave_shared[: end_w_idx + 1]
        start_lag_idx = int(np.argmin(np.abs(t_shared_lag - t_shared[0])))
        chart_shared = chart_shared[start_lag_idx:]
    elif t_shared[-1] < t_shared_lag[-1]:  # If wave ends before the chart wave
        start_w_idx = int(np.argmin(np.abs(t_shared - t_shared_lag[0])))
        wave_shared = wave_shared[start_w_idx:]
        end_lag_idx = int(np.argmin(np.abs(t_shared_lag - t_shared[-1])))
        chart_shared = chart_shared[: end_lag_idx + 1]
    idx = chart_shared > 0
    distance = np.mean(np.abs(wave_shared[idx] - chart_shared[idx]))
    return float(distance)
