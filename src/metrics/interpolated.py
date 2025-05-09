import numpy as np
from numba import jit
from numpy import ndarray

# # interpolated_lact_times
# input_lact_times = np.asarray([10, 11, 11, 12, 19, 21, 22, 29, 31, 32, 33])
# # non-interpolated_lact_times
# rescaled_lact_times = np.asarray([10, 20, 30, 40])


@jit(nopython=True)
def nearest_next(input_lact_times: ndarray, rescaled_lact_times: ndarray) -> ndarray:
    interpolated_dist = np.full_like(input_lact_times, np.nan, dtype=np.float64)
    i = 0
    j = 0

    while j < len(rescaled_lact_times):
        if input_lact_times[i] <= rescaled_lact_times[j]:
            interpolated_dist[i] = np.abs(input_lact_times[i] - rescaled_lact_times[j])
            if i < len(input_lact_times) - 1:
                i += 1
            else:
                break
        else:
            j += 1
    return interpolated_dist


@jit(nopython=True, parallel=True)
def nearest_previous(input_lact_times: ndarray, rescaled_lact_times: ndarray) -> ndarray:
    interpolated_dist = np.full_like(input_lact_times, np.nan)
    j = 0

    for i in range(len(input_lact_times)):
        if input_lact_times[i] > rescaled_lact_times[j + 1]:
            j += 1
        interpolated_dist[i] = np.abs(input_lact_times[i] - rescaled_lact_times[j])
    return interpolated_dist


@jit(nopython=True, parallel=True)
def minimal(input_lact_times: ndarray, rescaled_lact_times: ndarray) -> ndarray:
    interpolated_dist = np.full_like(input_lact_times, np.nan)
    j = 0

    for i in range(len(input_lact_times)):
        if input_lact_times[i] > rescaled_lact_times[j + 1]:
            j += 1
        temp1 = np.abs(input_lact_times[i] - rescaled_lact_times[j])
        temp2 = np.abs(input_lact_times[i] - rescaled_lact_times[j + 1])
        if temp1 < temp2:
            interpolated_dist[i] = temp1
        elif temp1 > temp2:
            interpolated_dist[i] = temp2
        else:
            interpolated_dist[i] = temp1
    return interpolated_dist
