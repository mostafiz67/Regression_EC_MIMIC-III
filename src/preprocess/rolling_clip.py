from warnings import filterwarnings  # isort:skip

filterwarnings("ignore", category=DeprecationWarning)  # isort:skip

from typing import List, Sized

import numpy as np
from numba import jit, prange

# from numpy.typing import NDArray

FloatArray = np.ndarray


@jit(nopython=True, parallel=False, cache=True)
def clip(a: FloatArray, lo: np.float64, hi: np.float64) -> None:
    """Clip in place"""
    for i, val in enumerate(a):
        if val > hi:
            a[i] = hi
        if val < lo:
            a[i] = lo


@jit(nopython=True, parallel=True, cache=True)
def clip_to_vals(arr: FloatArray, lows: FloatArray, highs: FloatArray) -> None:
    """Clip in place"""
    for i in prange(len(arr)):
        if arr[i] > highs[i]:
            arr[i] = highs[i]
        if arr[i] < lows[i]:
            arr[i] = lows[i]


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def percentile_clip(
    a: FloatArray, window_size: int, perc: float = 5.0, padding_mode: str = "reflect"
) -> FloatArray:
    """Clip using rolling windows and percentiles p, 100 - p.

    Parameters
    ----------
    a: NDArray[np.float64]
        NumPy array of values to clip. Must be NumPy for Numba to work, not Series, DataFrame, or Python list.

    window_size: int
        Since of rolling window (number of timepoints).

    p: float
        Value in (0, 50) to use to compute compute clamp values. E.g. p=5 will use 5, 95 percentiles.

    padding_mode: Literal["reflect", "nearest", "median"]
        For exampling if we need to pad by 3 values (windows size 4):
        reflect: [1, 2, 3, 4, 5, 6, 7]  -->  [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        nearest: [1, 2, 3, 4, 5, 6, 7]  -->  [1, 1, 1, 1, 2, 3, 4, 5, 6, 7]
        median:  [1, 2, 3, 4, 5, 6, 7]  -->  [2.5, 2.5, 2.5, 1, 2, 3, 4, 5, 6, 7]  (2.5 = median([1, 2, 3, 4]))
    """
    # note we would strongly prefer to avoid copying the entire `a` when padding, so should instead
    # grab and access padding values from a or as a constant (e.g. if using median)
    p = window_size - 1  # pad length
    if padding_mode == "nearest":
        pads = np.full((p,), a[0], dtype=np.float64)
    elif padding_mode == "median":
        med = np.median(a[:window_size])
        pads = np.full((p,), med, dtype=np.float64)
    else:
        pads = a[p:0:-1]  # reflect by default

    clipped = np.empty(a.shape, dtype=np.float64)
    for i in prange(len(a)):
        if i < p:  # copies, but only for first few values
            if padding_mode == "nearest" or padding_mode == "median":
                window = np.empty((window_size,), dtype=np.float64)
                window[: p - i] = pads
                window[p - i :] = a[: i + 1]
            else:  # reflect by default
                window = np.concatenate((pads[i:], a[: i + 1]))
        else:  # no padding values to use
            window = np.copy(a[i - p : i - p + window_size])  # copy is necessary here
        lo, hi = np.percentile(window, [perc, 100 - perc])
        clip(window, lo, hi)
        clipped[i] = window[-1]
    return clipped


def slow_percentile_clip(a: Sized, w: int, p: float, padding_mode: str = "reflect") -> FloatArray:
    a_pad = np.pad(a, pad_width=(w - 1, 0), mode=padding_mode)
    clipped: List[float] = []
    for i in range(len(a)):
        window = a_pad[i : i + w]
        p_lo, p_hi = np.percentile(window, [p, 100 - p])
        clipped.append(np.clip(window, p_lo, p_hi)[-1])
    return np.array(clipped)


# We'll have you start with uncommenting the Numba decorator below and seeing
# what happens as we try to naively numba-ize the rolling percentile-clip
# @jit(nopython=True, parallel=True, cache=True, fastmath=True)
def naive_percentile_clip(a: np.ndarray, w: int, perc: float) -> FloatArray:
    # a_pad = np.pad(a, pad_width=(w - 1, 0), mode="reflect")
    p = w - 1  # pad length
    pads = np.full((p,), a[0])
    clipped = np.empty(a.shape)
    for i in range(len(a)):
        if i < p:
            window = np.empty((w,))
            window[: p - i] = pads
            window[p - i :] = a[: i + 1]
        else:
            window = a[i - p : i - p + w]
        low, high = np.percentile(a, [perc, 100 - perc])
        val = np.clip(window, low, high)[-1]
        clipped[i] = val
    # clipped: List[float] = []
    # for i in range(len(a)):
    #     window = np.copy(a_pad[i : i + w])
    #     p_lo, p_hi = np.percentile(window, [p, 100 - p])
    #     clipped.append(np.clip(window, p_lo, p_hi)[-1])
    return clipped


# if __name__ == "__main__":
#     a = np.array([1, 1, 5, 1, 1], dtype=np.float64)
#     result = naive_percentile_clip(a, w=2, perc=25)
