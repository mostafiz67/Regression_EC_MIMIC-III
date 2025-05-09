from math import ceil, floor
from typing import Optional

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from typing_extensions import Literal


def test_endpoints(capsys: CaptureFixture) -> None:
    N = 20000
    for _ in range(N):
        verify_window_endpoints()
    with pytest.raises(AssertionError):
        for _ in range(N):
            try:
                verify_window_endpoints(muck_with="S")
                verify_window_endpoints(muck_with="F")
                verify_window_endpoints(muck_with="F_lag")
            except IndexError:
                raise AssertionError


def script_test_endpoints() -> None:
    for _ in range(1000):
        verify_window_endpoints()


def verify_window_endpoints(muck_with: Optional[Literal["S", "F", "F_lag"]] = None) -> None:
    """Comp start and end indices of first and last usable windows and verify these
    indices are in fact correct

    Parameters
    ----------
    muck_with: Optional[Literal["S", "F", "F_lag"]] = None
        Computed results to perturb by 1 to check for off-by-one errors. Perturbs by
        one in the direction that would gain us a window, e.g. decreases S by 1, or
        increases F or F_lag by 1.

    Notes
    -----
    The main challenge here is just holding all the names in your head, and unfortunately long /
    descriptive names make it worse, so we have to choose some short names. The main reason there
    are so many names is every type of variable (predictor, target) has three values of interest
    (indices, timepoints, values) [at 6 so far], a window (2 more, so at 12 values to name), a first
    window, last window, and starts and ends and etc. It's a mess.

    In any case, the GOAL here is to compute just the two values (indices) that allow us to compute
    everything we need from target inputs. I call them "F" and "S" for "start" and F for "finish"
    values of interest.

    The only indices we need are indices into the predictor (wave) data that allow accessing windows
    that can be used for supervised learning. Because in our case our regression target has a
    distance (lag) from the input, and because it itself has a duration, and we want to be able to
    tune all of these, not all windows that can be formed on the predictor are actually usable (e.g.
    not all predictor windows have sufficient data to predict). But assuming there is at least *one*
    usable window, then there is an index where the first usable window starts. This is "S".

    Likewise, exlcuding the cases where there is just one window, there is an index where the *last*
    usable window starts. This is "F".

    Other variables:

        x: predictor times
        y: target times


    For every possible predictor window x_window, there are two main conditions to check:

    1. x_window is in y
       - i.e. y[0] <= x_window[0] < x_window[-1] <= y[-1]
    2. The lagged y_window for x_window is in y
       - i.e. y[0] <=     y_window[0]    <=             y_window[-1]        <= y[-1]
       - i.e. y[0] <= x_window[-1] + lag <= x_window[-1] + lag + y_duration <= y[-1]

    We could start with S = 0, F = len(x), and check, incrementing / decrementing by one until we
    find the smallest S and largest F that satisfies the above conditions. But we have millions of
    points and this could become *very* slow.

    However, because we know the sampling frequency (after decimation) of x, we can just do some
    math and figure out exactly what these values are. Roughly, the key insight is that we can go
    from times back to indices by division by the period (or multiplication by the frequency) and
    then taking either a floor or ceil, as appropriate. Then it is just a matter of converting
    distances back to indices, and vice versa.
    """
    # use minutes and similar values to what we expect to start, but always
    # work with hours finally
    BASE_FREQ = 125
    DECIMATION = 25
    T = PERIOD = (DECIMATION / BASE_FREQ) / 3600  # hours
    LAG_MINUTES = np.random.uniform(0, 120)

    LAG_HRS = LAG_MINUTES / 60
    LAG_N_PERIODS = ceil(LAG_HRS / T)
    DESIRED_PRED_WINDOW_MINUTES = np.random.uniform(5, 120)
    W = PRED_WINDOW_SIZE = int(DESIRED_PRED_WINDOW_MINUTES * (BASE_FREQ / DECIMATION) * 60)
    PRED_WINDOW_MINUTES = PRED_WINDOW_SIZE * (PERIOD / 60)
    PRED_WINDOW_HOURS = PRED_WINDOW_SIZE * (PERIOD / 3600)
    Y_WIN_MINUTES = np.random.uniform(0.05, 120)  # no need to convert
    Y_WIN_HRS = Y_WIN_MINUTES / 60

    X_START = np.random.randint(0, 30)
    X_END = np.random.randint(30 + PERIOD, 60)
    X_COUNT = ceil((X_END - X_START) / PERIOD)

    # just want variable y_hours with some random but predictable structure
    noise = np.random.uniform(-30, 30, [3]) / np.array([2, 10, 1])
    y = np.sort(np.array([15, 30, 40]) + noise).round(1)  # in hours
    x = np.linspace(X_START, X_END, X_COUNT)

    # Degenerate cases
    if y[-1] <= x[0]:  # no valid predictor windows
        return
    if W > len(x):  # predictor window larger than x
        return
    if x[W - 1] > y[-1]:  # first window already off of y
        return
    if y[0] + PRED_WINDOW_HOURS > x[-1]:  #
        # first x window can only be as early as y_hours[0], but in this
        # case not enough room in x for window
        return

    # More interesting degenerate cases:
    #
    # First window already past y:
    """
        predictor window size: W = 17049 (0.02 min)
        lag: 16.4 minutes
        y: [3.0, 16.2, 28.8]
        x: [28.0, ..., 43.0] (len(x) = 270000)
        S = 0 (start index for first valid predictor window)
        F = -2650 (start index for last valid predictor window)
    """

    # S + W already past *x* (or F < S), i.e. y_start + PRED_WINDOW_HOURS > x_end
    """
        predictor window size: W = 30733 (0.03 min)
        lag: 10.0 minutes
        y: [29.9, 32.8, 33.7]
        x: [17.0, ..., 30.0] (len(x) = 234000)
        S = 232200 (start index for first valid predictor window)
        F = 203266 (start index for last valid predictor window)
        x[S]: 29.90005512844072 (S = 232200)
        IndexError: index 262932 is out of bounds for axis 0 with size 234000

        F < S only possible if S = ceil((y_start - x_start) / T)
    """

    # F < S again, (S + W > len(x))
    """
        predictor window size: W = 30750 (0.03 min)
        lag: 1.7 minutes
        y: [27.6, 27.7, 28.5]
        x: [3.0, ..., 36.0] (len(x) = 594000)
        S = 442800 (start index for first valid predictor window)
        F = 428249 (start index for last valid predictor window)
        x[S]: 27.600041414211137 (S = 442800)
        x[S:S + W] endpoint: 29.30832206788227 (y[-1] = 28.5)
        x[F]: 26.7916511643959 (F = 428249)
        x[F:F + W] endpoint: 28.499931818067036 (y[-1] = 28.5)
        FAILED tests/test_counting.py::test_endpoints - assert 29.308377623531353 <= 28.5
    """

    y_start, y_end = y[0], y[-1]
    x_start, x_end = x[0], x[-1]

    """
    # first (potentially) valid predictor index if including previous target
    if y_start <= x_start:
        S = x_min_valid_start_idx = 0
    else:
        S = x_min_valid_start_idx = ceil((y_start - x_start) / PERIOD)
    F = x_max_valid_end_idx = floor((y_end - x_start) / PERIOD) - PRED_WINDOW_SIZE

    max_n_windows = floor((y_end - y_start) / PRED_WINDOW_HOURS)
    """

    S = max(ceil((y_start - x_start) / T), 0)
    if muck_with == "S":
        S -= 1
    F = min(
        floor((y_end - x_start) / PERIOD) - PRED_WINDOW_SIZE - 1,
        len(x) - PRED_WINDOW_SIZE - 1,
    )
    if muck_with == "F":
        F += 1
    if F < S:  # handles "sliver" cases where x barely overlaps y
        return
    # now also adjust for lag issues:
    #
    # From S to F defines valid predictor windows which we can supplement with
    # past target data. But of these S to F windows, only a subset again has
    # interpolable paired target windows given the specified LAG_HRS and Y_WIN_HRS
    #
    N = ceil((LAG_HRS + Y_WIN_HRS) / T)
    F_lag = F - N
    if muck_with == "F_lag":
        F_lag += 1
    if F_lag < S:
        return  # not enough data

    print("\n")
    print(f"predictor window size: W = {W} ({np.round(PRED_WINDOW_MINUTES * 60, 1)} sec)")
    print(f"lag: {np.round(LAG_MINUTES, 1)} minutes")
    print(f"y: [{y[0]}, {y[1]}, {y[-1]}]")
    print(f"x: [{x[0]}, ..., {x[-1]}] (len(x) = {len(x)})")
    print(f"    S = {S} (start index for first valid predictor window)")
    print(f"    F = {F} (start index for last valid predictor window)")
    print(
        f"F_lag = {F_lag} (start index for last valid predictor window when considering lagged target)"
    )
    # first window
    print(f"x[S]: {x[S]}")
    print(f"x[S:S + W] endpoint: {x[S + W - 1]:.3f} (y = [{y_start}, {y_end}])")
    print(f"x[S + W - 1] + lag: {x[S + W - 1] + LAG_HRS:.3f} (y = [{y_start}, {y_end}])")
    # Starting x window in interpolable zone:
    assert y_start <= x[S] <= y_end
    assert y_start <= x[S + W] <= y_end
    assert y_start <= x[S + W] + LAG_HRS <= y_end
    assert y_start <= x[S + W] + LAG_HRS + Y_WIN_HRS <= y_end

    print(f"x[F]: {x[F]}")
    print(f"x[F:F + W] endpoint: {x[F + W - 1]:.3f} (y = [{y_start}, {y_end}])")
    print(f"x[F + W - 1] + lag: {x[F + W - 1] + LAG_HRS:.5f} (y = [{y_start}, {y_end}])")
    print(
        f"x[F + W - 1] + lag + y_win_duration: {x[F + W - 1] + LAG_HRS + Y_WIN_HRS:.5f} (y = [{y_start}, {y_end}])"
    )
    print(
        f"x[F_lag + W - 1] + lag + y_win_duration: {x[F_lag + W - 1] + LAG_HRS + Y_WIN_HRS:.5f} (y = [{y_start}, {y_end}])"
    )
    # ending x window in interpolable zone:
    assert y_start <= x[F] <= y_end, "F"
    assert y_start <= x[F + W] <= y_end, "F + W"

    assert y_start <= x[F_lag] <= y_end, "F_lag"
    assert y_start <= x[F_lag + W] <= y_end, "F_lag + W"
    assert y_start <= x[F_lag + W] + LAG_HRS <= y_end, "F_lag + W + LAG"
    assert y_start <= x[F_lag + W - 1] + LAG_HRS + Y_WIN_HRS <= y_end, "F_lag + W + LAG + Y_WIN"

    # now we need to check we are not off by one...

    assert x[F] >= y_start
    if y_start > x_start:
        diff = y_start - (x[F] - PERIOD)
        np.testing.assert_array_less(diff, 1e-4)  # can fail due to rounding / floating point

    # Now check with lags for target windows!
    # Starting y window in interpolable zone:
    assert y_start <= x[S + W] + LAG_HRS <= y_end


if __name__ == "__main__":
    script_test_endpoints()
