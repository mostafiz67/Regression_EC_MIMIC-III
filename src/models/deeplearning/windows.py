from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from math import ceil, floor
from pathlib import Path
from typing import Optional, Tuple

from pandas import DataFrame

from src.constants import BASE_FREQUENCY
from src.models.deeplearning.containers.lactate import Lactate
from src.preprocess.containers.wave import Wave


def period(decimation: int) -> float:
    """Return period in hours given the decimation. Helper."""
    return float(decimation / (BASE_FREQUENCY * 3600))


def get_wave_endpoints(
    wave: Wave,
    lactate: Lactate,
    predictor_window_size: int,
    lag_minutes: float,
    target_window_minutes: float,
    decimation: int,
    full_stats: bool = True,
) -> Optional[Tuple[int, int, Optional[DataFrame]]]:
    """
    Returns
    -------
    result: None | (S, F_lag, df)
        Returns None if there are no valid windows for the wave, otherwise returns (S, F_lag), where
        S is the first (Starting) fully-valid index into wave which is fully-interpolable, and where
        F_lag is the last (Final) fully-valid index into wave which is fully-interpolable. Also returns
        a DataFrame `df` with summary statistics of the wave values

    Notes
    -----
    See also tests/test_counting.py

    We have too many data points to just naively loop through and check if there is data,
    so we have to do some thinking.

    Example 1:

        S = Start, F = Finish (S, F are indices), T = wave period in hours

             x_start  y_start                           y_end    x_end
             |        |                                 |        |
             v        v                                 v        v
        __________________________________________________________________
        y             *                                 *
        x    . . . . . . . . . . . . . . . . . . . . . . . . . . .
        w_0            |   |
        w_f                                        |   |
        __________________________________________________________________
                       ^   ^                       ^   ^
                       |   |                       |   |
                       S   S + x_window_size       F   F + x_window_size

        - smallest possible value for S is ceil((y_start - x_start) / T)
        - largest possible value for F is floor((y_end - x_start) / PERIOD) - PRED_WINDOW_SIZE - 1
        - but we also require x[F + x_window_size] < x_end (F + x_window_size < len(x) - 1)
        - and we also require x[S] > x_start (S >= 0)

        - so we have:
            {S >= ceil((t_0 - x_start) / T)} AND {S >= 0}
            {F <= floor((t_f - x_start) / PERIOD) - PRED_WINDOW_SIZE - 1} AND { F < len(x) - PRED_WINDOW_SIZE - 1}  # noqa

            S = max(ceil((t_0 - x_start) / T), 0)
            F = min(floor((t_f - x_start) / PERIOD) - PRED_WINDOW_SIZE, len(x) - PRED_WINDOW_SIZE)

        - now also we have to factor in lag and duration of the target window

        TODO: diagram from notes


    Notes
    -----
    # On the exclusion of `include_previous_target_in_predictor` option

    Technically, if we drop the requirement to allow including previous target values in
    the predictor, then we can sometimes get much more usable windows. However, this also
    means the amount training data changes depending on the value of that hyperparameter,
    which then confounds analysis. For simplicity, we will construct windows *as if* it
    is required that inclusion of the previous target was always going to happen.
    """
    T = period(decimation)  # in hours
    W = predictor_window_size  # count
    W_hours = (W - 1) * T
    lag_hrs = lag_minutes / 60
    target_hrs = target_window_minutes / 60
    x_start, x_end = wave.hours_0, wave.hours_f
    y_start, y_end = lactate.hours[0], lactate.hours[-1]

    # Degenerate cases
    if x_end <= y_start or y_end <= x_start:  # no valid predictor windows
        return None
    if W > len(wave):  # predictor window larger than x
        return None
    if wave.hours_at(W - 1) > y_end:  # first window already off of y
        return None
    if y_start + W_hours > x_end:
        # first x window can only be as early as y_hours[0], but in this
        # case not enough room in x for window
        return None

    S = max(ceil((y_start - x_start) / T), 0)
    F = min(floor((y_end - x_start) / T) - W - 1, len(wave) - W - 1)
    if F < S:
        return None
    N = ceil((lag_hrs + target_hrs) / T)
    F_lag = F - N
    if F_lag < S:
        return None
    if full_stats:
        df = (
            DataFrame(wave.values[S : F_lag + 1])
            .describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.95, 0.99])
            .T
        )
    else:
        df = None
    return S, F_lag, df
