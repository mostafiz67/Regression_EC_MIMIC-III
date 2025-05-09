import numpy as np
from numpy import ndarray

from scripts.metrics import nearest_next


def test_next_distances() -> None:
    # interpolated_lact_times
    interps = [
        # simple sanity tests
        np.asarray([1, 2, 3, 4, 5]),
        np.asarray([1, 2, 3, 4, 5, 6]),
        # less values non_interps
        np.asarray([1, 2, 3, 4, 5]),
        np.asarray([1, 2, 3, 4, 5]),
        # less valyes in interps
        np.asarray([1]),
        np.asarray([1, 1]),
        # duplicates
        np.asarray([1, 1, 2, 3, 4, 5, 6]),
        # duplicates in interps and non_interps
        np.asarray([1, 1, 1, 2, 2, 2]),
        # negative
        np.asarray([-1, 1, 2, 3, 4, 5, 6]),
        np.asarray([-1, -1]),
        # checking nan value
        np.asarray([np.nan, 1, 1]),
    ]
    # non-interpolated_lact_times
    non_interps = [
        np.asarray([1, 3, 5]),
        np.asarray([1, 3, 5]),
        np.asarray([1]),
        np.asarray([5]),
        np.asarray([1, 3, 5]),
        np.asarray([1, 3, 5]),
        np.asarray([1, 3, 5]),
        np.asarray([1, 1, 2, 2, 2]),
        np.asarray([1, 3, 5]),
        np.asarray([1, 2, 3, 4, 5, 6, 7]),
        np.asarray([1, 3, 5]),
    ]
    correct_vals = [
        np.asarray([0, 1, 0, 1, 0]),
        np.asarray([0, 1, 0, 1, 0, np.nan]),
        np.asarray([0, np.nan, np.nan, np.nan, np.nan]),
        np.asarray([4, 3, 2, 1, 0]),
        np.asarray([0]),
        np.asarray([0, 0]),
        np.asarray([0, 0, 1, 0, 1, 0, np.nan]),
        np.asarray([0, 0, 0, 0, 0, 0]),
        np.asarray([2, 0, 1, 0, 1, 0, np.nan]),
        np.asarray([2, 2]),
        np.asarray([np.nan, np.nan, np.nan]),
    ]

    for a, b, c in zip(interps, non_interps, correct_vals):
        val = nearest_next(a, b)
        np.testing.assert_array_equal(c, val)


def correct_increasing_trend(interpolated_dist: ndarray) -> bool:
    count = 0
    for i in range(len(interpolated_dist) - 1):
        if interpolated_dist[i] or interpolated_dist[i + 1]:
            continue
        val = interpolated_dist[i] - interpolated_dist[i + 1]
        if val > 0:
            count += 1
            if count > 2:
                return False
        else:
            count = 0
    return True


def verify(interps: ndarray, non_interps: ndarray, interpolated_dist: ndarray) -> None:
    # check if array length of interpolated_dist is equal to len of interps
    assert len(interpolated_dist) == len(interps)
    # check if number of nans in interpolated_dist is equal to
    # number of values in interps greater than max value in non_interps
    assert (interps > non_interps[-1]).sum() == np.isnan(interpolated_dist).sum()
    # check if the lowest value of interpolated 0 or greater than 0
    if np.isnan(interpolated_dist).all():
        print("All elements are nan")
        return None
    assert np.nanmin(interpolated_dist) >= 0, f"Minimum: {np.min(interpolated_dist)}"
    # check the greatest value in interpolated dist
    # if interps[0] = -1, non_interps[-1] = 10 then max(interpolated_dist) < 11
    # as we are calculating absolute distance
    # if interps[0] = 1, non_interps[-1] = 10 then max(interpolated_dist) < 9
    if interps[0] < 0:
        assert np.nanmax(interpolated_dist) <= (non_interps[-1] + np.abs(interps[0]))
    else:
        assert np.nanmax(interpolated_dist) <= (non_interps[-1] - interps[0])
    # check if the maximum increasing trend in values is not more than 2
    assert bool(correct_increasing_trend(interpolated_dist)) == True, "Increasing trend more than 2"


def test_random_values() -> None:
    for _ in range(1000):
        N_INTERPS = np.random.randint(10, 1000)
        N_NON_INTERPS = np.random.randint(1, 17)
        INTERP_MIN, INTERP_MAX = np.sort(np.random.randint(-1000, 1000, [2]))
        NON_INTERP_MIN, NON_INTERP_MAX = np.sort(
            np.random.randint(-1000, 1000, [2])
        )  # may need to adjust depending on what is valid

        interps = np.sort(
            np.random.uniform(INTERP_MIN, INTERP_MAX, size=N_INTERPS)
        )  # uniform! floats!
        non_interps = np.sort(
            np.random.uniform(NON_INTERP_MIN, NON_INTERP_MAX, size=N_NON_INTERPS)
        )  # also floats
        interpolated_dist = nearest_next(interps, non_interps)
        verify(interps, non_interps, interpolated_dist)
