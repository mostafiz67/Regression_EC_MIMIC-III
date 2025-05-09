from math import ceil

import numpy as np
import pandas as pd


def test_endpoint() -> None:
    F = 125
    FREQ = f"{1/F}S"  # stupid way Pandas wants frequency input
    decimation = 10
    start1 = pd.to_datetime("21611209-175043")
    start2 = pd.to_datetime("21611210-175834")
    start3 = pd.to_datetime("21620110-185328")
    start = start1
    n1 = 10076798
    n2 = 805845
    n3 = 61149216

    # decimated counts
    dn1 = ceil(n1 / decimation) - 1
    dn2 = ceil(n2 / decimation) - 1
    dn3 = ceil(n3 / decimation) - 1

    # let's get endpoints for w1, w2, w3
    times1 = pd.date_range(start=start1, periods=n1, freq=FREQ)
    times2 = pd.date_range(start=start2, periods=n2, freq=FREQ)
    times3 = pd.date_range(start=start3, periods=n3, freq=FREQ)

    assert len(times1) == n1
    assert len(times2) == n2
    assert len(times3) == n3
    assert times1[0] == start1
    assert times2[0] == start2
    assert times3[0] == start3

    T_calc = np.mean(np.diff(times3)) * decimation

    # decimated endtimes in hours
    d_end1 = (times1[::decimation][-1] - start) / pd.Timedelta(hours=1)
    d_end2 = (times2[::decimation][-1] - start) / pd.Timedelta(hours=1)
    d_end3 = (times3[::decimation][-1] - start) / pd.Timedelta(hours=1)

    f_hrs = F * 3600
    f_dec_hrs = f_hrs / decimation
    T = 1 / f_dec_hrs

    end1 = ((start1 + pd.Timedelta(hours=T * dn1)) - start) / pd.Timedelta(hours=1)
    end2 = ((start2 + pd.Timedelta(hours=T * dn2)) - start) / pd.Timedelta(hours=1)
    end3 = ((start3 + pd.Timedelta(hours=T * dn3)) - start) / pd.Timedelta(hours=1)

    end1 = pd.Timedelta(hours=T * dn1) / pd.Timedelta(hours=1)

    print(f"Decimated end1 from Pandas: {d_end1}")
    print(f"Decimated end1 from period: {end1}")
    print(f"                     Diff1: {end1 - d_end1}")
    print(f"Decimated end2 from Pandas: {d_end2}")
    print(f"Decimated end2 from period: {end2}")
    print(f"                     Diff2: {end2 - d_end2}")
    print(f"Decimated end3 from Pandas: {d_end3}")
    print(f"Decimated end3 from period: {end3}")
    print(f"                     Diff3: {end3 - d_end3}")
    print(f" Calculated average period: {T_calc / pd.Timedelta(hours=1)}")


def test_time_at() -> None:
    F = 125
    FREQ = f"{1/F}S"  # stupid way Pandas wants frequency input
    decimation = 10
    start1 = pd.to_datetime("21611209-175043")
    start2 = pd.to_datetime("21611210-175834")
    start3 = pd.to_datetime("21620110-185328")
    start = start1
    n1 = 10076798
    n2 = 805845
    n3 = 61149216

    # decimated counts
    dn1 = ceil(n1 / decimation) - 1
    dn2 = ceil(n2 / decimation) - 1
    dn3 = ceil(n3 / decimation) - 1

    # let's get endpoints for w1, w2, w3
    times1 = pd.date_range(start=start1, periods=n1, freq=FREQ)
    times2 = pd.date_range(start=start2, periods=n2, freq=FREQ)
    times3 = pd.date_range(start=start3, periods=n3, freq=FREQ)

    T_calc = np.mean(np.diff(times3)) * decimation / pd.Timedelta(hours=1)

    # decimated times in hours
    dt1 = (times1[::decimation] - start) / pd.Timedelta(hours=1)
    dt2 = (times2[::decimation] - start) / pd.Timedelta(hours=1)
    dt3 = (times3[::decimation] - start) / pd.Timedelta(hours=1)

    f_hrs = F * 3600
    f_dec_hrs = f_hrs / decimation
    T = 1 / f_dec_hrs
    for i in np.random.randint(0, max(dn1, dn2, dn3), 50000):
        t1 = float(pd.Timedelta(hours=T * i) / pd.Timedelta(hours=1)) + dt1[0]
        t2 = float(pd.Timedelta(hours=T * i) / pd.Timedelta(hours=1)) + dt2[0]
        t3 = float(pd.Timedelta(hours=T * i) / pd.Timedelta(hours=1)) + dt3[0]
        if i < dn1:
            assert t1 - dt1[i] < T_calc / 100
            assert t1 - dt1[i] < 1e-10
        if i < dn2:
            assert t2 - dt2[i] < T_calc / 100
            assert t2 - dt2[i] < 1e-10
        if i < dn3:
            assert t3 - dt3[i] < T_calc / 100
            assert t3 - dt3[i] < 1e-10


if __name__ == "__main__":
    # test_endpoint()
    test_time_at()
