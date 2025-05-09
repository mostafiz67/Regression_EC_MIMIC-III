# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()  # isort: skip
# fmt: on

import traceback
from pathlib import Path
from shutil import copyfile
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate, resample_poly
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import FULL_DATA, PREDECIMATED_DATA
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.utils import (
    remove_border_nans,
    remove_internal_nans,
    split_at_nans,
    window_dimensions,
)

"""
If we want this to work, need *predecimation* and *presmoothing* (foolish to decimate without
pre-smoothing).

Pre-smoothing should be simple, and not require global info: e.g. front-padded uniform filter
(unform moving average), front-padded median filter. Still, *technically* if pre-smoothing
makes use of a window size that wouldn't be available in realtime processing we are cheating.

We know we will generally want to deal with predictor windows of >30 minutes. That is, the smallest
"allowed" window size to use in filtering is one of 30 minutes, i.e. 30min * 60 s/min * 125Hz = 225
000 points. This is a simplification as obviously we could reprocess, realtime, with larger windows
as data streams in, but for now, this is a precaution.

We only want to smooth to reduce noise relative to the decimation factor, i.e, retain "max
information" in the smoothed + decimated data. Likely, we want to wisely use `scipy.signal.decimate`.

"""

MODALITY = "ABP"
DROP_BORDER_NANS = True


def compare_methods() -> None:
    """We will just stick with scipy.signal.decimate with FIR"""
    SID = "p000188"
    DECIMATION = 1
    X_MINUTES = 60

    subject: DeepSubject = DeepSubject.initialize_from_sids(
        source=FULL_DATA,
        sids=[SID],
        predictor_window_size=window_dimensions(X_MINUTES, DECIMATION)[0],
        lag_minutes=0,
        target_window_minutes=24 * 60,
        decimation=DECIMATION,
        predecimated=False,
        interp_method=InterpMethod.previous,
    )[0]
    wave = subject.waves[0].values.numpy()
    hrs = subject.waves[0].hours.numpy()

    dec = [5, 25, 125, 250, 500, 1000]
    fig, axes = plt.subplots(ncols=1 + len(dec), nrows=3, sharex=True, sharey=True)

    axes[0][0].plot(hrs, wave, color="black", lw=0.5)
    axes[0][0].set_ylabel("`numpy [::d]`")
    for i, (d) in tqdm(enumerate(dec), total=len(dec), desc="Plotting"):
        axes[1][i + 1].plot(hrs[::d], wave[::d], color="black", lw=0.5, label=f"dec = {d}")
        axes[1][i + 1].legend().set_visible(True)

    axes[1][0].plot(hrs, wave, color="black", lw=0.5)
    axes[1][0].set_ylabel("`scipy.signal.decimate`")
    for i, (d) in tqdm(enumerate(dec), total=len(dec), desc="Plotting"):
        smooth = decimate(wave, d, ftype="fir")
        axes[0][i + 1].plot(hrs[::d], smooth, color="black", lw=0.5, label=f"dec = {d}")
        axes[0][i + 1].legend().set_visible(True)

    axes[2][0].plot(hrs, wave, color="black", lw=0.5)
    axes[2][0].set_ylabel("`resample_poly`")
    for i, (d) in tqdm(enumerate(dec), total=len(dec), desc="Plotting"):
        smooth = resample_poly(wave, up=2, down=d * 2)
        axes[2][i + 1].plot(hrs[::d], smooth, color="black", lw=0.5, label=f"dec = {d}")
        axes[2][i + 1].legend().set_visible(True)

    fig.set_size_inches(w=18, h=16)
    plt.show()


def predecimate(subject: DeepSubject) -> None:
    """Splits data in NaN-free chunks, decimates each chunk, and again
    removes any border NaNs induced by interpolation. Then also removes
    internal NaN values with
    """
    try:
        for decimation in [5, 25, 125, 250, 500, 1000]:
            dec_outdir = PREDECIMATED_DATA / f"decimation_{decimation}"
            if not dec_outdir.exists():
                dec_outdir.mkdir(exist_ok=True, parents=True)
            subj_outdir = dec_outdir / subject.sid
            wave_dir = subj_outdir / "waves"
            if not subj_outdir.exists():
                subj_outdir.mkdir(exist_ok=True)
                wave_dir.mkdir(exist_ok=True)
            lact_path = subject.lact_path
            copyfile(lact_path, subj_outdir / lact_path.name)

        for w, wave_path in enumerate(subject.wave_paths):
            full_wave = pd.read_parquet(wave_path)
            waves, starts = split_at_nans(full_wave, wave_path)
            for wave, start in zip(waves, starts):
                for decimation in [5, 25, 125, 250, 500, 1000]:
                    dec = resample_poly(
                        wave.astype(np.float64), up=2, down=decimation * 2, padtype="line"
                    )
                    dec_outdir = PREDECIMATED_DATA / f"decimation_{decimation}"
                    subj_outdir = dec_outdir / subject.sid
                    wave_dir = subj_outdir / "waves"
                    df = pd.DataFrame(dec, columns=[MODALITY], dtype=np.float32)
                    df, start = remove_border_nans(df, start, decimation)
                    df = remove_internal_nans(df)
                    outfile = wave_dir / f"{start}.parquet"
                    df.to_parquet(outfile)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}. Details above.")


if __name__ == "__main__":
    # compare_methods()
    # sys.exit()
    # choose small defaults because we want most usable subjects
    DECIMATION = 1
    X_MINUTES = 10
    TARGET_MINUTES = 30

    subjects: List[DeepSubject] = DeepSubject.initialize_from_sids(
        source=FULL_DATA,
        sids=None,
        predictor_window_size=window_dimensions(X_MINUTES, DECIMATION)[0],
        lag_minutes=0,
        target_window_minutes=TARGET_MINUTES,
        decimation=DECIMATION,
        predecimated=False,
        interp_method=InterpMethod.previous,
    )
    # Should be done in about 20 minutes on Niagara with below
    process_map(predecimate, subjects, max_workers=20)
